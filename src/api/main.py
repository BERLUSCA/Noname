"""FastAPI app: quiz creation, answer scoring, challenges, leaderboard, chat agent.

Scoring is code, never the LLM. Every paid call is counted by the session budget.
"""

import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path

from datapizza.core.clients import Client
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agent.chat import run_chat
from src.agent.memory import WINDOW_SIZE, render_history
from src.agent.tools import ChatContext
from src.api.db import Database
from src.observability.budget import BudgetExceeded, SessionBudget
from src.observability.logger import CallLogger
from src.quiz.generator import QuizGenerator
from src.quiz.scoring import Answer, ScoreResult, score_answers
from src.quiz.verified import NotEnoughVerifiedQuestions, VerificationUnavailable, generate_verified
from src.rag.index import InsufficientMaterial, normalize_title
from src.rag.fingerprint import IndexMismatch
from src.rag.service import TitleRetriever, load_retriever_from_env
from src.rag.verifier import FactVerifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "app.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"


class QuizRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    n_questions: int = Field(default=3, ge=1, le=10)


class AnswerRequest(BaseModel):
    quiz_id: str
    player: str = Field(min_length=1, max_length=40)
    answers: list[int | None]


class ChallengeAnswerRequest(BaseModel):
    player: str = Field(min_length=1, max_length=40)
    answers: list[int | None]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    # Omit to start a new session; reuse the returned id to continue the conversation.
    session_id: str | None = Field(default=None, min_length=1, max_length=64)


SESSION_TTL_HOURS = 24  # chat history is deleted after this many hours (privacy)


def _winner(player_a: str, points_a: int, player_b: str, points_b: int) -> str | None:
    if points_a == points_b:
        return None  # draw
    return player_a if points_a > points_b else player_b


def create_app(
    db: Database,
    generator_factory: Callable[[], QuizGenerator],
    retriever_factory: Callable[[], TitleRetriever],
    verifier_factory: Callable[[], FactVerifier],
    chat_client_factory: Callable[[], Client] | None = None,
    call_logger: CallLogger | None = None,
    budget: SessionBudget | None = None,
) -> FastAPI:
    app = FastAPI(title="Quanto ne sai di film e serie TV?")
    state: dict[str, object] = {}

    # Components are built on first use, so the app starts (and tests run) without API keys.
    def get_generator() -> QuizGenerator:
        if "generator" not in state:
            state["generator"] = generator_factory()
        return state["generator"]

    def get_retriever() -> TitleRetriever:
        if "retriever" not in state:
            try:
                state["retriever"] = retriever_factory()
            except IndexMismatch as exc:
                # The index does not match the corpus or the model: refuse, do not serve stale answers.
                raise HTTPException(status_code=503, detail=f"Indice non allineato: {exc}") from exc
        return state["retriever"]

    def get_verifier() -> FactVerifier:
        if "verifier" not in state:
            state["verifier"] = verifier_factory()
        return state["verifier"]

    def check_budget() -> None:
        # Fail closed: refuse paid calls when the session budget is exhausted or not configured.
        if budget is None:
            return
        try:
            budget.ensure_available()
        except BudgetExceeded as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def score_and_save(quiz_id: str, player: str, answers: list[int | None]) -> ScoreResult:
        stored = db.get_questions(quiz_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Quiz non trovato")
        if db.has_attempt(quiz_id, player):
            raise HTTPException(status_code=409, detail="Il giocatore ha già risposto a questo quiz")
        try:
            result = score_answers(stored, [Answer(choice) for choice in answers])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.save_attempt(quiz_id, player, result.points, result.correct, result.total)
        return result

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        # Test page for the backend (static HTML, no build). Interactive docs are at /docs.
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/quiz")
    def create_quiz(body: QuizRequest) -> dict:
        # The passage comes from the corpus, never from the client. Source = URL of the best chunk.
        check_budget()
        title_key = normalize_title(body.title)
        try:
            chunks = get_retriever().retrieve(body.title)
        except InsufficientMaterial as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        passage = "\n".join(chunk.text for chunk in chunks)
        try:
            questions = generate_verified(
                get_generator(), get_verifier(), body.title, passage, chunks[0].url, body.n_questions,
                avoid=db.recent_questions(title_key),
            )
        except VerificationUnavailable as exc:
            raise HTTPException(status_code=503, detail="Verifica delle domande non disponibile") from exc
        except NotEnoughVerifiedQuestions as exc:
            raise HTTPException(status_code=503, detail="Non è stato possibile generare domande verificate, riprova") from exc
        quiz_id = db.save_quiz(body.title, questions)
        db.remember_questions(title_key, [q.question for q in questions])
        return {
            "quiz_id": quiz_id,
            "questions": [
                {"position": i, "question": q.question, "options": list(q.options)}
                for i, q in enumerate(questions)
            ],
        }

    @app.post("/chat")
    def chat(body: ChatRequest) -> dict:
        """Free-text chat: the agent decides whether to search, create a quiz, or just answer.

        Memory: the last WINDOW_SIZE messages and the session's quiz ids are sent as data.
        """
        if chat_client_factory is None:
            raise HTTPException(status_code=503, detail="Chat non configurata")
        check_budget()
        session_id = body.session_id or secrets.token_hex(16)
        db.purge_chat_older_than(SESSION_TTL_HOURS)
        history_block = render_history(
            db.recent_chat_messages(session_id, WINDOW_SIZE), db.chat_quiz_ids(session_id)
        )
        client = chat_client_factory()
        ctx = ChatContext(db=db, retriever=get_retriever(), generator=get_generator(), verifier=get_verifier())
        started = time.perf_counter()
        try:
            result = run_chat(client, ctx, body.message, history_block)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Agente non disponibile, riprova") from exc
        db.add_chat_message(session_id, "user", body.message)
        db.add_chat_message(session_id, "assistant", result.answer)
        for quiz_id in result.quiz_ids:
            db.add_chat_quiz(session_id, quiz_id)
        if call_logger is not None:
            call_logger.log(
                model=client.model_name,
                operation="chat",
                request=body.message,
                response=result.answer,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_s=time.perf_counter() - started,
                tools_called=result.tools_called,
            )
        return {
            "session_id": session_id,
            "answer": result.answer,
            "tools_called": result.tools_called,
            "quiz_ids": result.quiz_ids,
        }

    @app.delete("/chat/{session_id}")
    def delete_chat(session_id: str) -> dict:
        """Deletes the messages and quiz ids of a session (privacy)."""
        db.delete_chat_session(session_id)
        return {"deleted": True}

    @app.post("/answer")
    def answer(body: AnswerRequest) -> dict:
        result = score_and_save(body.quiz_id, body.player, body.answers)
        return {
            "points": result.points,
            "correct": result.correct,
            "unanswered": result.unanswered,
            "total": result.total,
            "results": list(result.results),
            # Sources are shown only after answering, never the correct option.
            "sources": [q.source for q in db.get_questions(body.quiz_id)],
        }

    @app.post("/challenge", status_code=201)
    def create_challenge(body: AnswerRequest) -> dict:
        # The challenger plays first. The returned code is shared with the opponent.
        result = score_and_save(body.quiz_id, body.player, body.answers)
        code = secrets.token_hex(3).upper()
        db.create_challenge(code, body.quiz_id, body.player)
        return {"code": code, "challenger": body.player, "status": "pending", "points": result.points}

    @app.get("/challenge/{code}")
    def get_challenge(code: str) -> dict:
        challenge = db.get_challenge(code)
        if challenge is None:
            raise HTTPException(status_code=404, detail="Sfida non trovata")
        questions = db.get_questions(challenge["quiz_id"])
        body = {
            "code": code,
            "challenger": challenge["challenger"],
            "questions": [
                {"position": i, "question": q.question, "options": list(q.options)}
                for i, q in enumerate(questions)
            ],
            "status": "pending",
        }
        if challenge["opponent"] is None:
            # Opponent has not played: no scores and no answers are revealed.
            return body
        # Both played: reveal only the totals, never per-question answers.
        challenger = db.get_attempt(challenge["quiz_id"], challenge["challenger"])
        opponent = db.get_attempt(challenge["quiz_id"], challenge["opponent"])
        body.update({
            "status": "completed",
            "scores": {
                challenge["challenger"]: challenger["points"],
                challenge["opponent"]: opponent["points"],
            },
            "winner": _winner(challenge["challenger"], challenger["points"],
                              challenge["opponent"], opponent["points"]),
        })
        return body

    @app.post("/challenge/{code}/answer")
    def answer_challenge(code: str, body: ChallengeAnswerRequest) -> dict:
        challenge = db.get_challenge(code)
        if challenge is None:
            raise HTTPException(status_code=404, detail="Sfida non trovata")
        if body.player == challenge["challenger"]:
            raise HTTPException(status_code=400, detail="Lo sfidante non può rispondere alla propria sfida")
        if challenge["opponent"] not in (None, body.player):
            raise HTTPException(status_code=409, detail="La sfida è già stata accettata da un altro giocatore")
        result = score_and_save(challenge["quiz_id"], body.player, body.answers)
        db.set_challenge_opponent(code, body.player)
        challenger = db.get_attempt(challenge["quiz_id"], challenge["challenger"])
        return {
            "status": "completed",
            "you": {"player": body.player, "points": result.points, "results": list(result.results)},
            "scores": {challenge["challenger"]: challenger["points"], body.player: result.points},
            "winner": _winner(challenge["challenger"], challenger["points"], body.player, result.points),
        }

    @app.get("/leaderboard")
    def leaderboard() -> dict:
        return {"leaderboard": db.leaderboard()}

    return app


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def _generator_from_env() -> QuizGenerator:
    from src.agent.llm import create_client

    return QuizGenerator(create_client("MODEL_AGENT"), logger=SESSION_LOGGER)


def _verifier_from_env() -> FactVerifier:
    from src.agent.llm import create_client

    return FactVerifier(create_client("MODEL_JUDGE"), logger=SESSION_LOGGER)


def _agent_client_from_env() -> Client:
    from src.agent.llm import create_client

    return create_client("MODEL_AGENT")


_load_env()
# One budget and one logger for the whole server process: every paid call is counted here.
SESSION_BUDGET = SessionBudget.from_env()
SESSION_LOGGER = CallLogger(budget=SESSION_BUDGET)

app = create_app(
    Database(os.environ.get("DB_PATH", DEFAULT_DB_PATH)),
    _generator_from_env,
    load_retriever_from_env,
    _verifier_from_env,
    chat_client_factory=_agent_client_from_env,
    call_logger=SESSION_LOGGER,
    budget=SESSION_BUDGET,
)
