"""FastAPI app: quiz creation, answer scoring, challenges, leaderboard, chat agent.

Scoring is code, never the LLM. Every paid call is counted by the session budget.
"""

import asyncio
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from collections.abc import Callable
import json
from pathlib import Path

from datapizza.core.clients import Client
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.chat import run_chat
from src.agent.memory import WINDOW_SIZE, render_history
from src.agent.tools import ChatContext
from src.api.db import Database, player_key
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
PAGE_ROUTES = {"/"}  # the single HTML page: always revalidated
OLD_PAGES = ("/quiz", "/sfide", "/chat", "/classifica")  # removed pages: redirected to the single page


def corpus_titles() -> list[str]:
    """Titles of the corpus, read from the manifest (no model is loaded)."""
    manifest = ROOT / "data" / "raw" / "manifest.json"
    if not manifest.exists():
        return []
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    return sorted((e["title"] for e in entries), key=str.lower)


class QuizRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    n_questions: int = Field(default=3, ge=1, le=10)


class AnswerRequest(BaseModel):
    quiz_id: str
    player: str = Field(min_length=1, max_length=40)
    answers: list[int | None]


class PlayerRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class NamedChallengeRequest(BaseModel):
    quiz_id: str
    challenger: str = Field(min_length=1, max_length=40)
    opponent: str = Field(min_length=1, max_length=40)
    answers: list[int | None]


class QuizChallengeRequest(BaseModel):
    challenger: str = Field(min_length=1, max_length=40)
    opponent: str = Field(min_length=1, max_length=40)


class ChallengeAnswerRequest(BaseModel):
    player: str = Field(min_length=1, max_length=40)
    answers: list[int | None]


class ChatRequest(BaseModel):
    player: str = Field(min_length=1, max_length=40)
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
    preload: bool = False,
) -> FastAPI:
    app = FastAPI(title="Quanto ne sai di film e serie TV?")
    state: dict[str, object] = {}
    lock = threading.RLock()  # several pages may open at once: each component is built once

    # Components are built on first use (or by POST /warmup when a page opens), so the app starts
    # and tests run without API keys.
    def get_generator() -> QuizGenerator:
        with lock:
            if "generator" not in state:
                state["generator"] = generator_factory()
            return state["generator"]

    def get_retriever() -> TitleRetriever:
        with lock:
            if "retriever" not in state:
                try:
                    state["retriever"] = retriever_factory()
                except IndexMismatch as exc:
                    # The index does not match the corpus or the model: refuse, do not serve stale answers.
                    raise HTTPException(status_code=503, detail=f"Indice non allineato: {exc}") from exc
            return state["retriever"]

    def get_verifier() -> FactVerifier:
        with lock:
            if "verifier" not in state:
                state["verifier"] = verifier_factory()
            return state["verifier"]

    def get_chat_client() -> Client:
        with lock:
            if "chat_client" not in state:
                state["chat_client"] = chat_client_factory()
            return state["chat_client"]

    def warm_up() -> dict[str, str]:
        """Loads every component now. Returns 'ok' or the reason for each one; never raises."""
        status: dict[str, str] = {}
        for name, load in [("indice", get_retriever), ("generatore", get_generator),
                           ("giudice", get_verifier)]:
            try:
                load()
                status[name] = "ok"
            except HTTPException as exc:
                status[name] = str(exc.detail)
            except Exception as exc:  # missing key, unreachable model, ...
                status[name] = f"errore: {exc}"
        if chat_client_factory is None:
            status["chat"] = "non configurata"
        else:
            try:
                get_chat_client()
                status["chat"] = "ok"
            except Exception as exc:
                status["chat"] = f"errore: {exc}"
        return status

    if preload:
        # Load the models when the application starts, before it accepts requests.
        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            status = await asyncio.to_thread(warm_up)  # blocking loads run off the event loop
            logging.getLogger("uvicorn.error").info("Componenti caricati all'avvio: %s", status)
            yield

        app.router.lifespan_context = lifespan

    @app.post("/warmup")
    def warmup() -> dict:
        """Called by every test page when it opens: loads the models before the first question."""
        status = warm_up()
        return {"ready": all(v == "ok" for v in status.values()), "components": status}

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

    # Test pages for the backend (static HTML + plain JavaScript, no build). Interactive docs: /docs.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        # Without this a browser can keep an old common.js next to a new page ("markResults is not defined").
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path in PAGE_ROUTES:
            response.headers["Cache-Control"] = "no-cache"
        return response

    def page(name: str) -> Callable[[], FileResponse]:
        def serve() -> FileResponse:
            return FileResponse(STATIC_DIR / name)
        return serve

    app.add_api_route("/", page("index.html"), include_in_schema=False, response_class=FileResponse)

    def to_home() -> RedirectResponse:
        return RedirectResponse("/")

    for old in OLD_PAGES:  # old bookmarks still work
        app.add_api_route(old, to_home, include_in_schema=False, methods=["GET"])

    @app.get("/titles")
    def titles() -> dict:
        """Films and series the quizzes and questions can be about."""
        return {"titles": corpus_titles()}

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
        # A chat session belongs to one player: another player cannot read or extend it.
        if body.session_id and db.chat_session_owner(body.session_id) not in (None, player_key(body.player)):
            raise HTTPException(status_code=403, detail="Questa sessione appartiene a un altro giocatore")
        session_id = body.session_id or secrets.token_hex(16)
        db.claim_chat_session(session_id, body.player)
        db.purge_chat_older_than(SESSION_TTL_HOURS)
        history_block = render_history(
            db.recent_chat_messages(session_id, WINDOW_SIZE), db.chat_quiz_ids(session_id)
        )
        client = get_chat_client()
        ctx = ChatContext(db=db, retriever=get_retriever(), generator=get_generator(), verifier=get_verifier(),
                          player=body.player, titles=corpus_titles(), history=db.recent_chat_messages(session_id, WINDOW_SIZE),
                          session_quiz_ids=db.chat_quiz_ids(session_id))
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
            # Set only when the agent listed the player's challenges (codes, roles, status: no scores).
            "challenges": ctx.challenges,
            # Code of a challenge accepted in this turn: the page opens its questions in the chat.
            "accept": ctx.challenge_to_play,
            # Questions of the quizzes created in this turn, so the page can show a form. Never the correct answers.
            "quizzes": [
                {
                    "quiz_id": quiz_id,
                    "questions": [
                        {"position": i, "question": q.question, "options": list(q.options)}
                        for i, q in enumerate(db.get_questions(quiz_id) or [])
                    ],
                }
                for quiz_id in result.quiz_ids
            ],
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

    @app.post("/players")
    def register_player(body: PlayerRequest) -> dict:
        """Registers a name (no password). Idempotent: created=False if the name already exists."""
        created = db.add_player(body.name)
        return {"name": body.name.strip(), "created": created}

    @app.get("/players")
    def list_players() -> dict:
        return {"players": db.list_players()}

    @app.post("/challenges", status_code=201)
    def create_named_challenge(body: NamedChallengeRequest) -> dict:
        """Challenger answers the quiz and invites a registered opponent. The opponent answers with the code."""
        if not db.has_player(body.challenger):
            raise HTTPException(status_code=400, detail="Registra prima il tuo nome")
        if not db.has_player(body.opponent):
            raise HTTPException(status_code=400, detail="L'avversario deve essere tra i giocatori registrati")
        if player_key(body.opponent) == player_key(body.challenger):
            raise HTTPException(status_code=400, detail="Non puoi sfidare te stesso")
        result = score_and_save(body.quiz_id, body.challenger, body.answers)
        code = secrets.token_hex(3).upper()
        db.create_challenge(code, body.quiz_id, body.challenger, invited=body.opponent)
        return {"code": code, "challenger": body.challenger, "opponent": body.opponent,
                "status": "pending", "points": result.points}

    @app.post("/quiz/{quiz_id}/challenge", status_code=201)
    def challenge_with_answered_quiz(quiz_id: str, body: QuizChallengeRequest) -> dict:
        """Challenges a registered player with a quiz the challenger has already answered (score taken from the attempt)."""
        if db.get_questions(quiz_id) is None:
            raise HTTPException(status_code=404, detail="Quiz non trovato")
        if not db.has_player(body.challenger):
            raise HTTPException(status_code=400, detail="Registra prima il tuo nome")
        if not db.has_player(body.opponent):
            raise HTTPException(status_code=400, detail="L'avversario deve essere tra i giocatori registrati")
        if player_key(body.opponent) == player_key(body.challenger):
            raise HTTPException(status_code=400, detail="Non puoi sfidare te stesso")
        attempt = db.get_attempt(quiz_id, body.challenger)
        if attempt is None:
            raise HTTPException(status_code=400, detail="Rispondi prima al quiz, poi sfida un giocatore")
        if db.has_attempt(quiz_id, body.opponent):
            raise HTTPException(status_code=409, detail="L'avversario ha già risposto a questo quiz")
        code = secrets.token_hex(3).upper()
        db.create_challenge(code, quiz_id, body.challenger, invited=body.opponent)
        return {"code": code, "challenger": body.challenger, "opponent": body.opponent,
                "status": "pending", "points": attempt["points"]}

    @app.get("/players/{name}/challenges")
    def player_challenges(name: str, x_player: str | None = Header(default=None)) -> dict:
        """Only the player's own challenges: the caller must declare the same name in the X-Player header.

        Names are not passwords, so this is an identity declared by the client, not an authentication.
        Codes, names, dates and status only: no scores, no answers.
        """
        if x_player is None or player_key(x_player) != player_key(name):
            raise HTTPException(status_code=403, detail="Vedi solo le tue sfide")
        return {"challenges": db.challenges_for(name)}

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
        if challenge["invited"] is not None and player_key(body.player) != player_key(challenge["invited"]):
            raise HTTPException(status_code=403, detail="Questa sfida è riservata a un altro giocatore")
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
    preload=True,
)
