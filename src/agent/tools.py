"""Tools the chat agent can call. Each tool returns TEXT to the model, never correct answers.

Built per request (ChatContext) so each request has its own call counter (guardrail 6: max 5 tool calls).
"""

from dataclasses import dataclass, field

from datapizza.tools import Tool

from src.api.db import Database
from src.quiz.generator import QuizGenerator
from src.quiz.verified import NotEnoughVerifiedQuestions, VerificationUnavailable, generate_verified
from src.rag.index import InsufficientMaterial, normalize_title
from src.rag.service import TitleRetriever
from src.rag.verifier import FactVerifier

MAX_TOOL_CALLS = 5
MAX_QUESTIONS = 5
EXCERPT_CHARS = 500  # measured: the corpus excerpt was the main token cost of a chat turn
EXCERPT_CHUNKS = 2  # best-ranked chunks only


@dataclass
class ChatContext:
    db: Database
    retriever: TitleRetriever
    generator: QuizGenerator
    verifier: FactVerifier
    calls: list[str] = field(default_factory=list)
    quiz_ids: list[str] = field(default_factory=list)
    # Corpus documents actually found in this request. Without one, the answer is out of scope.
    documents_found: list[str] = field(default_factory=list)


def build_tools(ctx: ChatContext) -> list[Tool]:
    def _count(name: str) -> bool:
        if len(ctx.calls) >= MAX_TOOL_CALLS:
            return False
        ctx.calls.append(name)
        return True

    def search_corpus(title: str) -> str:
        """Cerca nel corpus Wikipedia il documento con il titolo indicato e restituisce un estratto."""
        if not _count("search_corpus"):
            return "Limite di chiamate ai tool raggiunto."
        try:
            chunks = ctx.retriever.retrieve(title)
        except InsufficientMaterial:
            return f"Nessun documento nel corpus per il titolo '{title}'. Non inventare informazioni."
        ctx.documents_found.append(chunks[0].title)
        excerpt = "\n".join(c.text for c in chunks[:EXCERPT_CHUNKS])[:EXCERPT_CHARS]
        return f"Documento: {chunks[0].title}\nFonte: {chunks[0].url}\nEstratto:\n{excerpt}"

    def create_quiz(title: str, n_questions: int = 3) -> str:
        """Crea un quiz verificato su un titolo presente nel corpus. Restituisce id e domande, senza risposte."""
        if not _count("create_quiz"):
            return "Limite di chiamate ai tool raggiunto."
        if not 1 <= n_questions <= MAX_QUESTIONS:
            return f"Numero di domande non valido: scegli tra 1 e {MAX_QUESTIONS}."
        title_key = normalize_title(title)
        try:
            chunks = ctx.retriever.retrieve(title)
        except InsufficientMaterial:
            return f"Nessun documento nel corpus per il titolo '{title}': impossibile creare il quiz."
        passage = "\n".join(c.text for c in chunks)
        try:
            questions = generate_verified(
                ctx.generator, ctx.verifier, title, passage, chunks[0].url, n_questions,
                avoid=ctx.db.recent_questions(title_key),
            )
        except (VerificationUnavailable, NotEnoughVerifiedQuestions) as exc:
            return f"Quiz non creato: {exc}. Dillo all'utente senza inventare domande."
        ctx.documents_found.append(chunks[0].title)
        quiz_id = ctx.db.save_quiz(title, questions)
        ctx.db.remember_questions(title_key, [q.question for q in questions])
        ctx.quiz_ids.append(quiz_id)
        lines = [f"quiz_id: {quiz_id}"]
        for i, q in enumerate(questions):
            options = " | ".join(f"{j}) {o}" for j, o in enumerate(q.options))
            lines.append(f"{i}. {q.question}  [{options}]")
        lines.append("Le risposte corrette non vanno rivelate all'utente prima che risponda.")
        return "\n".join(lines)

    return [
        Tool(func=search_corpus, description="Cerca nel corpus il documento con il titolo indicato."),
        Tool(func=create_quiz, description="Crea un quiz di domande a scelta multipla su un titolo del corpus."),
    ]
