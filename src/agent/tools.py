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
    player: str | None = None
    # Answers that come from the app database (players, challenges) or from this chat, not from the corpus.
    app_data_used: bool = False
    # The player's challenges, when the agent listed them: the page shows them as a list with a "play" button.
    challenges: list[dict] | None = None
    # Titles of the corpus: the films and series quizzes and questions can be about.
    titles: list[str] = field(default_factory=list)
    # Code of a challenge the player accepted in the chat: the page opens its questions in the conversation.
    challenge_to_play: str | None = None
    # The player's challenges, when the agent listed them: the page shows them as a list with a "play" button.
    challenges: list[dict] | None = None
    # Fixed message shown when a quiz could not be created (title not in corpus, questions not verified).
    notice: str | None = None
    # Text returned by those tools: the only source an answer without a corpus document may rely on.
    app_data_texts: list[str] = field(default_factory=list)
    # What was said earlier in this session (the memory window) and the quizzes created in it.
    history: list[dict] = field(default_factory=list)
    session_quiz_ids: list[str] = field(default_factory=list)


def conversation_text(ctx: ChatContext) -> str:
    """The recent messages of this session and the titles of its quizzes, as plain text. Empty if there is nothing."""
    lines = []
    for m in ctx.history:
        who = "Utente" if m["role"] == "user" else "Assistente"
        lines.append(f"{who}: {m['content'][:300]}")
    titles = [t for t in (ctx.db.quiz_title(q) for q in ctx.session_quiz_ids) if t]
    if titles:
        lines.append("Quiz creati in questa conversazione su: " + ", ".join(titles))
    return "\n".join(lines)


def build_tools(ctx: ChatContext) -> list[Tool]:
    def _count(name: str) -> bool:
        if len(ctx.calls) >= MAX_TOOL_CALLS:
            return False
        ctx.calls.append(name)
        return True

    def _app_data(text: str) -> str:
        ctx.app_data_used = True
        ctx.app_data_texts.append(text)
        return text

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
            ctx.notice = f"Non ho trovato \"{title.strip()}\" nel corpus, quindi non posso creare il quiz. Prova con un altro titolo, per esempio Inception."
            return f"Nessun documento nel corpus per il titolo '{title}': impossibile creare il quiz."
        passage = "\n".join(c.text for c in chunks)
        try:
            questions = generate_verified(
                ctx.generator, ctx.verifier, title, passage, chunks[0].url, n_questions,
                avoid=ctx.db.recent_questions(title_key),
            )
        except (VerificationUnavailable, NotEnoughVerifiedQuestions) as exc:
            ctx.notice = f"Non sono riuscito a creare un quiz verificato su {title.strip()} ({exc}). Riprova: le domande vengono scartate se il testo di Wikipedia non le conferma."
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

    def ask_which_title() -> str:
        """Da usare quando l'utente vuole un quiz ma non ha indicato un film o una serie."""
        if not _count("ask_which_title"):
            return "Limite di chiamate ai tool raggiunto."
        ctx.notice = "Su quale film o serie TV vuoi il quiz? Per esempio Inception, Matrix o Il padrino."
        return "Chiedi all'utente su quale titolo vuole il quiz."

    def list_players() -> str:
        """Elenca i giocatori registrati, tra cui si può scegliere un avversario."""
        if not _count("list_players"):
            return "Limite di chiamate ai tool raggiunto."
        names = ctx.db.list_players()
        return _app_data("Giocatori: " + ", ".join(names) if names else "Nessun giocatore registrato.")

    def my_challenges() -> str:
        """Elenca le sfide del giocatore corrente (codice, ruolo, stato). Non rivela punteggi o risposte."""
        if not _count("my_challenges"):
            return "Limite di chiamate ai tool raggiunto."
        if not ctx.player:
            return _app_data("Giocatore non indicato.")
        items = ctx.db.challenges_for(ctx.player)
        if not items:
            return _app_data("Nessuna sfida.")
        ctx.challenges = [
            {k: c[k] for k in ("code", "role", "status", "challenger", "invited", "created_at")} for c in items
        ]
        return _app_data("\n".join(
            f"codice {c['code']}: {c['role']}, {c['status']}" + (f", avversario {c['invited']}" if c['invited'] else "")
            for c in items
        ))

    def accept_challenge(code: str = "") -> str:
        """Accetta una sfida ricevuta dal giocatore corrente: le domande si aprono nella chat. Senza codice, se ce n'è una sola in attesa, usa quella."""
        if not _count("accept_challenge"):
            return "Limite di chiamate ai tool raggiunto."
        if not ctx.player:
            return _app_data("Giocatore non indicato.")
        pending = [c for c in ctx.db.challenges_for(ctx.player) if c["role"] == "avversario" and c["status"] == "in attesa"]
        if not pending:
            return _app_data("Nessuna sfida in attesa di risposta.")
        wanted = code.strip().upper()
        if wanted:
            chosen = next((c for c in pending if c["code"].upper() == wanted), None)
            if chosen is None:
                return _app_data(f"La sfida {wanted} non è tra quelle in attesa per questo giocatore.")
        elif len(pending) == 1:
            chosen = pending[0]
        else:
            codes = ", ".join(f"{c['code']} (da {c['challenger']})" for c in pending)
            return _app_data(f"Ci sono più sfide in attesa: {codes}. Chiedi all'utente quale codice vuole accettare.")
        ctx.challenge_to_play = chosen["code"]
        return _app_data(f"Sfida {chosen['code']} di {chosen['challenger']} accettata: le domande compaiono nella chat.")

    def list_titles() -> str:
        """Elenca tutti i film e le serie TV del corpus, su cui si possono fare quiz e domande."""
        if not _count("list_titles"):
            return "Limite di chiamate ai tool raggiunto."
        if not ctx.titles:
            return _app_data("Elenco dei titoli non disponibile.")
        return _app_data("Film e serie TV disponibili: " + ", ".join(ctx.titles) + ".")

    def challenge_help() -> str:
        """Spiega come lanciare una sfida a un altro giocatore e come giocare le sfide ricevute."""
        if not _count("challenge_help"):
            return "Limite di chiamate ai tool raggiunto."
        return _app_data(
            "Per sfidare un altro giocatore: crea un quiz in chat, rispondi alle domande e premi Invia risposte; "
            "sotto il quiz scegli l'avversario tra i giocatori registrati e premi Sfida. "
            "L'avversario vede la sfida chiedendo 'le mie sfide' nella chat o nella pagina Sfide e la gioca con il suo nome. "
            "Si vedono solo codici e stato delle sfide, mai i punteggi finché entrambi non hanno risposto."
        )

    def recall_conversation() -> str:
        """Restituisce i messaggi recenti di questa conversazione e i quiz creati, per rispondere a "di cosa abbiamo parlato"."""
        if not _count("recall_conversation"):
            return "Limite di chiamate ai tool raggiunto."
        text = conversation_text(ctx)
        return _app_data(text or "La conversazione è appena iniziata: non c'è nulla da ricordare.")

    return [
        Tool(func=ask_which_title, description="Chiede all'utente il titolo quando vuole un quiz ma non l'ha indicato."),
        Tool(func=accept_challenge, description="Accetta una sfida ricevuta (codice opzionale): le domande si aprono nella chat."),
        Tool(func=list_titles, description="Elenca tutti i film e le serie TV su cui si possono fare quiz e domande."),
        Tool(func=challenge_help, description="Spiega come sfidare un altro giocatore e come giocare le sfide ricevute."),
        Tool(func=recall_conversation, description="Restituisce cosa vi siete detti in questa conversazione (messaggi recenti e quiz creati)."),
        Tool(func=list_players, description="Elenca i giocatori registrati tra cui scegliere un avversario."),
        Tool(func=my_challenges, description="Elenca le sfide del giocatore corrente, con codice e stato."),
        Tool(func=search_corpus, description="Cerca nel corpus il documento con il titolo indicato."),
        Tool(func=create_quiz, description="Crea un quiz di domande a scelta multipla su un titolo del corpus."),
    ]
