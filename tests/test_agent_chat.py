"""Tests for the chat agent with a scripted client: no network. The agent really runs its loop."""

import json

from datapizza.clients.mock_client import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.type import FunctionCallBlock, TextBlock

from src.agent.chat import OUT_OF_SCOPE_ANSWER, SYSTEM_PROMPT, UNVERIFIED_NOTE, run_chat
from src.agent.tools import MAX_TOOL_CALLS, ChatContext
from src.api.db import Database
from src.quiz.generator import Question, QuizGenerator
from src.rag.index import InsufficientMaterial
from src.rag.verifier import FactVerifier
from tests.fakes import FakeRetriever


class ScriptedClient(MockClient):
    """Replays a script: each entry is ('tool', name, args) or ('text', answer)."""

    def __init__(self, script):
        super().__init__(model_name="scripted")
        self.script = list(script)
        self.calls = 0

    def _invoke(self, input, tools=None, memory=None, tool_choice="auto", temperature=None,
                max_tokens=None, system_prompt=None, **kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else ("text", "fine")
        if step[0] == "tool":
            block = FunctionCallBlock(id=str(self.calls), name=step[1], arguments=step[2],
                                      tool=next(t for t in tools if t.name == step[1]))
            return ClientResponse(content=[block])
        return ClientResponse(content=[TextBlock(content=step[1])])


class StubGenerator(QuizGenerator):
    def __init__(self):
        self.client = None
        self.max_tokens = 400
        self.logger = None
        self.last_usage = None

    def generate(self, title, passage, source, n_questions=3, focus=None, avoid=None):
        return [Question(question=f"Domanda su {title} {i}", options=("A", "B", "C", "D"),
                         correct_index=1, source=source) for i in range(n_questions)]


class AllSupportedJudge(FactVerifier):
    def __init__(self):
        self.client = None
        self.logger = None

    def verify_many(self, claims, text):
        return [True] * len(claims)

    def verify_answer(self, claim, text):
        from src.rag.verifier import FactVerdict

        self.seen = (claim, text)
        return FactVerdict(supported="BAD" not in claim, reason="fake")


def make_ctx(retriever=None):
    return ChatContext(db=Database(":memory:"), retriever=retriever or FakeRetriever(),
                       generator=StubGenerator(), verifier=AllSupportedJudge())


def test_text_only_answer_without_document_is_out_of_scope():
    ctx = make_ctx()
    result = run_chat(ScriptedClient([("text", "Ciao!")]), ctx, "Cos'è la carbonara?")
    assert result.answer.endswith(UNVERIFIED_NOTE)
    assert result.tools_called == []


def test_agent_creates_quiz_through_tool_and_returns_no_answers():
    ctx = make_ctx()
    client = ScriptedClient([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}),
                             ("text", "Ecco il quiz.")])
    result = run_chat(client, ctx, "fammi un quiz su Inception")
    assert result.tools_called == ["create_quiz"]
    assert len(result.quiz_ids) == 1
    assert ctx.db.get_questions(result.quiz_ids[0])[0].correct_index == 1  # stored server-side


def test_tool_result_given_to_model_contains_no_correct_index():
    from src.agent.tools import build_tools

    ctx = make_ctx()
    create = next(t for t in build_tools(ctx) if t.name == "create_quiz")
    text = create.func(title="Inception", n_questions=2)
    assert "correct_index" not in text
    assert "1) B" in text  # options are shown to the model; the correct index is not


def test_search_corpus_reports_missing_title_without_inventing():
    from src.agent.tools import build_tools

    ctx = make_ctx(retriever=FakeRetriever())
    search = next(t for t in build_tools(ctx) if t.name == "search_corpus")
    text = search.func(title="Ricetta della pizza")
    assert text.startswith("Nessun documento nel corpus per il titolo 'Ricetta della pizza'")
    assert "Non inventare" in text


def test_tool_calls_are_capped():
    ctx = make_ctx()
    script = [("tool", "search_corpus", {"title": "Inception"})] * (MAX_TOOL_CALLS + 3) + [("text", "fine")]
    run_chat(ScriptedClient(script), ctx, "cerca tanto")
    assert len(ctx.calls) == MAX_TOOL_CALLS


def test_system_prompt_treats_user_text_as_data():
    assert "DATI" in SYSTEM_PROMPT
    assert "non dire quale opzione" in SYSTEM_PROMPT


def test_search_corpus_excerpt_is_short():
    from src.agent.tools import EXCERPT_CHARS, build_tools

    ctx = make_ctx()
    search = next(t for t in build_tools(ctx) if t.name == "search_corpus")
    text = search.func(title="Inception")
    excerpt = text.split("Estratto:\n", 1)[1]
    assert len(excerpt) <= EXCERPT_CHARS


def test_system_prompt_allows_general_answers_from_corpus():
    assert "Non rifiutare per principio" in SYSTEM_PROMPT
    assert "cita la fonte" in SYSTEM_PROMPT


def test_answer_without_corpus_document_is_shown_with_an_unverified_note():
    ctx = make_ctx()
    result = run_chat(ScriptedClient([("text", "La carbonara è un piatto romano.")]), ctx, "Cos'è la carbonara?")
    assert result.answer == "La carbonara è un piatto romano." + UNVERIFIED_NOTE


def test_answer_backed_by_a_corpus_document_has_no_note():
    ctx = make_ctx()
    client = ScriptedClient([("tool", "search_corpus", {"title": "Inception"}), ("text", "Inception è di Nolan.")])
    assert run_chat(client, ctx, "Parlami di Inception").answer == "Inception è di Nolan."



def test_missing_title_in_corpus_gives_fixed_message_even_after_search():
    ctx = make_ctx(retriever=FakeRetriever())
    script = [("tool", "search_corpus", {"title": "Ricetta della carbonara"}),
              ("text", "La carbonara è un piatto romano.")]
    result = run_chat(ScriptedClient(script), ctx, "Cos'è la carbonara?")
    assert result.answer.endswith(UNVERIFIED_NOTE)


def test_answer_with_corpus_document_is_kept():
    ctx = make_ctx()
    script = [("tool", "search_corpus", {"title": "Inception"}), ("text", "Inception è di Nolan.")]
    result = run_chat(ScriptedClient(script), ctx, "Chi ha diretto Inception?")
    assert result.answer == "Inception è di Nolan."


def test_greeting_gets_welcome_without_calling_model():
    from src.agent.chat import GREETING_ANSWER

    client = ScriptedClient([("text", "risposta del modello")])
    result = run_chat(client, make_ctx(), "Ciao!")
    assert result.answer == GREETING_ANSWER
    assert client.calls == 0


def test_greeting_with_request_is_not_a_greeting():
    from src.agent.chat import GREETING_ANSWER

    result = run_chat(ScriptedClient([("text", "x")]), make_ctx(), "ciao, dammi la ricetta della carbonara")
    assert result.answer != GREETING_ANSWER


def test_recall_conversation_lets_the_agent_answer_about_the_chat():
    ctx = make_ctx()
    quiz_id = ctx.db.save_quiz("Inception", [Question("Q?", ["a", "b", "c", "d"], 0, "src")])
    ctx.history = [{"role": "user", "content": "Parliamo di Matrix"}, {"role": "assistant", "content": "Volentieri."}]
    ctx.session_quiz_ids = [quiz_id]
    client = ScriptedClient([("tool", "recall_conversation", {}), ("text", "Abbiamo parlato di Matrix e Inception.")])
    result = run_chat(client, ctx, "Di che film abbiamo parlato?")
    assert result.answer == "Abbiamo parlato di Matrix e Inception."
    assert result.tools_called == ["recall_conversation"]


def test_recall_conversation_returns_history_and_quiz_titles():
    from src.agent.tools import build_tools

    ctx = make_ctx()
    quiz_id = ctx.db.save_quiz("Inception", [Question("Q?", ["a", "b", "c", "d"], 0, "src")])
    ctx.history = [{"role": "user", "content": "Parliamo di Matrix"}]
    ctx.session_quiz_ids = [quiz_id]
    text = {t.name: t for t in build_tools(ctx)}["recall_conversation"].func()
    assert "Parliamo di Matrix" in text and "Inception" in text


def test_recall_conversation_does_not_open_general_questions():
    # Without calling any tool, a general question still gets the out-of-scope answer.
    client = ScriptedClient([("text", "La carbonara è un piatto romano.")])
    assert run_chat(client, make_ctx(), "Cos'è la carbonara?").answer.endswith(UNVERIFIED_NOTE)


def test_conversation_answer_not_backed_by_the_conversation_is_rejected():
    # The agent calls the harmless tool, then answers from its own knowledge: the judge rejects it.
    ctx = make_ctx()
    ctx.history = [{"role": "user", "content": "Parliamo di Matrix"}]
    client = ScriptedClient([("tool", "recall_conversation", {}), ("text", "BAD: la carbonara è un piatto romano.")])
    result = run_chat(client, ctx, "Cos'è la carbonara?")
    assert result.answer.endswith(UNVERIFIED_NOTE)
    assert result.tools_called == ["recall_conversation"]


def test_judge_reads_the_tool_output_not_the_model_text():
    ctx = make_ctx()
    ctx.history = [{"role": "user", "content": "Parliamo di Matrix"}]
    client = ScriptedClient([("tool", "recall_conversation", {}), ("text", "Abbiamo parlato di Matrix.")])
    assert run_chat(client, ctx, "Di che film parliamo?").answer == "Abbiamo parlato di Matrix."
    claim, text = ctx.verifier.seen
    assert claim == "Abbiamo parlato di Matrix."
    assert "Parliamo di Matrix" in text


def test_conversation_answer_is_rejected_when_the_judge_fails():
    class BrokenJudge(AllSupportedJudge):
        def verify_answer(self, claim, text):
            raise RuntimeError("judge down")

    ctx = make_ctx()
    ctx.verifier = BrokenJudge()
    client = ScriptedClient([("tool", "recall_conversation", {}), ("text", "Abbiamo parlato di Matrix.")])
    assert run_chat(client, ctx, "Di che film parliamo?").answer.endswith(UNVERIFIED_NOTE)


def test_answers_backed_by_a_corpus_document_skip_the_conversation_judge():
    ctx = make_ctx()
    client = ScriptedClient([("tool", "search_corpus", {"title": "Inception"}), ("text", "BAD ma dal corpus.")])
    assert run_chat(client, ctx, "Parlami di Inception").answer == "BAD ma dal corpus."


def test_failed_quiz_explains_why_instead_of_saying_i_dont_know():
    class RejectAllJudge(AllSupportedJudge):
        def verify_many(self, claims, text):
            return [False] * len(claims)

    ctx = make_ctx()
    ctx.verifier = RejectAllJudge()
    client = ScriptedClient([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Boh.")])
    result = run_chat(client, ctx, "fammi un quiz su Inception")
    assert result.answer != OUT_OF_SCOPE_ANSWER
    assert "Inception" in result.answer and "verificat" in result.answer
    assert result.quiz_ids == []


def test_quiz_on_unknown_title_names_the_title():
    client = ScriptedClient([("tool", "create_quiz", {"title": "Ricetta della carbonara", "n_questions": 2}), ("text", "Boh.")])
    result = run_chat(client, make_ctx(), "fammi un quiz sulla carbonara")
    assert "Ricetta della carbonara" in result.answer and result.answer != OUT_OF_SCOPE_ANSWER


def test_quiz_request_without_title_asks_which_title():
    client = ScriptedClient([("tool", "ask_which_title", {}), ("text", "Boh.")])
    result = run_chat(client, make_ctx(), "fammi un quiz")
    assert "Su quale film o serie" in result.answer
    assert result.tools_called == ["ask_which_title"]
    assert result.quiz_ids == []


def test_question_about_the_chat_is_answered_without_calling_a_tool():
    # "mi hai salutato?" must work even if the model answers directly: the history is checked by the judge.
    ctx = make_ctx()
    ctx.history = [{"role": "user", "content": "ciao"}, {"role": "assistant", "content": "Ciao! Posso farti un quiz."}]
    result = run_chat(ScriptedClient([("text", "Sì, ti ho salutato.")]), ctx, "mi hai salutato?")
    assert result.answer == "Sì, ti ho salutato."
    assert result.tools_called == []
    assert "Ciao! Posso farti un quiz." in ctx.verifier.seen[1]


def test_general_question_with_history_is_still_out_of_scope_when_unsupported():
    ctx = make_ctx()
    ctx.history = [{"role": "user", "content": "ciao"}]
    result = run_chat(ScriptedClient([("text", "BAD: la carbonara è un piatto romano.")]), ctx, "cos'è la carbonara?")
    assert result.answer.endswith(UNVERIFIED_NOTE)


def test_quiz_notice_is_not_overridden_by_the_conversation_check():
    # Title not in the corpus while a history exists: the clear message wins over the judge on the history.
    ctx = make_ctx()
    ctx.history = [{"role": "user", "content": "genera un quiz"}]
    client = ScriptedClient([("tool", "create_quiz", {"title": "Ricetta della carbonara", "n_questions": 2}), ("text", "Boh.")])
    result = run_chat(client, ctx, "ricetta della carbonara")
    assert "Ricetta della carbonara" in result.answer


def test_answer_check_uses_a_prompt_that_ignores_greetings_and_questions():
    from src.rag.verifier import ANSWER_VERIFY_PROMPT, FactVerifier

    assert "Ignora saluti" in ANSWER_VERIFY_PROMPT
    assert hasattr(FactVerifier, "verify_answer")


def test_challenge_help_explains_how_to_challenge_and_counts_as_app_data():
    client = ScriptedClient([("tool", "challenge_help", {}), ("text", "Rispondi al quiz e poi scegli l'avversario.")])
    result = run_chat(client, make_ctx(), "come sfido un altro giocatore?")
    assert result.answer == "Rispondi al quiz e poi scegli l'avversario."


def _ctx_with_challenges():
    ctx = make_ctx()
    for name in ("Anna", "Luca", "Marco"):
        ctx.db.add_player(name)
    ctx.player = "Luca"
    return ctx


def test_accept_challenge_with_a_code_opens_that_challenge():
    ctx = _ctx_with_challenges()
    ctx.db.create_challenge("ABC123", "q", "Anna", invited="Luca")
    client = ScriptedClient([("tool", "accept_challenge", {"code": "abc123"}), ("text", "Ecco la sfida.")])
    result = run_chat(client, ctx, "accetto la sfida ABC123")
    assert ctx.challenge_to_play == "ABC123"
    assert result.answer == "Ecco la sfida."


def test_accept_challenge_without_a_code_uses_the_only_pending_one():
    ctx = _ctx_with_challenges()
    ctx.db.create_challenge("ABC123", "q", "Anna", invited="Luca")
    run_chat(ScriptedClient([("tool", "accept_challenge", {}), ("text", "Ok.")]), ctx, "accetto la sfida")
    assert ctx.challenge_to_play == "ABC123"


def test_accept_challenge_with_several_pending_asks_for_the_code():
    ctx = _ctx_with_challenges()
    ctx.db.create_challenge("AAA111", "q", "Anna", invited="Luca")
    ctx.db.create_challenge("BBB222", "q", "Marco", invited="Luca")
    run_chat(ScriptedClient([("tool", "accept_challenge", {}), ("text", "Quale codice?")]), ctx, "accetto")
    assert ctx.challenge_to_play is None


def test_accept_challenge_refuses_challenges_of_other_players_or_already_played():
    ctx = _ctx_with_challenges()
    ctx.db.create_challenge("OTHER1", "q", "Anna", invited="Marco")  # not for Luca
    ctx.db.create_challenge("MINE01", "q", "Luca", invited="Anna")   # Luca is the challenger
    for code in ("OTHER1", "MINE01"):
        run_chat(ScriptedClient([("tool", "accept_challenge", {"code": code}), ("text", "No.")]), ctx, f"accetto {code}")
        assert ctx.challenge_to_play is None


def test_list_titles_tool_returns_the_corpus_titles_and_counts_as_app_data():
    ctx = make_ctx()
    ctx.titles = ["Inception", "Matrix"]
    client = ScriptedClient([("tool", "list_titles", {}), ("text", "Ci sono Inception e Matrix.")])
    result = run_chat(client, ctx, "che film ci sono?")
    assert result.answer == "Ci sono Inception e Matrix."
    assert result.tools_called == ["list_titles"]
    assert "Inception, Matrix" in ctx.verifier.seen[1]
