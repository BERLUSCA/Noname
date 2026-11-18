"""Tests for the chat agent with a scripted client: no network. The agent really runs its loop."""

import json

from datapizza.clients.mock_client import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.type import FunctionCallBlock, TextBlock

from src.agent.chat import OUT_OF_SCOPE_ANSWER, SYSTEM_PROMPT, run_chat
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


def make_ctx(retriever=None):
    return ChatContext(db=Database(":memory:"), retriever=retriever or FakeRetriever(),
                       generator=StubGenerator(), verifier=AllSupportedJudge())


def test_text_only_answer_without_document_is_out_of_scope():
    ctx = make_ctx()
    result = run_chat(ScriptedClient([("text", "Ciao!")]), ctx, "Cos'è la carbonara?")
    assert result.answer == OUT_OF_SCOPE_ANSWER
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


def test_answer_without_corpus_document_is_replaced_by_fixed_message():
    ctx = make_ctx()
    result = run_chat(ScriptedClient([("text", "La carbonara è un piatto romano.")]), ctx, "Cos'è la carbonara?")
    assert result.answer == OUT_OF_SCOPE_ANSWER
    assert "carbonara" not in result.answer


def test_missing_title_in_corpus_gives_fixed_message_even_after_search():
    ctx = make_ctx(retriever=FakeRetriever())
    script = [("tool", "search_corpus", {"title": "Ricetta della carbonara"}),
              ("text", "La carbonara è un piatto romano.")]
    result = run_chat(ScriptedClient(script), ctx, "Cos'è la carbonara?")
    assert result.answer == OUT_OF_SCOPE_ANSWER


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
