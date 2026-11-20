"""Tests for the single static page (login, chat, film list, leaderboard). No network, no browser."""

import re

import pytest
from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import STATIC_DIR, create_app
from tests.fakes import FakeRetriever, FakeVerifier
from tests.test_api import FakeGenerator

PAGES = {"/": "index.html"}
OLD_ROUTES = ["/quiz", "/sfide", "/chat", "/classifica"]


def make_client():
    return TestClient(create_app(Database(":memory:"), lambda: FakeGenerator(), lambda: FakeRetriever(),
                                 lambda: FakeVerifier()))


@pytest.mark.parametrize("route", list(PAGES))
def test_each_page_is_served(route):
    response = make_client().get(route)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.parametrize("route", OLD_ROUTES)
def test_removed_pages_redirect_to_the_home_page(route):
    response = make_client().get(route, follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/"


def test_only_the_single_page_exists():
    assert sorted(p.name for p in STATIC_DIR.glob("*.html")) == ["index.html"]


def test_static_javascript_is_served():
    response = make_client().get("/static/common.js")
    assert response.status_code == 200
    assert "function api(" in response.text


def _all_static_text() -> str:
    parts = [p.read_text(encoding="utf-8") for p in STATIC_DIR.glob("*.html")]
    parts.append((STATIC_DIR / "common.js").read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_no_page_builds_html_from_text():
    text = _all_static_text()
    for unsafe in ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write"):
        assert unsafe not in text


def test_no_page_contains_correct_answer_fields():
    assert "correct_index" not in _all_static_text()


def test_page_does_not_poll_the_server():
    assert "setInterval" not in _all_static_text()


def test_name_is_only_stored_on_confirm():
    # The home page reads the input only inside the click handler, never on typing.
    home = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "oninput" not in home and "onkeyup" not in home and "onchange" not in home
    assert re.search(r'getElementById\("enter"\)\.onclick', home)
    assert "setPlayer(data.name)" in home


def test_page_has_login_chat_films_and_leaderboard_sections():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for element in ('id="login"', 'id="app"', 'id="log"', 'id="titles"', 'id="board"', 'id="logout"'):
        assert element in html, element
    assert "renderHeader()" in html and "renderNav" not in html


def test_all_pages_use_absolute_static_paths():
    # A relative path (e.g. src="common.js") breaks on /chat: the browser resolves it to /common.js.
    for name in PAGES.values():
        html = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert 'src="common.js"' not in html and 'href="style.css"' not in html, name
        for ref in re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', html):
            assert ref.startswith("/static/"), f"{name}: {ref}"


def test_challenge_send_asks_for_confirmation():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "window.confirm" in page
    assert "non si cambia" in page


def test_pages_warm_up_models_on_open():
    common = (STATIC_DIR / "common.js").read_text(encoding="utf-8")
    assert '"/warmup"' in common
    assert "startWarmUp()" in common  # called from renderHeader, which the page calls on open
    assert 'line.id = "warm"' in common


def test_warmup_loads_each_component_once(tmp_path):
    calls = {"retriever": 0, "generator": 0, "verifier": 0}

    def counting(name, factory):
        def build():
            calls[name] += 1
            return factory()
        return build

    from tests.fakes import FakeRetriever as R, FakeVerifier as V

    app = create_app(Database(":memory:"), counting("generator", lambda: FakeGenerator()),
                     counting("retriever", lambda: R()), counting("verifier", lambda: V()))
    client = TestClient(app)
    first = client.post("/warmup").json()
    second = client.post("/warmup").json()
    assert first["components"]["indice"] == "ok"
    assert first["components"]["generatore"] == "ok"
    assert first["components"]["giudice"] == "ok"
    assert first["components"]["chat"] == "non configurata"
    assert first["ready"] is False  # chat not configured in this test app
    assert second == first
    assert calls == {"retriever": 1, "generator": 1, "verifier": 1}


def test_models_load_at_application_startup_when_preload_is_on():
    from tests.fakes import FakeRetriever as R, FakeVerifier as V

    calls = {"retriever": 0}

    def build_retriever():
        calls["retriever"] += 1
        return R()

    app = create_app(Database(":memory:"), lambda: FakeGenerator(), build_retriever, lambda: V(), preload=True)
    assert calls["retriever"] == 0  # not loaded before the application starts
    with TestClient(app) as client:  # entering the context runs the startup handler
        assert calls["retriever"] == 1
        client.post("/warmup")  # already loaded: no second load
        assert calls["retriever"] == 1


def test_without_preload_models_are_not_loaded_at_startup():
    from tests.fakes import FakeRetriever as R, FakeVerifier as V

    calls = {"retriever": 0}

    def build_retriever():
        calls["retriever"] += 1
        return R()

    app = create_app(Database(":memory:"), lambda: FakeGenerator(), build_retriever, lambda: V())
    with TestClient(app):
        assert calls["retriever"] == 0


def test_chat_page_shows_quiz_form_and_posts_answers():
    chat = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "data.quizzes" in chat and "questionForm(" in chat
    assert '"/answer"' in chat


def test_static_files_are_revalidated_by_the_browser():
    response = make_client().get("/static/common.js")
    assert response.headers["cache-control"] == "no-cache"


def test_html_pages_are_revalidated_and_assets_are_versioned():
    for route, name in PAGES.items():
        assert make_client().get(route).headers["cache-control"] == "no-cache"
        html = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert "/static/common.js?v=" in html and "/static/style.css?v=" in html


def test_titles_endpoint_lists_the_corpus():
    titles = make_client().get("/titles").json()["titles"]
    assert "Inception" in titles
    assert titles == sorted(titles, key=str.lower)


def test_page_lists_all_films_with_quiz_and_info_buttons():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert '"/titles"' in page and "loadTitles()" in page
    assert "Fammi un quiz su" in page and "Parlami di" in page
    assert 'id="film-filter"' in page


def test_leaderboard_is_loaded_and_refreshed_after_answering():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert '"/leaderboard"' in page
    assert page.count("loadBoard();") >= 3  # at start and after each answered quiz or challenge


def test_logout_clears_the_player():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "clearPlayer()" in page


def test_chat_scrolls_inside_its_own_box_not_the_page():
    chat = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "scrollIntoView" not in chat
    assert "log.scrollTo(" in chat
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    assert "#log {" in css and "overflow-y: auto" in css


def test_chat_page_can_list_play_and_launch_challenges():
    chat = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "data.challenges" in chat and "challengesBox(" in chat
    assert "/challenge/${encodeURIComponent(code)}/answer" in chat
    assert "/challenge`" in chat and "window.confirm" in chat


def test_chat_page_offers_to_accept_pending_challenges():
    chat = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "showPendingInvites()" in chat and '"Accetta"' in chat
    assert "data.accept" in chat and "playChallenge(data.accept" in chat
