"""Tests for the static test page served at /. No network."""

from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import STATIC_DIR, create_app
from tests.fakes import FakeRetriever, FakeVerifier
from tests.test_api import FakeGenerator


def make_client():
    return TestClient(create_app(Database(":memory:"), lambda: FakeGenerator(), lambda: FakeRetriever(),
                                 lambda: FakeVerifier()))


def test_root_serves_the_test_page():
    response = make_client().get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Quanto ne sai di film e serie TV?" in response.text


def test_page_never_builds_html_from_server_text():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for unsafe in ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write"):
        assert unsafe not in html


def test_page_does_not_contain_correct_answer_fields():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "correct_index" not in html
