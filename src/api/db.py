"""SQLite storage for quizzes, questions and attempts. Correct answers never leave the server."""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.quiz.generator import Question

SCHEMA = """
CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
    quiz_id TEXT NOT NULL REFERENCES quizzes(id),
    position INTEGER NOT NULL,
    question TEXT NOT NULL,
    options TEXT NOT NULL,
    correct_index INTEGER NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (quiz_id, position)
);
CREATE TABLE IF NOT EXISTS asked_questions (
    title_key TEXT NOT NULL,
    question TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_quizzes (
    session_id TEXT NOT NULL,
    quiz_id TEXT NOT NULL REFERENCES quizzes(id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, quiz_id)
);
CREATE TABLE IF NOT EXISTS challenges (
    code TEXT PRIMARY KEY,
    quiz_id TEXT NOT NULL REFERENCES quizzes(id),
    challenger TEXT NOT NULL,
    opponent TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    quiz_id TEXT NOT NULL REFERENCES quizzes(id),
    player TEXT NOT NULL,
    points INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    total INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (quiz_id, player)
);
"""


@dataclass(frozen=True)
class StoredQuestion:
    question: str
    options: tuple[str, ...]
    correct_index: int
    source: str


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save_quiz(self, title: str, questions: list[Question]) -> str:
        quiz_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("INSERT INTO quizzes (id, title, created_at) VALUES (?, ?, ?)", (quiz_id, title, now))
            self.conn.executemany(
                "INSERT INTO questions (quiz_id, position, question, options, correct_index, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (quiz_id, i, q.question, json.dumps(list(q.options), ensure_ascii=False), q.correct_index, q.source)
                    for i, q in enumerate(questions)
                ],
            )
        return quiz_id

    def get_questions(self, quiz_id: str) -> list[StoredQuestion] | None:
        if not self.conn.execute("SELECT 1 FROM quizzes WHERE id = ?", (quiz_id,)).fetchone():
            return None
        rows = self.conn.execute(
            "SELECT question, options, correct_index, source FROM questions WHERE quiz_id = ? ORDER BY position",
            (quiz_id,),
        ).fetchall()
        return [
            StoredQuestion(r["question"], tuple(json.loads(r["options"])), r["correct_index"], r["source"])
            for r in rows
        ]

    def has_attempt(self, quiz_id: str, player: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM attempts WHERE quiz_id = ? AND player = ?", (quiz_id, player)
        ).fetchone() is not None

    def save_attempt(self, quiz_id: str, player: str, points: int, correct: int, total: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO attempts (id, quiz_id, player, points, correct, total, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, quiz_id, player, points, correct, total, now),
            )

    def get_attempt(self, quiz_id: str, player: str) -> dict | None:
        row = self.conn.execute(
            "SELECT player, points, correct, total FROM attempts WHERE quiz_id = ? AND player = ?",
            (quiz_id, player),
        ).fetchone()
        return dict(row) if row else None

    def create_challenge(self, code: str, quiz_id: str, challenger: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO challenges (code, quiz_id, challenger, created_at) VALUES (?, ?, ?, ?)",
                (code, quiz_id, challenger, now),
            )

    def get_challenge(self, code: str) -> dict | None:
        row = self.conn.execute(
            "SELECT code, quiz_id, challenger, opponent FROM challenges WHERE code = ?", (code,)
        ).fetchone()
        return dict(row) if row else None

    def set_challenge_opponent(self, code: str, opponent: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE challenges SET opponent = ? WHERE code = ?", (opponent, code))

    def remember_questions(self, title_key: str, questions: list[str]) -> None:
        """Stores the questions already asked on a title, so the next quiz does not repeat them."""
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.executemany(
                "INSERT INTO asked_questions (title_key, question, created_at) VALUES (?, ?, ?)",
                [(title_key, q, now) for q in questions],
            )

    def recent_questions(self, title_key: str, limit: int = 20) -> list[str]:
        rows = self.conn.execute(
            "SELECT question FROM asked_questions WHERE title_key = ? ORDER BY rowid DESC LIMIT ?",
            (title_key, limit),
        ).fetchall()
        return [r["question"] for r in rows]

    def add_chat_message(self, session_id: str, role: str, content: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )

    def recent_chat_messages(self, session_id: str, limit: int) -> list[dict]:
        """Last `limit` messages, oldest first."""
        rows = self.conn.execute(
            "SELECT role, content FROM (SELECT id, role, content FROM chat_messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_chat_quiz(self, session_id: str, quiz_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO chat_quizzes (session_id, quiz_id, created_at) VALUES (?, ?, ?)",
                (session_id, quiz_id, now),
            )

    def chat_quiz_ids(self, session_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT quiz_id FROM chat_quizzes WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [r["quiz_id"] for r in rows]

    def delete_chat_session(self, session_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM chat_quizzes WHERE session_id = ?", (session_id,))

    def purge_chat_older_than(self, hours: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.conn:
            cur = self.conn.execute("DELETE FROM chat_messages WHERE created_at < ?", (cutoff,))
            self.conn.execute("DELETE FROM chat_quizzes WHERE created_at < ?", (cutoff,))
        return cur.rowcount

    def leaderboard(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT player, SUM(points) AS points, COUNT(*) AS games FROM attempts "
            "GROUP BY player ORDER BY points DESC, player ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
