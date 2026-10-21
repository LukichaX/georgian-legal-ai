"""
src/telemetry_store.py
=============================================================================
ცენტრალიზებული SQLite + JSONL საცავი მრავალმომხმარებლიანი ტესტირებისთვის:
- თითოეული მომხმარებლის (tester_id) ჩატების იზოლაცია
- ყველა კითხვა-პასუხის, როუტერის გადაწყვეტილებისა და მუხლების ავტომატური ჩაწერა
- 1-10 ბალიანი შეფასების, შეცდომის ტეგებისა და კომენტარების (Notes) შენახვა
- AI ანალიზისთვის სრული JSONL ექსპორტის გენერაცია
=============================================================================
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name in ("src", "backend", "utils") else _THIS_DIR
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = LOGS_DIR / "crowdsourced_eval.db"
JSONL_EXPORT_PATH = LOGS_DIR / "feedback_for_ai_analysis.jsonl"


class TelemetryStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=20.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    thread_id TEXT PRIMARY KEY,
                    tester_id TEXT NOT NULL,
                    tester_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    turn_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    tester_id TEXT NOT NULL,
                    tester_name TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_query TEXT NOT NULL,
                    assistant_answer TEXT NOT NULL,
                    router_decision_json TEXT,
                    retrieved_articles_json TEXT,
                    citation_status_json TEXT,
                    timings_json TEXT,
                    rating INTEGER,
                    issue_tags TEXT,
                    expected_article_hint TEXT,
                    tester_note TEXT,
                    created_at TEXT NOT NULL,
                    rated_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_tester ON chat_sessions(tester_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_thread ON chat_turns(thread_id)")
            conn.commit()

    def create_session(self, thread_id: str, tester_id: str, tester_name: str, title: str = "ახალი დიალოგი") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions (thread_id, tester_id, tester_name, title, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, tester_id, tester_name, title, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM chat_sessions WHERE thread_id = ?", (thread_id,)).fetchone()
            return dict(row) if row else {
                "thread_id": thread_id,
                "tester_id": tester_id,
                "tester_name": tester_name,
                "title": title,
                "created_at": now,
            }

    def update_session_title(self, thread_id: str, first_query: str) -> None:
        short_title = first_query.strip()[:42] + ("..." if len(first_query.strip()) > 42 else "")
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE thread_id = ? AND title = 'ახალი დიალოგი'",
                (short_title, thread_id),
            )
            conn.commit()

    def get_user_sessions(self, tester_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions WHERE tester_id = ? ORDER BY created_at DESC",
                (tester_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_session_turns(self, thread_id: str, tester_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chat_turns
                WHERE thread_id = ? AND tester_id = ?
                ORDER BY turn_index ASC
                """,
                (thread_id, tester_id),
            ).fetchall()
            turns = []
            for r in rows:
                d = dict(r)
                d["router_decision"] = json.loads(d["router_decision_json"] or "{}")
                d["retrieved_articles"] = json.loads(d["retrieved_articles_json"] or "[]")
                d["citation_status"] = json.loads(d["citation_status_json"] or "{}")
                d["timings"] = json.loads(d["timings_json"] or "{}")
                d["issue_tags_list"] = json.loads(d["issue_tags"] or "[]")
                turns.append(d)
            return turns

    def log_turn(
        self,
        turn_id: str,
        thread_id: str,
        tester_id: str,
        tester_name: str,
        turn_index: int,
        user_query: str,
        assistant_answer: str,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        cited_markers = set((meta.get("citation_status") or {}).get("cited_markers") or [])

        articles_compact = []
        for i, a in enumerate(meta.get("retrieved_articles") or [], 1):
            marker = f"E{i}"
            sc = round(float(a.get("rerank_score", a.get("score", 0.0))), 4)
            articles_compact.append({
                "marker": marker,
                "node_id": a.get("node_id", ""),
                "code_slug": a.get("code_slug", ""),
                "article_no": a.get("article_no", ""),
                "title": a.get("title", ""),
                "breadcrumb": a.get("breadcrumb", ""),
                "retrieval_tier": a.get("retrieval_tier", "TIER_2_CROSS_ENCODER"),
                "score": sc,
                "rerank_score": sc,
                "is_cited": marker in cited_markers,
            })

        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_turns (
                    turn_id, thread_id, tester_id, tester_name, turn_index,
                    user_query, assistant_answer, router_decision_json,
                    retrieved_articles_json, citation_status_json, timings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    thread_id,
                    tester_id,
                    tester_name,
                    turn_index,
                    user_query,
                    assistant_answer,
                    json.dumps(meta.get("router_decision") or {}, ensure_ascii=False),
                    json.dumps(articles_compact, ensure_ascii=False),
                    json.dumps(meta.get("citation_status") or {}, ensure_ascii=False),
                    json.dumps(meta.get("timings") or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
        self.update_session_title(thread_id, user_query)
        self._rebuild_jsonl_export()
        return articles_compact

    def save_feedback(
        self,
        turn_id: str,
        rating: int,
        issue_tags: List[str],
        expected_article_hint: str,
        tester_note: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                UPDATE chat_turns
                SET rating = ?, issue_tags = ?, expected_article_hint = ?, tester_note = ?, rated_at = ?
                WHERE turn_id = ?
                """,
                (
                    rating,
                    json.dumps(issue_tags, ensure_ascii=False),
                    expected_article_hint.strip(),
                    tester_note.strip(),
                    now,
                    turn_id,
                ),
            )
            conn.commit()
        self._rebuild_jsonl_export()

    def _rebuild_jsonl_export(self) -> str:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM chat_turns ORDER BY created_at ASC").fetchall()

        lines: List[str] = []
        for r in rows:
            record = {
                "turn_id": r["turn_id"],
                "thread_id": r["thread_id"],
                "tester": r["tester_name"],
                "turn_index": r["turn_index"],
                "timestamp": r["created_at"],
                "user_query": r["user_query"],
                "router_decision": json.loads(r["router_decision_json"] or "{}"),
                "retrieved_articles": json.loads(r["retrieved_articles_json"] or "[]"),
                "assistant_answer": r["assistant_answer"],
                "citation_status": json.loads(r["citation_status_json"] or "{}"),
                "timings": json.loads(r["timings_json"] or "{}"),
                "evaluation": {
                    "rating_1_to_10": r["rating"],
                    "issue_tags": json.loads(r["issue_tags"] or "[]"),
                    "expected_article_hint": r["expected_article_hint"] or "",
                    "tester_note": r["tester_note"] or "",
                },
            }
            lines.append(json.dumps(record, ensure_ascii=False))

        content = "\n".join(lines)
        JSONL_EXPORT_PATH.write_text(content, encoding="utf-8")
        return content

    def get_summary_stats(self) -> Dict[str, Any]:
        with self._lock, self._get_conn() as conn:
            total_turns = conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0]
            rated_turns = conn.execute("SELECT COUNT(*) FROM chat_turns WHERE rating IS NOT NULL").fetchone()[0]
            avg_rating = conn.execute("SELECT AVG(rating) FROM chat_turns WHERE rating IS NOT NULL").fetchone()[0]
            testers = conn.execute("SELECT COUNT(DISTINCT tester_id) FROM chat_turns").fetchone()[0]
        return {
            "total_turns": total_turns,
            "rated_turns": rated_turns,
            "avg_rating": round(avg_rating, 2) if avg_rating else 0.0,
            "unique_testers": testers,
        }