"""
元数据存储 - 基于 SQLite
负责：文档注册/去重、全文关键词检索、入库状态管理
"""
from __future__ import annotations
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import SQLITE_PATH


CREATE_DOCS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    category     TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    file_size    INTEGER,
    page_count   INTEGER DEFAULT 0,
    chunk_count  INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'pending',   -- pending|indexed|error
    error_msg    TEXT,
    created_at   TEXT,
    updated_at   TEXT
)
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    category     TEXT NOT NULL,
    filename     TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    text         TEXT NOT NULL,
    seq          INTEGER,
    created_at   TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
)
"""

CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(chunk_id UNINDEXED, text, content=chunks, content_rowid=rowid)
"""


class MetaStore:
    def __init__(self, db_path: Path = SQLITE_PATH):
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(CREATE_DOCS_TABLE)
            conn.execute(CREATE_CHUNKS_TABLE)
            conn.execute(CREATE_FTS_TABLE)

    # ── 文档管理 ──────────────────────────────────────────────────
    def is_indexed(self, doc_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            return row is not None and row["status"] == "indexed"

    def register_document(self, doc_id: str, filename: str, category: str,
                          source_path: str, file_size: int):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO documents
                (doc_id, filename, category, source_path, file_size, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?)
            """, (doc_id, filename, category, source_path, file_size, now, now))

    def mark_indexed(self, doc_id: str, page_count: int, chunk_count: int):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE documents SET status='indexed', page_count=?, chunk_count=?, updated_at=?
                WHERE doc_id=?
            """, (page_count, chunk_count, now, doc_id))

    def mark_error(self, doc_id: str, error_msg: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE documents SET status='error', error_msg=?, updated_at=?
                WHERE doc_id=?
            """, (error_msg, now, doc_id))

    def delete_document(self, doc_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def list_documents(self, category: Optional[str] = None) -> List[Dict]:
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE category=? ORDER BY created_at DESC",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Chunk 全文检索 ────────────────────────────────────────────
    def save_chunks(self, chunks):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            for c in chunks:
                conn.execute("""
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, category, filename, source_path, text, seq, created_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (c.chunk_id, c.doc_id, c.category, c.filename,
                      c.source_path, c.text, c.seq, now))
            # 重建 FTS 索引
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    def keyword_search(self, query: str, top_k: int = 10,
                       category: Optional[str] = None) -> List[Dict]:
        """基于 SQLite FTS5 的关键词精准检索"""
        with self._conn() as conn:
            if category:
                rows = conn.execute("""
                    SELECT c.chunk_id, c.doc_id, c.category, c.filename,
                           c.source_path, c.text, c.seq,
                           bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ? AND c.category = ?
                    ORDER BY score
                    LIMIT ?
                """, (query, category, top_k)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT c.chunk_id, c.doc_id, c.category, c.filename,
                           c.source_path, c.text, c.seq,
                           bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (query, top_k)).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total_docs   = conn.execute("SELECT COUNT(*) FROM documents WHERE status='indexed'").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            by_cat       = conn.execute("""
                SELECT category, COUNT(*) as cnt FROM documents
                WHERE status='indexed' GROUP BY category
            """).fetchall()
            return {
                "total_documents": total_docs,
                "total_chunks":    total_chunks,
                "by_category":     {r["category"]: r["cnt"] for r in by_cat},
            }
