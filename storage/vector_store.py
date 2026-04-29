"""
向量存储 - 基于 ChromaDB
负责：存入向量 + 语义相似度召回
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, CHROMA_COLLECTION, TOP_K, SCORE_THRESHOLD
from ingest.chunker import TextChunk


class SearchResult:
    def __init__(self, chunk_id: str, text: str, score: float, metadata: dict):
        self.chunk_id = chunk_id
        self.text     = text
        self.score    = score
        self.metadata = metadata

    def __repr__(self):
        return f"SearchResult(score={self.score:.3f}, chunk_id={self.chunk_id})"


class VectorStore:
    def __init__(
        self,
        persist_dir: Path           = CHROMA_DIR,
        collection_name: str        = CHROMA_COLLECTION,
    ):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # 余弦相似度
        )

    # ── 写入 ─────────────────────────────────────────────────────
    def upsert(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """批量写入/更新"""
        if not chunks:
            return
        self._col.upsert(
            ids        = [c.chunk_id for c in chunks],
            embeddings = embeddings,
            documents  = [c.text for c in chunks],
            metadatas  = [{
                "doc_id":      c.doc_id,
                "category":    c.category,
                "filename":    c.filename,
                "source_path": c.source_path,
                "seq":         c.seq,
                **c.metadata,
            } for c in chunks],
        )

    def delete_by_doc(self, doc_id: str):
        """删除某文档的所有向量"""
        self._col.delete(where={"doc_id": doc_id})

    # ── 查询 ─────────────────────────────────────────────────────
    def search(
        self,
        query_embedding: List[float],
        top_k: int              = TOP_K,
        score_threshold: float  = SCORE_THRESHOLD,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        where = {"category": category} if category else None
        results = self._col.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine 距离 → 相似度
            score = 1.0 - dist
            if score >= score_threshold:
                output.append(SearchResult(
                    chunk_id = results["ids"][0][len(output)],
                    text     = doc,
                    score    = score,
                    metadata = meta,
                ))
        return sorted(output, key=lambda r: r.score, reverse=True)

    def count(self) -> int:
        return self._col.count()

    def list_doc_ids(self) -> List[str]:
        """列出所有已入库的 doc_id"""
        if self._col.count() == 0:
            return []
        res = self._col.get(include=["metadatas"])
        ids = {m["doc_id"] for m in res["metadatas"]}
        return sorted(ids)
