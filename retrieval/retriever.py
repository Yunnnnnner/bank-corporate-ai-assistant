"""
混合检索器 - 语义检索 + 关键词精准检索
融合策略: score = alpha * semantic_score + (1-alpha) * keyword_score
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from config import TOP_K, HYBRID_ALPHA, SCORE_THRESHOLD
from storage.vector_store import VectorStore
from storage.meta_store import MetaStore
from ingest.embedder import Embedder


@dataclass
class RetrievedContext:
    chunk_id:    str
    text:        str
    score:       float
    filename:    str
    category:    str
    source_path: str
    seq:         int
    retrieval_type: str    # "semantic" | "keyword" | "hybrid"


class HybridRetriever:
    """
    1. 语义检索（向量相似度）
    2. 关键词检索（FTS5 BM25）
    3. 融合排序（RRF 互惠排名融合）
    4. 过滤低分结果
    """

    def __init__(
        self,
        vector_store: VectorStore,
        meta_store:   MetaStore,
        embedder:     Embedder,
        top_k:        int   = TOP_K,
        alpha:        float = HYBRID_ALPHA,
        threshold:    float = SCORE_THRESHOLD,
    ):
        self.vs        = vector_store
        self.ms        = meta_store
        self.embedder  = embedder
        self.top_k     = top_k
        self.alpha     = alpha
        self.threshold = threshold

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int]    = None,
    ) -> List[RetrievedContext]:
        k = top_k or self.top_k

        # ── 1. 语义检索 ──────────────────────────────────────────
        q_emb     = self.embedder.embed_one(query)
        sem_hits  = self.vs.search(q_emb, top_k=k * 2, category=category,
                                   score_threshold=0.0)

        # ── 2. 关键词检索 ─────────────────────────────────────────
        kw_hits   = self.ms.keyword_search(
            self._build_fts_query(query), top_k=k * 2, category=category
        )

        # ── 3. RRF 融合 ───────────────────────────────────────────
        scores: dict[str, dict] = {}
        RRF_K = 60

        for rank, hit in enumerate(sem_hits):
            cid = hit.chunk_id
            if cid not in scores:
                scores[cid] = {"sem": 0.0, "kw": 0.0, "meta": hit}
            scores[cid]["sem"] = 1.0 / (RRF_K + rank + 1)

        for rank, hit in enumerate(kw_hits):
            cid = hit["chunk_id"]
            if cid not in scores:
                scores[cid] = {
                    "sem": 0.0, "kw": 0.0,
                    "meta": _kw_to_mock(hit),
                }
            scores[cid]["kw"] = 1.0 / (RRF_K + rank + 1)

        # ── 4. 合并得分 ───────────────────────────────────────────
        results = []
        for cid, s in scores.items():
            final = self.alpha * s["sem"] + (1 - self.alpha) * s["kw"]
            meta  = s["meta"]
            # 统一处理 SearchResult 或 mock dict
            if hasattr(meta, "metadata"):
                filename    = meta.metadata.get("filename", "")
                category_   = meta.metadata.get("category", "")
                source_path = meta.metadata.get("source_path", "")
                seq         = meta.metadata.get("seq", 0)
                text        = meta.text
            else:
                filename    = meta.get("filename", "")
                category_   = meta.get("category", "")
                source_path = meta.get("source_path", "")
                seq         = meta.get("seq", 0)
                text        = meta.get("text", "")
            rtype = (
                "hybrid"   if s["sem"] > 0 and s["kw"] > 0 else
                "semantic" if s["sem"] > 0 else "keyword"
            )
            results.append(RetrievedContext(
                chunk_id       = cid,
                text           = text,
                score          = final,
                filename       = filename,
                category       = category_,
                source_path    = source_path,
                seq            = seq,
                retrieval_type = rtype,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """将自然语言查询转换为 FTS5 查询（关键词 AND 组合）"""
        import jieba
        # 停用词（信贷领域常见无意义词）
        stop = {"的", "了", "是", "在", "和", "或", "有", "对", "与",
                "按", "应", "其", "该", "为", "等", "及", "以", "上", "下"}
        words = [
            w for w in jieba.cut(query, cut_all=False)
            if len(w) > 1 and w not in stop
        ]
        if not words:
            return query
        # FTS5 短语查询：每个词前缀匹配
        return " OR ".join(f'"{w}"' for w in words[:8])


def _kw_to_mock(hit: dict):
    """把关键词检索结果包装成类 SearchResult 的结构"""
    class _M:
        pass
    m = _M()
    m.text     = hit.get("text", "")
    m.metadata = hit
    return m
