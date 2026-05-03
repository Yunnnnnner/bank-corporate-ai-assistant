"""
核心流水线 - 把所有模块串联起来
摄入: load → parse → clean → chunk → embed → store
检索: embed_query → hybrid_retrieve → generate
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from config import RAW_DIR, TOP_K
from ingest.loader  import DocumentLoader
from ingest.parser  import DocumentParser
from ingest.cleaner import DataCleaner
from ingest.chunker import TextChunker
from ingest.embedder import Embedder
from storage.vector_store import VectorStore
from storage.meta_store   import MetaStore
from retrieval.retriever     import HybridRetriever, RetrievedContext
from retrieval.query_rewriter import QueryRewriter
from generation.generator    import CreditRAGGenerator, RAGResponse


class CreditKnowledgePipeline:
    def __init__(self):
        self.loader    = DocumentLoader()
        self.parser    = DocumentParser()
        self.cleaner   = DataCleaner()
        self.chunker   = TextChunker()
        self.embedder  = Embedder()
        self.vs        = VectorStore()
        self.rewriter  = QueryRewriter()
        self.ms        = MetaStore()
        self.retriever = HybridRetriever(self.vs, self.ms, self.embedder)
        self.generator = CreditRAGGenerator()

    # ═══════════════════════════════════════════════════════════════
    # 摄入流水线
    # ═══════════════════════════════════════════════════════════════

    def ingest_directory(
        self,
        directory: Path = RAW_DIR,
        skip_existing: bool = True,
        verbose: bool = True,
    ) -> dict:
        raw_docs = list(self.loader.scan(directory))
        results  = {"success": 0, "skipped": 0, "failed": 0, "errors": []}

        pbar = tqdm(raw_docs, desc="摄入文档", disable=not verbose)
        for raw_doc in pbar:
            pbar.set_description(f"处理: {raw_doc.filename[:30]}")
            if skip_existing and self.ms.is_indexed(raw_doc.doc_id):
                results["skipped"] += 1
                continue
            ok = self._ingest_one(raw_doc)
            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1

        return results

    def ingest_file(self, fp: Path) -> bool:
        raw_doc = self.loader.load_single(fp)
        return self._ingest_one(raw_doc)

    def _ingest_one(self, raw_doc) -> bool:
        try:
            # 1. 注册
            self.ms.register_document(
                raw_doc.doc_id, raw_doc.filename, raw_doc.category,
                str(raw_doc.path), raw_doc.file_size
            )
            # 2. 解析
            parsed = self.parser.parse(raw_doc)
            if parsed.parse_error:
                self.ms.mark_error(raw_doc.doc_id, parsed.parse_error)
                return False
            if not parsed.raw_text.strip():
                self.ms.mark_error(raw_doc.doc_id, "空文档")
                return False
            # 3. 清洗
            cleaned = self.cleaner.clean(parsed)
            # 4. 分块
            chunks = self.chunker.chunk(cleaned)
            if not chunks:
                self.ms.mark_error(raw_doc.doc_id, "分块结果为空")
                return False
            # 5. 向量化
            texts      = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            # 6. 存储
            self.vs.upsert(chunks, embeddings)
            self.ms.save_chunks(chunks)
            self.ms.mark_indexed(raw_doc.doc_id, parsed.page_count, len(chunks))
            return True
        except Exception as e:
            self.ms.mark_error(raw_doc.doc_id, str(e))
            return False

    def delete_document(self, doc_id: str):
        self.vs.delete_by_doc(doc_id)
        self.ms.delete_document(doc_id)

    # ═══════════════════════════════════════════════════════════════
    # 检索+生成
    # ═══════════════════════════════════════════════════════════════

    def ask(
        self,
        question: str,
        category: Optional[str]    = None,
        top_k: int                 = TOP_K,
        chat_history: Optional[list] = None,
    ) -> RAGResponse:
        contexts = self._smart_retrieve(question, category=category, top_k=top_k)
        return self.generator.generate(question, contexts, chat_history)

    def ask_stream(
        self,
        question: str,
        category: Optional[str] = None,
        top_k: int = TOP_K,
        chat_history: Optional[list] = None,
    ):
        """流式问答，返回 (contexts, token_generator)"""
        contexts = self._smart_retrieve(question, category=category, top_k=top_k)
        return contexts, self.generator.stream_generate(question, contexts, chat_history)

    def retrieve_only(
        self,
        question: str,
        category: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> List[RetrievedContext]:
        return self._smart_retrieve(question, category=category, top_k=top_k)

    def _smart_retrieve(
        self,
        question: str,
        category: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> List[RetrievedContext]:
        """多查询改写 + 合并去重检索"""
        queries = self.rewriter.rewrite(question)
        seen: dict[str, RetrievedContext] = {}
        for q in queries:
            for ctx in self.retriever.retrieve(q, category=category, top_k=top_k):
                if ctx.chunk_id not in seen:
                    seen[ctx.chunk_id] = ctx
                else:
                    seen[ctx.chunk_id].score = max(seen[ctx.chunk_id].score, ctx.score)
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]

    def get_stats(self) -> dict:
        db_stats = self.ms.get_stats()
        return {
            **db_stats,
            "vector_count": self.vs.count(),
            "embedding_model": self.embedder.model,
            "llm_model": self.generator.model,
        }
