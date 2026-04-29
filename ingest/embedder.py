"""
向量化 (Embedding) - 支持多后端
- sentence_transformers: 本地运行，无需 API Key，支持中文
- openai: 调用 OpenAI text-embedding-ada-002
- zhipu: 调用智谱 embedding-2
"""
from __future__ import annotations
import os
from typing import List
from config import EMBEDDING_BACKEND, EMBEDDING_MODEL, OPENAI_API_KEY, ZHIPU_API_KEY


class Embedder:
    def __init__(
        self,
        backend: str = EMBEDDING_BACKEND,
        model: str   = EMBEDDING_MODEL,
    ):
        self.backend = backend
        self.model   = model
        self._client = None
        self._init()

    def _init(self):
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self._client = SentenceTransformer(self.model)
        elif self.backend == "openai":
            import openai
            openai.api_key = OPENAI_API_KEY
            self._client = openai
        elif self.backend == "zhipu":
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=ZHIPU_API_KEY)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，返回 List[List[float]]"""
        if not texts:
            return []
        if self.backend == "sentence_transformers":
            return self._client.encode(texts, show_progress_bar=False).tolist()
        elif self.backend == "openai":
            resp = self._client.embeddings.create(input=texts, model=self.model)
            return [d.embedding for d in resp.data]
        elif self.backend == "zhipu":
            embeddings = []
            for text in texts:
                resp = self._client.embeddings.create(input=text, model="embedding-2")
                embeddings.append(resp.data[0].embedding)
            return embeddings
        raise ValueError(f"Unknown backend: {self.backend}")

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        dims = {
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "text-embedding-ada-002": 1536,
            "embedding-2": 1024,
        }
        return dims.get(self.model, 384)
