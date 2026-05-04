"""
生成器 - 基于 Claude API
将检索到的信贷文件片段注入 Prompt，生成带引用来源的专业回答
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from config import LLM_MODEL
from retrieval.retriever import RetrievedContext
from generation.prompts import build_system_prompt, build_user_prompt
from generation.llm_client import make_llm_client


@dataclass
class RAGResponse:
    answer:    str
    sources:   List[dict]       # 引用来源列表
    contexts:  List[RetrievedContext]
    model:     str
    prompt_tokens:      int = 0
    completion_tokens:  int = 0


class CreditRAGGenerator:
    def __init__(self, model: str = LLM_MODEL):
        self._client = make_llm_client(model=model)
        self.model   = model

    def generate(
        self,
        question: str,
        contexts: List[RetrievedContext],
        chat_history: Optional[List[dict]] = None,
        user_profile: Optional[dict] = None,
    ) -> RAGResponse:
        # ── 构建知识片段 + 个性化 prompt ─────────────────────────
        context_block = self._build_context_block(contexts)
        system_prompt = build_system_prompt(user_profile)
        user_content  = build_user_prompt(context_block, question)

        # ── 多轮对话历史 ──────────────────────────────────────────
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_content})

        # ── 调用 LLM ────────────────────────────────────────────
        resp = self._client.chat(system=system_prompt, messages=messages)

        return RAGResponse(
            answer   = resp.text,
            sources  = self._extract_sources(contexts),
            contexts = contexts,
            model    = self.model,
            prompt_tokens     = resp.usage.input_tokens,
            completion_tokens = resp.usage.output_tokens,
        )

    def _build_context_block(self, contexts: List[RetrievedContext]) -> str:
        """将检索片段格式化为 Prompt 片段"""
        if not contexts:
            return "（未检索到相关内容）"
        blocks = []
        for i, ctx in enumerate(contexts, 1):
            blocks.append(
                f"[来源{i}] 《{ctx.filename}》 ({ctx.category})\n"
                f"{ctx.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _extract_sources(self, contexts: List[RetrievedContext]) -> List[dict]:
        seen, sources = set(), []
        for i, ctx in enumerate(contexts, 1):
            key = ctx.filename
            if key not in seen:
                seen.add(key)
                sources.append({
                    "ref":      f"来源{i}",
                    "filename": ctx.filename,
                    "category": ctx.category,
                    "path":     ctx.source_path,
                    "score":    round(ctx.score, 3),
                })
        return sources

    def stream_generate(
        self,
        question: str,
        contexts: List[RetrievedContext],
        chat_history: Optional[List[dict]] = None,
        user_profile: Optional[dict] = None,
    ):
        """流式生成，yield token 字符串"""
        context_block = self._build_context_block(contexts)
        system_prompt = build_system_prompt(user_profile)
        user_content  = build_user_prompt(context_block, question)

        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_content})

        with self._client.chat_stream(system=system_prompt, messages=messages) as stream:
            for text in stream:
                yield text
