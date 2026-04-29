"""
生成器 - 基于 Claude API
将检索到的信贷文件片段注入 Prompt，生成带引用来源的专业回答
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from retrieval.retriever import RetrievedContext

# 信贷助手系统提示词
SYSTEM_PROMPT = """你是一位专业的银行公司信贷合规顾问，拥有丰富的信贷政策、监管法规和业务规则知识。

你的职责：
1. 严格依据知识库中的文件内容回答问题
2. 对每一个重要结论标注来源文件名
3. 若知识库中无相关内容，明确告知"知识库中暂无该信息"，不凭空推测
4. 回答专业、准确、简洁，使用银行业通用术语
5. 涉及数字、比例、期限等关键参数时，务必精确引用原文

回答格式：
- 先给出直接结论
- 再展开说明依据
- 最后列出引用来源
"""


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
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model   = model

    def generate(
        self,
        question: str,
        contexts: List[RetrievedContext],
        chat_history: Optional[List[dict]] = None,
    ) -> RAGResponse:
        # ── 构建知识片段 ──────────────────────────────────────────
        context_block = self._build_context_block(contexts)

        # ── 构建用户提问 ──────────────────────────────────────────
        user_content = f"""请根据以下知识库内容回答问题。

【知识库内容】
{context_block}

【问题】
{question}

请给出专业、准确的回答，并在文末列出引用来源（格式：[来源X] 文件名 - 分类）。"""

        # ── 多轮对话历史 ──────────────────────────────────────────
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_content})

        # ── 调用 Claude ───────────────────────────────────────────
        response = self._client.messages.create(
            model      = self.model,
            max_tokens = LLM_MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        )

        answer = response.content[0].text
        sources = self._extract_sources(contexts)

        return RAGResponse(
            answer   = answer,
            sources  = sources,
            contexts = contexts,
            model    = self.model,
            prompt_tokens     = response.usage.input_tokens,
            completion_tokens = response.usage.output_tokens,
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

    def stream_generate(self, question: str, contexts: List[RetrievedContext]):
        """流式生成，yield token 字符串"""
        context_block = self._build_context_block(contexts)
        user_content  = f"""请根据以下知识库内容回答问题。

【知识库内容】
{context_block}

【问题】
{question}

请给出专业、准确的回答，并在文末列出引用来源。"""

        with self._client.messages.stream(
            model      = self.model,
            max_tokens = LLM_MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                yield text
