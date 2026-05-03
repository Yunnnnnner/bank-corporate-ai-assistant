"""
生成器 - 基于 Claude API
将检索到的信贷文件片段注入 Prompt，生成带引用来源的专业回答
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P
from retrieval.retriever import RetrievedContext

# 信贷助手系统提示词
SYSTEM_PROMPT = """你是一位专业、审慎、可追溯的银行公司信贷合规顾问，负责基于企业信贷政策、监管法规、法律法规和业务规则知识库回答问题。

核心原则：
1. 知识库优先：所有政策、额度、期限、利率、担保、准入、审批权限、材料清单等结论，必须来自【知识库内容】。
2. 多轮上下文：如果用户追问中出现“这个/上述/那/如果是/再问一下”等指代，应结合对话历史识别真实问题，但最终答案仍必须依据本轮检索内容。
3. 引用约束：每个关键结论后都要标注来源编号，例如“（依据：[来源2]）”；涉及数字、比例、期限、条件时必须引用原文关键句或精确参数。
4. 不足则说明：若检索内容不能支撑结论，明确说“知识库中暂无该信息/暂无明确规定”，并给出建议补充检索的问题，不得凭经验编造。
5. 冲突处理：若多个来源存在差异，优先级为监管文件 > 法律法规 > 内部政策 > 业务规则，并说明冲突点。
6. 回答风格：专业、直接、结构化，优先给可执行结论，不输出泛泛而谈的百科解释。

回答格式：
- 直接结论：用 1-3 句话回答用户问题
- 依据说明：分点列出规则、条件、限制、适用范围，并逐条引用来源
- 风险/待确认：列出知识库未覆盖或需要人工核验的点
- 引用来源：按 [来源X] 文件名 - 分类 的格式列出
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
        user_content = f"""请作为银行公司信贷合规顾问，基于本轮检索到的知识库内容回答用户问题。

你需要先理解【当前问题】是否承接了上文；若是追问，请结合对话历史补全省略对象。但不要使用对话历史中的事实替代知识库证据。

【知识库内容】
{context_block}

【当前问题】
{question}

请输出结构化回答，并确保所有关键结论都能在【知识库内容】中找到对应依据。"""

        # ── 多轮对话历史 ──────────────────────────────────────────
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_content})

        # ── 调用 Claude ───────────────────────────────────────────
        response = self._client.messages.create(
            model      = self.model,
            max_tokens = LLM_MAX_TOKENS,
            temperature = LLM_TEMPERATURE,
            top_p      = LLM_TOP_P,
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

    def stream_generate(
        self,
        question: str,
        contexts: List[RetrievedContext],
        chat_history: Optional[List[dict]] = None,
    ):
        """流式生成，yield token 字符串"""
        context_block = self._build_context_block(contexts)
        user_content  = f"""请作为银行公司信贷合规顾问，基于本轮检索到的知识库内容回答用户问题。

你需要先理解【当前问题】是否承接了上文；若是追问，请结合对话历史补全省略对象。但不要使用对话历史中的事实替代知识库证据。

【知识库内容】
{context_block}

【当前问题】
{question}

请输出结构化回答，并确保所有关键结论都能在【知识库内容】中找到对应依据。"""

        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_content})

        with self._client.messages.stream(
            model      = self.model,
            max_tokens = LLM_MAX_TOKENS,
            temperature = LLM_TEMPERATURE,
            top_p      = LLM_TOP_P,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
