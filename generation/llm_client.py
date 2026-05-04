"""
统一 LLM 客户端封装
==========================================
屏蔽 anthropic / deepseek（OpenAI-compatible） 两套 SDK 差异，
对外暴露：
  - chat(system, messages, **kwargs) -> LLMResponse
  - chat_stream(system, messages, **kwargs) -> 上下文管理器，迭代得到 token；退出后 .usage 可读

环境变量 LLM_PROVIDER 控制后端：anthropic | deepseek（默认 anthropic）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterator, Optional
from contextlib import contextmanager

from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P,
    ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
)


@dataclass
class LLMUsage:
    input_tokens:  int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text:  str
    usage: LLMUsage


# ─────────────────────────────────────────────────────────────────
# Anthropic 实现
# ─────────────────────────────────────────────────────────────────
class _AnthropicClient:
    def __init__(self, model: str = LLM_MODEL):
        import anthropic
        self._sdk   = anthropic
        self._cli   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model  = model

    def chat(self, system: str, messages: List[dict],
             max_tokens: int = LLM_MAX_TOKENS,
             temperature: float = LLM_TEMPERATURE,
             top_p: float = LLM_TOP_P) -> LLMResponse:
        resp = self._cli.messages.create(
            model      = self.model,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p      = top_p,
            system     = system,
            messages   = messages,
        )
        return LLMResponse(
            text  = resp.content[0].text,
            usage = LLMUsage(resp.usage.input_tokens, resp.usage.output_tokens),
        )

    @contextmanager
    def chat_stream(self, system: str, messages: List[dict],
                    max_tokens: int = LLM_MAX_TOKENS,
                    temperature: float = LLM_TEMPERATURE,
                    top_p: float = LLM_TOP_P):
        usage = LLMUsage()
        with self._cli.messages.stream(
            model      = self.model,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p      = top_p,
            system     = system,
            messages   = messages,
        ) as stream:
            class _Wrap:
                def __iter__(self_):    # 迭代 token
                    yield from stream.text_stream
                @property
                def usage(self_):
                    return usage
            try:
                yield _Wrap()
            finally:
                final = stream.get_final_message()
                usage.input_tokens  = final.usage.input_tokens
                usage.output_tokens = final.usage.output_tokens


# ─────────────────────────────────────────────────────────────────
# DeepSeek（OpenAI-compatible）实现
# ─────────────────────────────────────────────────────────────────
class _DeepSeekClient:
    def __init__(self, model: str = LLM_MODEL):
        from openai import OpenAI
        self._cli  = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = model

    @staticmethod
    def _to_openai_messages(system: str, messages: List[dict]) -> List[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            # Anthropic 允许 content 是 list of blocks，DeepSeek/OpenAI 需 str
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            out.append({"role": m["role"], "content": content})
        return out

    def chat(self, system: str, messages: List[dict],
             max_tokens: int = LLM_MAX_TOKENS,
             temperature: float = LLM_TEMPERATURE,
             top_p: float = LLM_TOP_P) -> LLMResponse:
        resp = self._cli.chat.completions.create(
            model       = self.model,
            messages    = self._to_openai_messages(system, messages),
            max_tokens  = max_tokens,
            temperature = temperature,
            top_p       = top_p,
        )
        u = resp.usage
        return LLMResponse(
            text  = resp.choices[0].message.content or "",
            usage = LLMUsage(u.prompt_tokens, u.completion_tokens),
        )

    @contextmanager
    def chat_stream(self, system: str, messages: List[dict],
                    max_tokens: int = LLM_MAX_TOKENS,
                    temperature: float = LLM_TEMPERATURE,
                    top_p: float = LLM_TOP_P):
        usage  = LLMUsage()
        stream = self._cli.chat.completions.create(
            model       = self.model,
            messages    = self._to_openai_messages(system, messages),
            max_tokens  = max_tokens,
            temperature = temperature,
            top_p       = top_p,
            stream      = True,
            stream_options = {"include_usage": True},
        )

        class _Wrap:
            def __iter__(self_):
                for chunk in stream:
                    # 末尾 usage chunk 没有 choices 内容
                    if chunk.usage:
                        usage.input_tokens  = chunk.usage.prompt_tokens
                        usage.output_tokens = chunk.usage.completion_tokens
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
            @property
            def usage(self_):
                return usage

        try:
            yield _Wrap()
        finally:
            try: stream.close()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────
# 工厂
# ─────────────────────────────────────────────────────────────────
def make_llm_client(model: Optional[str] = None):
    provider = LLM_PROVIDER
    m = model or LLM_MODEL
    if provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        return _DeepSeekClient(model=m)
    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY 未配置")
        return _AnthropicClient(model=m)
    raise ValueError(f"未知的 LLM_PROVIDER: {provider}（支持: anthropic | deepseek）")


def is_llm_configured() -> bool:
    if LLM_PROVIDER == "deepseek":
        return bool(DEEPSEEK_API_KEY)
    if LLM_PROVIDER == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    return False
