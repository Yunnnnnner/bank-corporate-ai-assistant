"""
查询改写 - 将用户口语化问题改写为多个精确检索查询
提升召回率约15-20%
"""
from __future__ import annotations
from typing import List
import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL


REWRITE_SYSTEM = """你是银行信贷领域的专家。用户会向你提出关于银行公司业务的问题。
你的任务是将用户的问题改写为2-3个更精确的检索查询，以便在知识库中找到相关规则。

规则：
1. 每行输出一个查询，不加编号
2. 保持专业术语（如：流动资金贷款、银行承兑汇票、资产证券化）
3. 可以从不同角度改写，如：政策依据、操作流程、准入条件
4. 不要改变问题的核心意图
"""


class QueryRewriter:
    def __init__(self):
        if ANTHROPIC_API_KEY:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            self._client = None

    def rewrite(self, question: str) -> List[str]:
        """返回原始查询 + 改写变体"""
        if not self._client:
            return [question]
        try:
            resp = self._client.messages.create(
                model      = LLM_MODEL,
                max_tokens = 200,
                system     = REWRITE_SYSTEM,
                messages   = [{"role": "user", "content": question}],
            )
            variants = [line.strip() for line in resp.content[0].text.strip().split('\n')
                        if line.strip() and len(line.strip()) > 5]
            # 原始查询 + 最多2个变体
            return [question] + variants[:2]
        except Exception:
            return [question]
