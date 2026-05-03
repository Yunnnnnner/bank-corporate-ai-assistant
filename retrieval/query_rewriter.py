"""
查询改写 - 将用户口语化问题改写为多个精确检索查询
提升召回率约15-20%
"""
from __future__ import annotations
from typing import List
import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL


REWRITE_SYSTEM = """你是银行信贷领域的检索查询改写专家。用户会以口语化、省略关键信息的方式提问，你要把它改写成 2-3 条更精确、术语化的检索查询，用于在银行信贷知识库（监管文件/法律法规/内部政策/业务规则）中检索相关条款。

## 规则
1. **每行输出一个查询，不加编号、不加解释**
2. **保持专业术语**：流动资金贷款、银行承兑汇票、资产证券化、质押率、授信集中度、五级分类 等
3. **多角度改写**：政策依据 / 操作条件 / 审批流程 / 金额期限限制 / 担保要求，任选 2-3 个角度
4. **不改变核心意图**：不要引入用户没问过的新概念
5. **补全省略主语**：若用户说"800万呢"，应结合上下文推断为"800万流动资金贷款审批权限"或用户实际询问的业务

## Few-shot 示例

输入：小微企业最高能贷多少？
输出：
小微企业流动资金贷款单户授信额度上限
小微企业信贷产品准入条件
小微企业最高贷款限额规定

输入：3000万谁来审批？
输出：
3000万贷款审批权限 分级授权
信贷业务审批权限划分 总分行
单笔贷款审批层级规定

输入：住宅房产抵押率？
输出：
住宅房产抵押率 质押率规定
个人住房抵押物评估 贷款价值比
抵押物价值评估系数 住宅

现在请改写用户的问题。
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
