"""
核心场景：信贷预审合规助手
==========================================
这是 RAG 最刚需的落地点：

  客户经理填写一笔申请基本要素
        │
        ▼
  系统并行检索 5 类知识库
  (准入条件 / 审批权限 / 定价规则 / 风控要求 / 监管规定)
        │
        ▼
  Claude 综合推理，生成结构化预审报告
  (每条结论标注来源文件 + 原文依据)
        │
        ▼
  客户经理 2 分钟内得到：
  ✅ 哪些条件已满足
  ⚠️  哪些条件需要补充材料
  ❌ 哪些硬性条款触碰红线
  📋 建议审批层级 + 所需材料清单

为什么这个场景最能体现 RAG：
  - 纯 LLM：不知道行内规则，无法判断审批权限/定价底线
  - 纯检索：能找到片段，但无法综合推理出"这笔业务能不能做"
  - RAG：两者结合，有依据地给出专业判断
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import Optional, List
from generation.llm_client import make_llm_client
from dotenv import load_dotenv

load_dotenv()

from pipeline import CreditKnowledgePipeline
from config import LLM_MODEL

# ─────────────────────────────────────────────────────────────────
# 申请要素数据模型
# ─────────────────────────────────────────────────────────────────
@dataclass
class LoanApplication:
    """客户经理录入的贷款申请基本要素"""
    # 借款人信息
    company_name:     str            # 企业名称
    established_year: int            # 成立年份（推算年限）
    industry:         str            # 所属行业
    company_type:     str            # 企业类型（小微/中型/大型）
    annual_revenue:   float          # 年营业收入（万元）
    asset_liability_ratio: float     # 资产负债率（%）
    credit_rating:    str            # 内部评级（AAA/AA/A/BBB…）
    has_bad_credit:   bool = False   # 是否有不良记录

    # 申请业务信息
    product_type:     str  = ""      # 产品类型（流动资金贷款/固定资产贷款/并购贷款/票据/ABS…）
    loan_amount:      float = 0.0    # 申请金额（万元）
    loan_term_months: int   = 12     # 期限（月）
    loan_purpose:     str  = ""      # 贷款用途
    guarantee_type:   str  = ""      # 担保方式（信用/保证/抵押/质押）
    guarantee_value:  float = 0.0    # 担保物估值（万元，无则0）

    # 补充信息
    remarks: str = ""                # 其他补充说明

    @property
    def loan_term_years(self) -> float:
        return self.loan_term_months / 12

    def to_description(self) -> str:
        return f"""【申请基本信息】
企业名称：{self.company_name}
行业：{self.industry} | 类型：{self.company_type}
成立时间：约{2025 - self.established_year}年前（{self.established_year}年）
年营业收入：{self.annual_revenue}万元
资产负债率：{self.asset_liability_ratio}%
内部评级：{self.credit_rating}
不良记录：{"有" if self.has_bad_credit else "无"}

【申请业务要素】
产品类型：{self.product_type}
申请金额：{self.loan_amount}万元（{self.loan_amount/10000:.2f}亿元）
贷款期限：{self.loan_term_months}个月（{self.loan_term_years:.1f}年）
贷款用途：{self.loan_purpose}
担保方式：{self.guarantee_type}
担保物估值：{self.guarantee_value}万元
{f"其他说明：{self.remarks}" if self.remarks else ""}"""


# ─────────────────────────────────────────────────────────────────
# 预审报告数据结构
# ─────────────────────────────────────────────────────────────────
@dataclass
class PrereviewReport:
    application: LoanApplication
    overall:     str            # PASS / CONDITIONAL / REJECT
    report_text: str            # 完整报告正文（含引用）
    sources:     List[dict]     # 检索到的来源
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ─────────────────────────────────────────────────────────────────
# 预审提示词（这是 RAG 的灵魂）
# ─────────────────────────────────────────────────────────────────
PREREVIEW_SYSTEM = """你是一位严格、审慎的银行信贷合规预审专员，负责在正式审批流程启动前对贷款申请进行合规性初审。

## 工作原则
1. **唯一依据**：以下方【知识库检索内容】为唯一判断依据，严禁凭经验或常识推断
2. **逐条引用**：每一个判断点必须在括号内注明 `来源文件名` + 关键原文
3. **缺失告知**：若知识库无明确规定，如实说“知识库中暂无明确规定，建议人工核查”，**不得默认通过**
4. **严谨态度**：发现风险必须明确指出，不可模糊带过或一笔略过
5. **数字精确**：所有金额、比例、期限必须精确到原文数值，不做“大概”“约”等模糊表述

## 报告输出格式（严格遵守，不得改动小节标题）

## 📋 信贷申请预审报告

### 一、基本准入审查
逐项检查：行业准入、企业年限、内部评级、资产负债率、不良记录、年营业收入等。格式：
- ✅ [条件名称]：[判断结论]（依据：[来源文件名] - "[原文关键句]"）
- ⚠️ [条件名称]：[需补充/关注事项]（依据：[来源文件名]）
- ❌ [条件名称]：[不满足原因]（依据：[来源文件名] - "[原文关键句]"）

### 二、额度与期限合规性
检查申请金额是否在产品额度上限内、期限是否符合期限规定，引用具体数字。

### 三、担保合规性
检查担保方式是否符合产品要求、担保物估值与贷款金额的覆盖率（抵押率/质押率）是否合规。

### 四、定价建议
根据客户评级和产品类型给出利率区间建议，引用定价规则中的加点范围。

### 五、审批权限建议
明确指出应由哪个层级审批（支行/分行/总行/董事会），引用授权规定原文。

### 六、风险提示
列出 2-5 条需要重点关注的风险点（如行业周期/集中度/用途合规/反洗钱等）。

### 七、预审结论
给出三选一结论（必须明确选一个，不得给出“视情况而定”）：
- ✅ **建议受理**（所有硬性条件均满足，可进入正式审批）
- ⚠️ **有条件受理**（需补充以下材料/满足以下条件后受理）：列出具体补齐项
- ❌ **建议不受理**（存在以下硬性障碍）：列出具体阻断项

### 八、所需材料清单
列出正式申报需提交的材料（营业执照、审计报告、担保物权证、用途证明等），引用材料清单条款。
"""


# ─────────────────────────────────────────────────────────────────
# 预审助手核心类
# ─────────────────────────────────────────────────────────────────
class CreditPrereviewAssistant:

    # 针对预审场景的专项检索查询集
    # 不同于通用问答，预审需要主动检索多个维度
    RETRIEVAL_QUERIES = [
        "{product_type} 准入条件",
        "{product_type} 申请材料清单",
        "{amount}万元 审批权限 分级授权",
        "{product_type} 贷款期限 上限",
        "{guarantee_type} 担保要求 质押率 抵押率",
        "{rating}评级 贷款利率 定价加点",
        "{industry} 行业 授信限制 集中度",
        "资产负债率 {ratio}% 准入门槛",
        "{product_type} 用途合规 不得用于",
        "反洗钱 受益所有人 {product_type}",
    ]

    def __init__(self, pipeline: Optional[CreditKnowledgePipeline] = None):
        self.pl     = pipeline or CreditKnowledgePipeline()
        self.client = make_llm_client()

    def review(self, app: LoanApplication, verbose: bool = True) -> PrereviewReport:
        """执行预审，返回完整报告"""

        if verbose:
            print(f"\n{'='*60}")
            print(f"  正在预审：{app.company_name} / {app.product_type}")
            print(f"{'='*60}")

        # ── Step 1: 构造检索查询（多维度主动检索）──────────────────
        queries = self._build_queries(app)
        if verbose:
            print(f"\n[RAG] 并行检索 {len(queries)} 个维度...\n")

        # ── Step 2: 并行检索，合并去重 ──────────────────────────────
        all_contexts = {}
        for q in queries:
            hits = self.pl._smart_retrieve(q, top_k=4)
            for ctx in hits:
                if ctx.chunk_id not in all_contexts:
                    all_contexts[ctx.chunk_id] = ctx
                    if verbose:
                        print(f"  ✦ [{ctx.score:.2f}] {ctx.filename[:28]:28s} | {ctx.text[:50].strip()}…")

        contexts = sorted(all_contexts.values(), key=lambda x: x.score, reverse=True)[:12]

        if verbose:
            print(f"\n[RAG] 共召回 {len(contexts)} 条相关规则片段，来自 "
                  f"{len({c.filename for c in contexts})} 份文件")

        # ── Step 3: 构建知识片段注入 Prompt ─────────────────────────
        knowledge_block = self._build_knowledge_block(contexts)

        user_prompt = f"""{app.to_description()}

---

【知识库检索内容】（这是你进行判断的唯一依据，共 {len(contexts)} 条片段）
{knowledge_block}

---

请严格按照系统提示中定义的 8 段格式，对上述贷款申请出具预审报告。注意：
1. 每段都必须至少引用一条 [来源X]
2. 如某段无相关检索内容，明确标注“知识库中暂无该维度规定，建议人工核查”
3. 第七段预审结论必须明确选一个（✅/⚠️/❌）"""

        # ── Step 4: 调用 Claude 生成报告 ────────────────────────────
        if verbose:
            print("\n[LLM] 正在生成预审报告...\n")
            print("─" * 60)

        full_text = ""
        usage     = None
        with self.client.chat_stream(
            system     = PREREVIEW_SYSTEM,
            max_tokens = 2500,
            messages   = [{"role": "user", "content": user_prompt}],
        ) as stream:
            for token in stream:
                if verbose:
                    print(token, end="", flush=True)
                full_text += token
            usage = stream.usage

        if verbose:
            print("\n" + "─" * 60)

        # ── Step 5: 解析结论 ─────────────────────────────────────────
        overall = self._parse_conclusion(full_text)
        sources = self._extract_sources(contexts)

        return PrereviewReport(
            application       = app,
            overall           = overall,
            report_text       = full_text,
            sources           = sources,
            prompt_tokens     = usage.input_tokens if usage else 0,
            completion_tokens = usage.output_tokens if usage else 0,
        )

    # ── 内部工具方法 ─────────────────────────────────────────────────
    def _build_queries(self, app: LoanApplication) -> List[str]:
        """构造多维度检索查询"""
        fmt = dict(
            product_type = app.product_type or "信贷",
            amount       = int(app.loan_amount),
            guarantee_type = app.guarantee_type or "担保",
            rating       = app.credit_rating,
            industry     = app.industry,
            ratio        = app.asset_liability_ratio,
        )
        queries = [q.format(**fmt) for q in self.RETRIEVAL_QUERIES]
        # 补充申请用途专项查询
        if app.loan_purpose:
            queries.append(f"贷款用途 {app.loan_purpose[:20]}")
        return queries

    def _build_knowledge_block(self, contexts) -> str:
        if not contexts:
            return "（未检索到相关规则）"
        blocks = []
        for i, ctx in enumerate(contexts, 1):
            blocks.append(
                f"[规则{i}]《{ctx.filename}》(相关度:{ctx.score:.2f})\n{ctx.text}"
            )
        return "\n\n".join(blocks)

    def _parse_conclusion(self, text: str) -> str:
        if "建议不受理" in text or "❌" in text.split("预审结论")[-1][:50]:
            return "REJECT"
        if "有条件受理" in text or "⚠️" in text.split("预审结论")[-1][:50]:
            return "CONDITIONAL"
        return "PASS"

    def _extract_sources(self, contexts) -> List[dict]:
        seen, sources = set(), []
        for ctx in contexts:
            if ctx.filename not in seen:
                seen.add(ctx.filename)
                sources.append({
                    "filename": ctx.filename,
                    "category": ctx.category,
                    "score":    round(ctx.score, 3),
                })
        return sources
