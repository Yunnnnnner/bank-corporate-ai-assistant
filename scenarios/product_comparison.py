"""
核心场景：产品方案智能比对
==========================================
从客户经理输入的「业务诉求 + 融资金额 + 场景描述」出发：

  Step 1 · 信号抽取：从口语化场景描述中识别 6 类信号
  Step 2 · 候选产品池：基于信号在知识库中圈出 3-5 个候选产品
  Step 3 · 8 维度并行检索：额度/期限/利率/担保/审批/材料/适用条件/风险
  Step 4 · 评分推荐：按场景吻合 / 期限 / 额度 / 成本 / 效率 加权打分

为什么这个场景体现 RAG：
  - 客户经理面对 10+ 个公司业务产品，无法一一翻手册
  - 仅靠 LLM：无法获取行内最新利率/额度/审批层级数据
  - 仅靠检索：无法跨产品对齐 8 个维度并出推荐
  - RAG：召回多份产品手册片段 → 结构化对齐 → 生成可执行的产品组合方案

对应文档：prompts/prompts.md §「产品方案智能比对」
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from pipeline import CreditKnowledgePipeline
from config import LLM_MODEL
from generation.llm_client import make_llm_client


# ─────────────────────────────────────────────────────────────────
# 业务诉求数据模型
# ─────────────────────────────────────────────────────────────────
@dataclass
class FinancingNeed:
    """客户经理录入的融资诉求（对应 demo_web 的 page-compare 表单）"""
    company_type:  str           # 企业类型（中型制造业/小微贸易/核心企业上游供应商等）
    amount:        float         # 融资金额（万元）
    description:   str           # 场景描述（口语化 textarea 内容）

    # 可选补充
    expected_term:    Optional[str] = None   # 期望期限（若描述里没明说）
    industry:         Optional[str] = None
    rating:           Optional[str] = None
    cost_sensitive:   bool = False           # 是否对成本敏感

    def to_brief(self) -> str:
        lines = [
            f"企业类型：{self.company_type}",
            f"融资金额：{self.amount} 万元",
            f"场景描述：{self.description}",
        ]
        if self.expected_term:
            lines.append(f"期望期限：{self.expected_term}")
        if self.industry:
            lines.append(f"行业：{self.industry}")
        if self.rating:
            lines.append(f"客户评级：{self.rating}")
        if self.cost_sensitive:
            lines.append("特殊要求：成本敏感")
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    need:        FinancingNeed
    report_text: str
    sources:     List[dict]
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ─────────────────────────────────────────────────────────────────
# 产品比对系统提示词（草案 B 落地版）
# ─────────────────────────────────────────────────────────────────
COMPARE_SYSTEM = """你是银行公司业务的产品方案咨询专家，专门帮客户经理为企业的具体融资诉求匹配最合适的信贷产品组合。
你的输出**直接进入客户经理工作台**，因此必须可追溯、可解释、可立即执行。

## 一、工作流程（严格按四步推理）

### Step 1 · 抽取诉求信号
从【融资场景描述】中识别以下 6 类信号（缺失时标"未提及"）：
- **场景类别**：应收账款融资 / 采购付款 / 经营周转 / 项目建设 / 票据贴现 / 跨境结算 / 其他
- **关键期限**：N 天/月（含账期、还款期、用款期）
- **核心交易凭证**：应收账款 / 银票 / 商票 / 信用证 / 采购合同 / 销售合同 / 无
- **核心企业关系**：是否依托核心企业 / 是否在行内白名单
- **利率敏感度**：是否明示"成本最低""利率敏感"等
- **特殊要求**：是否要求随借随还、线上化、快速放款等

### Step 2 · 候选产品池
基于信号在【知识库检索内容】中识别 3-5 个候选产品，可选范围参考但不限于：
银行保理 / 应收账款质押贷 / 供应链 e 点贷 / 供应链 e 销贷 / 流动资金贷款 / 银行承兑汇票 / 票据贴现 / 国内信用证 / 透支额度 / 固定资产贷款。
**只允许选择能在【知识库检索内容】中找到产品手册片段的产品**，找不到的不得编造。

### Step 3 · 8 维度横向对齐
为每个候选产品从知识库中摘取以下 8 个维度的具体取值：
1. 融资额度（含上限比例，如"应收账款 ×90%"）
2. 融资期限（具体天数/月数）
3. 利率/费率（具体加点区间或费率‰）
4. 担保要求（信用/保证/抵押/质押 + 担保物类型）
5. 审批层级（支行/分行/总行 + 金额阈值）
6. 所需材料（核心 3-5 项）
7. 适用条件（评级/白名单/行业等）
8. 风险点（产品固有风险）

任一维度若知识库无明确数据，写"知识库中暂无明确数据"，**不得用常识填补**。

### Step 4 · 评分与推荐
为每个产品给 0.00-1.00 的**匹配度**，评分依据公开列出：
- 场景吻合度（40%）：产品定位是否匹配【场景类别】
- 期限吻合度（20%）：产品期限是否覆盖【关键期限】
- 额度吻合度（15%）：产品额度是否覆盖【融资金额】
- 成本优势（15%）：相对其他候选的利率/费率
- 操作效率（10%）：审批层级 + 材料复杂度

按匹配度降序排列，最高者标 `⭐ 推荐`，次高 1-2 个标 `备选`，明显不匹配的标 `⚠ 不适用`。

## 二、严格约束（违反任一条即为不合格输出）
1. **唯一信源**：所有维度取值必须来自【知识库检索内容】；找不到就标"知识库中暂无明确数据"
2. **逐项引用**：每个产品的每个数值后都要带 `[来源X]` 标注；推荐结论必须引用 2 条以上来源
3. **禁用编造**：禁止给出【知识库】未提及的产品名、利率、额度、审批层级
4. **禁用句式**：禁止"通常""市面上""行业惯例""一般而言""大致""可能"等
5. **场景错配显式化**：若候选产品本不适用本场景（如本场景是应收账款，但候选含"票据贴现"），必须显式标注"⚠ 不适用 - 原因..."

## 三、输出格式（小节标题不得改动）

## 🎯 产品方案智能比对报告

### 一、诉求信号抽取
按 Step 1 的 6 个维度逐项列出（缺失标"未提及"）。

### 二、候选产品池（从知识库召回）
列出 3-5 个产品名 + 命中的来源片段编号。

### 三、横向对比矩阵
以 Markdown 表格列出 8 维度 × N 产品，每格末尾带 `[来源X]`。每行最优值用 **加粗** 标记。

### 四、匹配度评分
| 产品 | 场景40% | 期限20% | 额度15% | 成本15% | 效率10% | 总分 |
逐项给 0-1 分数，总分 = 加权和（保留两位小数）。

### 五、综合推荐
- 🎯 **推荐方案**：[产品名]
- **推荐理由**（3 条，每条带引用）
- **备选方案**：[产品名] - 适用情形（1 条带引用）
- **不适用产品**：[产品名] - 不适用原因

### 六、下一步建议
列出客户经理需要做的 2-4 件事（核实材料/确认白名单/补充资信等），每条注明操作理由。

## 四、引用质量示例（自检对照）

✅ 好的产品行：
| 银行保理 | 应收账款 ×90%（最高 900 万）[来源1] | ≤ 账期 90 天 [来源1] | LPR+50~80BP [来源3] | ... |

❌ 坏的产品行：
| 银行保理 | 比例较高 | 较短 | 适中 | ... |
"""


# ─────────────────────────────────────────────────────────────────
# 比对助手核心类
# ─────────────────────────────────────────────────────────────────
class ProductComparisonAssistant:

    # 通用维度查询（不依赖产品名）
    BASE_QUERIES = [
        "{scene} 融资产品 适用",
        "{amount}万元 融资 审批权限 分级",
        "公司业务 产品 利率定价 加点",
        "公司业务 产品 所需材料 审批流程",
    ]

    # 候选产品族（key=信号, value=触发的产品检索查询）
    PRODUCT_QUERIES_BY_SIGNAL = {
        "应收账款": [
            "银行保理 应收账款 融资比例 期限",
            "应收账款质押贷款 质押率 登记",
            "供应链 e 销贷 核心企业 应收",
        ],
        "采购付款": [
            "供应链 e 点贷 上游 采购融资",
            "国内信用证 贸易融资",
            "银行承兑汇票 采购付款",
        ],
        "短期周转": [
            "流动资金贷款 短期 期限",
            "银行承兑汇票 周转",
            "票据贴现 短期融资",
            "透支额度 随借随还",
        ],
        "项目建设": [
            "固定资产贷款 项目建设",
            "项目融资 期限 担保",
        ],
        "跨境结算": [
            "国内信用证 跨境结算",
            "贸易融资 进出口",
        ],
        # 默认（场景无法识别时用通用产品族）
        "_default": [
            "流动资金贷款 准入条件",
            "银行承兑汇票 适用",
            "供应链金融 产品体系",
        ],
    }

    def __init__(self, pipeline: Optional[CreditKnowledgePipeline] = None):
        self.pl     = pipeline or CreditKnowledgePipeline()
        self.client = make_llm_client()

    # ── 主入口 ──────────────────────────────────────────────────
    def compare(self, need: FinancingNeed, verbose: bool = True) -> ComparisonReport:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  产品方案比对：{need.company_type} · {need.amount} 万")
            print(f"{'='*60}")

        # ── Step A: 从描述抽取场景关键词（轻量启发式 + 兜底默认） ──
        scene = self._infer_scene(need.description)
        if verbose:
            print(f"\n[SIGNAL] 场景判定：{scene}")

        # ── Step B: 构造检索查询并并行召回 ────────────────────────
        queries = self._build_queries(need, scene)
        if verbose:
            print(f"[RAG] 并行检索 {len(queries)} 个查询...")

        all_contexts = {}
        for q in queries:
            hits = self.pl._smart_retrieve(q, top_k=3)
            for ctx in hits:
                if ctx.chunk_id not in all_contexts:
                    all_contexts[ctx.chunk_id] = ctx
                    if verbose:
                        print(f"  ✦ [{ctx.score:.2f}] {ctx.filename[:32]:32s} | {ctx.text[:48].strip()}…")

        contexts = sorted(all_contexts.values(), key=lambda x: x.score, reverse=True)[:14]
        if verbose:
            print(f"\n[RAG] 共召回 {len(contexts)} 条片段，来自 "
                  f"{len({c.filename for c in contexts})} 份文件")

        # ── Step C: 拼装 Prompt 调用 LLM ───────────────────────────
        knowledge_block = self._build_knowledge_block(contexts)
        user_prompt = f"""{need.to_brief()}

---

【知识库检索内容】（这是你比对推理的唯一依据，共 {len(contexts)} 条片段）
{knowledge_block}

---

请严格按照系统提示中定义的 4 步推理 + 6 段输出格式，对上述融资诉求出具产品比对报告。注意：
1. 横向对比矩阵每格末尾必须带 [来源X]
2. 知识库无对应数据的格子写"知识库中暂无明确数据"，不得编造
3. 第五段必须明确给出 1 个 ⭐ 推荐 + 至少 1 个备选
4. 第六段下一步建议必须可执行（具体材料/动作/责任人）"""

        if verbose:
            print("\n[LLM] 正在生成比对报告...\n" + "─" * 60)

        full_text = ""
        usage     = None
        with self.client.chat_stream(
            system     = COMPARE_SYSTEM,
            max_tokens = 3000,
            messages   = [{"role": "user", "content": user_prompt}],
        ) as stream:
            for token in stream:
                if verbose:
                    print(token, end="", flush=True)
                full_text += token
            usage = stream.usage
        if verbose:
            print("\n" + "─" * 60)

        return ComparisonReport(
            need        = need,
            report_text = full_text,
            sources     = self._extract_sources(contexts),
            prompt_tokens     = usage.input_tokens if usage else 0,
            completion_tokens = usage.output_tokens if usage else 0,
        )

    # ── 内部工具方法 ────────────────────────────────────────────
    def _infer_scene(self, desc: str) -> str:
        """启发式场景识别（兜底；最终判定由 LLM 做）"""
        d = desc or ""
        if any(k in d for k in ["应收账款", "账期", "保理", "回笼"]):
            return "应收账款"
        if any(k in d for k in ["采购", "上游", "付款", "供应商"]):
            return "采购付款"
        if any(k in d for k in ["项目", "基建", "厂房", "设备", "固定资产"]):
            return "项目建设"
        if any(k in d for k in ["跨境", "进出口", "外贸", "美元", "结汇"]):
            return "跨境结算"
        if any(k in d for k in ["短期", "周转", "30天", "60天", "临时"]):
            return "短期周转"
        return "_default"

    def _build_queries(self, need: FinancingNeed, scene: str) -> List[str]:
        fmt = dict(
            scene  = scene if scene != "_default" else "公司业务",
            amount = int(need.amount),
        )
        base    = [q.format(**fmt) for q in self.BASE_QUERIES]
        product = self.PRODUCT_QUERIES_BY_SIGNAL.get(scene,
                  self.PRODUCT_QUERIES_BY_SIGNAL["_default"])
        # 行业/评级补充
        extra = []
        if need.industry:
            extra.append(f"{need.industry} 行业 信贷产品 适用")
        if need.rating:
            extra.append(f"{need.rating}评级 客户 产品准入")
        return base + product + extra

    def _build_knowledge_block(self, contexts) -> str:
        if not contexts:
            return "（未检索到相关产品手册）"
        blocks = []
        for i, ctx in enumerate(contexts, 1):
            blocks.append(
                f"[来源{i}]《{ctx.filename}》(相关度:{ctx.score:.2f})\n{ctx.text}"
            )
        return "\n\n".join(blocks)

    def _extract_sources(self, contexts) -> List[dict]:
        seen, sources = set(), []
        for i, ctx in enumerate(contexts, 1):
            if ctx.filename not in seen:
                seen.add(ctx.filename)
                sources.append({
                    "ref":      f"来源{i}",
                    "filename": ctx.filename,
                    "category": ctx.category,
                    "score":    round(ctx.score, 3),
                })
        return sources


# ─────────────────────────────────────────────────────────────────
# CLI 自测：python scenarios/product_comparison.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_need = FinancingNeed(
        company_type = "中型制造业",
        amount       = 1000.0,
        description  = "企业有 1000 万应收账款（核心企业开具，账期 90 天），希望提前回笼资金，融资期限 6 个月以内",
        rating       = "AA",
    )
    asst = ProductComparisonAssistant()
    report = asst.compare(demo_need, verbose=True)
    print("\n\n>>> 来源摘要：")
    for s in report.sources:
        print(f"  - {s['filename']} ({s['category']}) score={s['score']}")
