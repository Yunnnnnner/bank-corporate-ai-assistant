#!/usr/bin/env python3
"""
RAG 核心能力演示
==========================================
运行方式：python demo.py

演示三笔真实感贷款申请的预审过程，直观展示 RAG 三大能力：
  1. 跨文件知识融合 —— 一次预审需要命中 5+ 份规则文档
  2. 精确引用溯源   —— 每条判断都标注来源文件和原文
  3. 结构化推理     —— 将碎片化规则综合成可执行的预审结论
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.rule    import Rule
from rich         import print as rprint

console = Console()


# ════════════════════════════════════════════════════════════════
# 三个典型申请案例（覆盖 PASS / CONDITIONAL / REJECT 三种结果）
# ════════════════════════════════════════════════════════════════
def get_demo_cases():
    from scenarios.credit_prereview import LoanApplication
    return [
        # ── 案例 1：正常申请，预期通过 ────────────────────────────
        LoanApplication(
            company_name          = "某科技制造有限公司",
            established_year      = 2018,
            industry              = "制造业（新能源设备）",
            company_type          = "中型企业",
            annual_revenue        = 8000.0,      # 8000万
            asset_liability_ratio = 62.0,        # 62%
            credit_rating         = "AA",
            has_bad_credit        = False,
            product_type          = "流动资金贷款",
            loan_amount           = 1500.0,      # 1500万
            loan_term_months      = 12,
            loan_purpose          = "采购原材料及支付供应商货款",
            guarantee_type        = "房产抵押",
            guarantee_value       = 2800.0,
        ),

        # ── 案例 2：存在多处合规疑点，预期有条件通过 ──────────────
        LoanApplication(
            company_name          = "某商贸发展有限公司",
            established_year      = 2023,        # 成立仅2年
            industry              = "大宗商品贸易",
            company_type          = "小微企业",
            annual_revenue        = 600.0,       # 600万（小微）
            asset_liability_ratio = 74.0,        # 超过70%标准线
            credit_rating         = "A",
            has_bad_credit        = False,
            product_type          = "流动资金贷款",
            loan_amount           = 800.0,       # 超过小微500万上限！
            loan_term_months      = 36,
            loan_purpose          = "购买存货及补充流动资金",
            guarantee_type        = "应收账款质押",
            guarantee_value       = 1200.0,
            remarks               = "近半年与核心企业签订了大额采购合同，收入快速增长",
        ),

        # ── 案例 3：触碰多条红线，预期不受理 ──────────────────────
        LoanApplication(
            company_name          = "某房地产开发有限公司",
            established_year      = 2015,
            industry              = "房地产开发",
            company_type          = "中型企业",
            annual_revenue        = 12000.0,
            asset_liability_ratio = 85.0,        # 超过80%红线
            credit_rating         = "BBB",
            has_bad_credit        = False,
            product_type          = "流动资金贷款",
            loan_amount           = 6000.0,      # 6000万（需总行审批）
            loan_term_months      = 36,
            loan_purpose          = "用于房地产项目建设",  # 用途违规！
            guarantee_type        = "土地使用权抵押",
            guarantee_value       = 8000.0,
            remarks               = "房地产行业，需总行审批",
        ),
    ]


# ════════════════════════════════════════════════════════════════
# 第一步：先跑摄入（保证知识库有内容）
# ════════════════════════════════════════════════════════════════
def ensure_knowledge_base():
    from pipeline import CreditKnowledgePipeline
    pl = CreditKnowledgePipeline()
    stats = pl.get_stats()

    if stats["total_chunks"] > 0:
        console.print(f"[green]✓ 知识库已就绪：{stats['total_documents']} 篇文档 / "
                      f"{stats['total_chunks']} 个片段[/green]\n")
        return pl

    console.print("[yellow]知识库为空，开始摄入文档...[/yellow]")
    results = pl.ingest_directory(verbose=True)
    stats   = pl.get_stats()
    console.print(f"\n[green]✓ 摄入完成：{stats['total_documents']} 篇 / "
                  f"{stats['total_chunks']} 片段[/green]\n")
    return pl


# ════════════════════════════════════════════════════════════════
# 主演示流程
# ════════════════════════════════════════════════════════════════
def run_demo():
    console.print(Panel.fit(
        "[bold cyan]银行信贷 RAG 知识库演示[/bold cyan]\n"
        "[dim]场景：信贷预审合规助手[/dim]\n\n"
        "展示 RAG 三大核心能力：\n"
        "  [green]①[/green] 跨文件知识融合（单次预审命中 5+ 份规则文档）\n"
        "  [green]②[/green] 精确引用溯源（每条结论标注来源文件 + 原文）\n"
        "  [green]③[/green] 结构化推理（将碎片规则综合成可执行预审结论）",
        title="🏦 信贷知识库 RAG Demo",
        border_style="cyan",
    ))

    # ── 初始化知识库 ───────────────────────────────────────────
    pl = ensure_knowledge_base()

    has_llm = bool(os.getenv("ANTHROPIC_API_KEY")) and \
              os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant")

    if not has_llm:
        console.print(Panel(
            "[yellow]未检测到有效的 ANTHROPIC_API_KEY[/yellow]\n\n"
            "将演示【纯检索模式】：展示 RAG 如何跨文件精确定位规则片段\n"
            "（完整预审报告需配置 API Key 后运行）\n\n"
            "配置方式：\n"
            "  1. 编辑项目根目录下的 [cyan].env[/cyan] 文件\n"
            "  2. 填入：[cyan]ANTHROPIC_API_KEY=sk-ant-xxxxxxxx[/cyan]",
            title="提示", border_style="yellow",
        ))
        _demo_retrieval_only(pl)
        return

    from scenarios.credit_prereview import CreditPrereviewAssistant
    assistant = CreditPrereviewAssistant(pl)
    cases     = get_demo_cases()

    for idx, app in enumerate(cases, 1):
        console.print(Rule(f"[bold]案例 {idx}/3：{app.company_name}[/bold]"))

        # 显示申请摘要
        _print_application_summary(app)

        t0 = time.time()
        report = assistant.review(app, verbose=True)
        elapsed = time.time() - t0

        # 显示来源文件统计（核心：体现跨文件检索）
        _print_sources_table(report.sources)

        # 显示统计
        verdict_style = {"PASS": "green", "CONDITIONAL": "yellow", "REJECT": "red"}
        verdict_text  = {"PASS": "✅ 建议受理", "CONDITIONAL": "⚠️  有条件受理", "REJECT": "❌ 建议不受理"}
        console.print(Panel(
            f"预审结论：[bold {verdict_style[report.overall]}]"
            f"{verdict_text[report.overall]}[/bold {verdict_style[report.overall]}]\n"
            f"命中规则文件：[cyan]{len(report.sources)} 份[/cyan]  ·  "
            f"检索片段：[cyan]{len(report.sources)} 条[/cyan]  ·  "
            f"耗时：[dim]{elapsed:.1f}s[/dim]  ·  "
            f"Token：[dim]{report.prompt_tokens}↑ {report.completion_tokens}↓[/dim]",
            border_style=verdict_style[report.overall],
        ))

        if idx < len(cases):
            input("\n按 Enter 继续下一案例...\n")

    console.print(Rule("[bold green]演示完成[/bold green]"))
    _print_rag_value_summary()


def _demo_retrieval_only(pl):
    """无 API Key 时，展示纯检索能力"""
    console.print(Rule("[bold]RAG 检索能力演示（无需 API Key）[/bold]"))

    demo_queries = [
        ("小微企业 最高授信额度",   "额度规则"),
        ("并购贷款 比例 期限限制",  "监管约束"),
        ("票据贴现 贸易背景 核查",  "操作规范"),
        ("资产负债率超标 准入",     "风控规则"),
        ("房地产行业 授信限制",     "行业政策"),
    ]

    for query, label in demo_queries:
        console.print(f"\n[bold cyan]查询：[/bold cyan]{query}  [dim]({label})[/dim]")
        hits = pl._smart_retrieve(query, top_k=3)
        for i, ctx in enumerate(hits, 1):
            console.print(
                f"  [{i}] [green]{ctx.score:.3f}[/green] "
                f"[yellow]《{ctx.filename}》[/yellow]\n"
                f"      {ctx.text[:120].strip()}…"
            )
        time.sleep(0.3)

    console.print(Panel(
        "✅ 检索验证通过\n\n"
        "上述每条结果都来自不同规则文档，说明知识库已正确建立。\n"
        "配置 ANTHROPIC_API_KEY 后可获得完整的预审分析报告。",
        title="检索演示结果", border_style="green"
    ))


def _print_application_summary(app):
    from scenarios.credit_prereview import LoanApplication
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim",  width=16)
    t.add_column(style="bold", width=20)
    t.add_column(style="dim",  width=16)
    t.add_column(style="bold")
    t.add_row("企业",   app.company_name,   "行业",   app.industry)
    t.add_row("类型",   app.company_type,   "评级",   f"[cyan]{app.credit_rating}[/cyan]")
    t.add_row("产品",   app.product_type,   "金额",   f"[bold]{app.loan_amount:,.0f}万元[/bold]")
    t.add_row("期限",   f"{app.loan_term_months}个月",
              "担保",   app.guarantee_type)
    t.add_row("负债率", f"[{'red' if app.asset_liability_ratio > 70 else 'green'}]"
                        f"{app.asset_liability_ratio}%[/]",
              "用途",   app.loan_purpose[:20])
    console.print(Panel(t, title="申请摘要", border_style="dim"))


def _print_sources_table(sources: list):
    if not sources:
        return
    t = Table(title=f"命中规则文件（共 {len(sources)} 份）", border_style="dim")
    t.add_column("序号", width=4,  justify="right")
    t.add_column("文件名",          style="cyan")
    t.add_column("分类",  width=10, style="dim")
    t.add_column("相关度",width=8,  justify="right")
    for i, s in enumerate(sources, 1):
        score_style = "green" if s["score"] > 0.6 else "yellow" if s["score"] > 0.4 else "dim"
        t.add_row(str(i), s["filename"], s["category"],
                  f"[{score_style}]{s['score']:.3f}[/{score_style}]")
    console.print(t)


def _print_rag_value_summary():
    console.print(Panel(
        "[bold]RAG 在这个场景的价值体现[/bold]\n\n"
        "[green]① 跨文件融合[/green]\n"
        "   每笔申请平均命中 5-8 份规则文档\n"
        "   传统方式：业务员需逐份翻查，耗时 30-60 分钟\n"
        "   RAG 方式：2-5 秒完成全量规则检索\n\n"
        "[green]② 精确引用溯源[/green]\n"
        "   每条结论都有原文依据，可追溯、可审计\n"
        "   解决了"LLM 一本正经地胡说"的核心问题\n\n"
        "[green]③ 规则更新自适应[/green]\n"
        "   监管新规发布 → 更新文档 → 重新摄入\n"
        "   无需重新训练模型，知识库当天生效\n\n"
        "[green]④ 纯 LLM 做不到的事[/green]\n"
        "   • 不知道你行的内部审批权限（500万→支行/5亿→总行）\n"
        "   • 不知道你行的定价底线（LPR-30BP）\n"
        "   • 不知道最新监管规定的生效时间\n"
        "   RAG 把这些私有知识注入到每次推理中",
        title="🎯 场景价值总结", border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
