#!/usr/bin/env python3
"""
信贷知识库 CLI - 命令行界面
用法:
  python cli.py ingest             # 摄入 data/raw/ 下所有文档
  python cli.py ingest <文件路径>  # 摄入单个文件
  python cli.py ask "问题"         # 一次性问答
  python cli.py chat               # 交互式多轮对话
  python cli.py list               # 列出已索引文档
  python cli.py stats              # 查看系统统计
  python cli.py delete <doc_id>    # 删除文档
"""
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.markdown import Markdown
from rich import print as rprint

app     = typer.Typer(help="银行信贷知识库 RAG 系统")
console = Console()

def get_pipeline():
    from pipeline import CreditKnowledgePipeline
    return CreditKnowledgePipeline()


# ═══════════════════════════════════════════════════════════════════
# ingest
# ═══════════════════════════════════════════════════════════════════
@app.command()
def ingest(
    path: str = typer.Argument(None, help="文件或目录路径，默认扫描 data/raw/"),
    force: bool = typer.Option(False, "--force", "-f", help="强制重新摄入已索引文档"),
):
    """摄入文档到知识库"""
    pl = get_pipeline()
    if path:
        fp = Path(path)
        if not fp.exists():
            console.print(f"[red]路径不存在: {path}[/red]")
            raise typer.Exit(1)
        if fp.is_dir():
            results = pl.ingest_directory(fp, skip_existing=not force)
        else:
            ok = pl.ingest_file(fp)
            results = {"success": int(ok), "skipped": 0, "failed": int(not ok), "errors": []}
    else:
        results = pl.ingest_directory(skip_existing=not force)

    console.print(Panel(
        f"[green]成功: {results['success']}[/green]  "
        f"[yellow]跳过: {results['skipped']}[/yellow]  "
        f"[red]失败: {results['failed']}[/red]",
        title="摄入完成",
    ))


# ═══════════════════════════════════════════════════════════════════
# ask
# ═══════════════════════════════════════════════════════════════════
@app.command()
def ask(
    question: str = typer.Argument(..., help="查询问题"),
    category: str = typer.Option(None, "--cat", "-c", help="限定分类: 监管文件/内部政策/业务规则/法律法规"),
    top_k: int    = typer.Option(6, "--top-k", "-k", help="召回片段数"),
    no_llm: bool  = typer.Option(False, "--no-llm", help="只检索，不调用 LLM 生成"),
):
    """向知识库提问"""
    pl = get_pipeline()

    if no_llm:
        contexts = pl.retrieve_only(question, category=category, top_k=top_k)
        _print_contexts(contexts)
        return

    console.print(f"\n[bold blue]问题:[/bold blue] {question}\n")
    console.print("[dim]检索中...[/dim]")

    resp = pl.ask(question, category=category, top_k=top_k)

    console.print(Panel(Markdown(resp.answer), title="回答", border_style="green"))
    _print_sources(resp.sources)
    console.print(
        f"\n[dim]Token 用量: 输入 {resp.prompt_tokens} | 输出 {resp.completion_tokens}[/dim]"
    )


# ═══════════════════════════════════════════════════════════════════
# chat (交互式多轮对话)
# ═══════════════════════════════════════════════════════════════════
@app.command()
def chat(
    category: str = typer.Option(None, "--cat", "-c", help="限定文档分类"),
):
    """交互式多轮对话"""
    pl = get_pipeline()
    history = []

    console.print(Panel(
        "[bold green]银行信贷知识库问答系统[/bold green]\n"
        "输入问题开始对话，输入 [yellow]quit[/yellow] 或 [yellow]exit[/yellow] 退出\n"
        "输入 [yellow]clear[/yellow] 清空对话历史",
        title="欢迎使用"
    ))

    while True:
        try:
            q = console.input("\n[bold cyan]你:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "退出"):
            console.print("[dim]再见！[/dim]")
            break
        if q.lower() in ("clear", "清空"):
            history = []
            console.print("[dim]对话历史已清空[/dim]")
            continue

        console.print("[dim]思考中...[/dim]")
        contexts, token_gen = pl.ask_stream(q, category=category)

        console.print("\n[bold green]助手:[/bold green] ", end="")
        full_answer = ""
        for token in token_gen:
            console.print(token, end="", highlight=False)
            full_answer += token
        console.print()

        # 更新对话历史（用于多轮上下文）
        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant", "content": full_answer})
        # 保留最近 10 轮
        if len(history) > 20:
            history = history[-20:]

        if contexts:
            _print_sources([{
                "ref":      f"来源{i+1}",
                "filename": c.filename,
                "category": c.category,
                "score":    round(c.score, 3),
                "path":     c.source_path,
            } for i, c in enumerate(contexts[:3])])


# ═══════════════════════════════════════════════════════════════════
# list
# ═══════════════════════════════════════════════════════════════════
@app.command(name="list")
def list_docs(
    category: str = typer.Option(None, "--cat", "-c", help="按分类筛选"),
):
    """列出已索引的文档"""
    pl = get_pipeline()
    docs = pl.ms.list_documents(category=category)
    if not docs:
        console.print("[yellow]知识库为空，请先运行 ingest 摄入文档[/yellow]")
        return

    table = Table(title=f"已索引文档 ({len(docs)} 个)")
    table.add_column("Doc ID",    style="dim",     width=18)
    table.add_column("文件名",    style="bold",    max_width=35)
    table.add_column("分类",      style="cyan",    width=10)
    table.add_column("状态",      width=8)
    table.add_column("分块数",    justify="right", width=6)
    table.add_column("页数",      justify="right", width=6)

    STATUS_STYLE = {"indexed": "green", "error": "red", "pending": "yellow"}
    for d in docs:
        table.add_row(
            d["doc_id"],
            d["filename"],
            d["category"],
            f"[{STATUS_STYLE.get(d['status'],'white')}]{d['status']}[/]",
            str(d.get("chunk_count", 0)),
            str(d.get("page_count", 0)),
        )
    console.print(table)


# ═══════════════════════════════════════════════════════════════════
# stats
# ═══════════════════════════════════════════════════════════════════
@app.command()
def stats():
    """查看系统统计信息"""
    pl = get_pipeline()
    s  = pl.get_stats()

    console.print(Panel(
        f"[bold]文档总数:[/bold] {s['total_documents']}\n"
        f"[bold]片段总数:[/bold] {s['total_chunks']}\n"
        f"[bold]向量总数:[/bold] {s['vector_count']}\n"
        f"[bold]Embedding:[/bold] {s['embedding_model']}\n"
        f"[bold]LLM 模型:[/bold] {s['llm_model']}",
        title="系统统计"
    ))

    if s.get("by_category"):
        table = Table(title="按分类统计")
        table.add_column("分类")
        table.add_column("文档数", justify="right")
        for cat, cnt in s["by_category"].items():
            table.add_row(cat, str(cnt))
        console.print(table)


# ═══════════════════════════════════════════════════════════════════
# delete
# ═══════════════════════════════════════════════════════════════════
@app.command()
def delete(
    doc_id: str = typer.Argument(..., help="要删除的文档 ID"),
    yes: bool   = typer.Option(False, "--yes", "-y", help="跳过确认"),
):
    """从知识库删除文档"""
    if not yes:
        confirm = typer.confirm(f"确认删除文档 {doc_id}？")
        if not confirm:
            raise typer.Abort()
    pl = get_pipeline()
    pl.delete_document(doc_id)
    console.print(f"[green]已删除: {doc_id}[/green]")


# ── 辅助函数 ─────────────────────────────────────────────────────
def _print_sources(sources: list):
    if not sources:
        return
    console.print("\n[dim]引用来源:[/dim]")
    for s in sources:
        console.print(
            f"  [dim]{s['ref']}[/dim] [cyan]{s['filename']}[/cyan] "
            f"({s['category']}) [dim]score={s['score']}[/dim]"
        )

def _print_contexts(contexts):
    for i, c in enumerate(contexts, 1):
        console.print(Panel(
            c.text,
            title=f"[{i}] {c.filename} ({c.category}) score={c.score:.3f}",
            border_style="dim",
        ))


if __name__ == "__main__":
    app()
