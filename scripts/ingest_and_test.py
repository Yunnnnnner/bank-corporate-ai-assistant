#!/usr/bin/env python3
"""
快速验证脚本 - 摄入示例文档并测试问答
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from pipeline import CreditKnowledgePipeline

def main():
    print("=== 初始化知识库系统 ===")
    pl = CreditKnowledgePipeline()

    print("\n=== 摄入示例文档 ===")
    results = pl.ingest_directory(verbose=True)
    print(f"结果: 成功={results['success']}, 跳过={results['skipped']}, 失败={results['failed']}")

    stats = pl.get_stats()
    print(f"\n知识库状态: {stats['total_documents']} 篇文档, {stats['total_chunks']} 个片段")

    if stats['total_chunks'] == 0:
        print("⚠️  知识库为空，请检查文档路径和解析配置")
        return

    print("\n=== 测试问答（仅检索，不调 LLM）===")
    test_queries = [
        "小微企业贷款准入条件",
        "逾期90天以上贷款如何分类",
        "抵押贷款房产抵押率是多少",
        "普惠型小微企业贷款不良率容忍度",
    ]
    for q in test_queries:
        print(f"\n问: {q}")
        contexts = pl.retrieve_only(q, top_k=3)
        if contexts:
            best = contexts[0]
            print(f"最佳匹配 ({best.score:.3f}): [{best.filename}]")
            print(f"  {best.text[:100].strip()}...")
        else:
            print("  未检索到相关内容")

    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_API_KEY") != "sk-ant-xxxx":
        print("\n=== 测试 LLM 问答 ===")
        q = "小微企业申请信贷需要满足哪些财务指标要求？"
        print(f"问: {q}")
        resp = pl.ask(q)
        print(f"答: {resp.answer[:300]}...")
        print(f"来源: {[s['filename'] for s in resp.sources]}")
    else:
        print("\n⚠️  未配置 ANTHROPIC_API_KEY，跳过 LLM 问答测试")
        print("   请在 .env 中设置: ANTHROPIC_API_KEY=sk-ant-xxxx")

if __name__ == "__main__":
    main()
