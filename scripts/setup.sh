#!/bin/bash
# 一键安装脚本
set -e

echo "=== 银行信贷知识库 RAG 系统安装 ==="

# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 配置环境变量
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件填入你的 API Key"
else
    echo "✓  .env 文件已存在"
fi

echo ""
echo "=== 安装完成 ==="
echo "下一步:"
echo "  1. 编辑 .env 填入 ANTHROPIC_API_KEY"
echo "  2. 将文档放入 data/raw/ 对应分类子目录"
echo "  3. 运行: python cli.py ingest    # 摄入文档"
echo "  4. 运行: python cli.py chat      # 开始问答"
