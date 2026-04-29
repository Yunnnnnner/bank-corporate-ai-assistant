# 银行公司信贷知识库 RAG 系统

基于 RAG（检索增强生成）技术构建的银行公司信贷业务规则与政策法规知识管理系统。

## 系统架构

```
用户提问
   │
   ▼
问题向量化 (Embedding)
   │
   ├── 语义检索 (ChromaDB 向量数据库)
   └── 关键词检索 (SQLite FTS5)
         │
         ▼
    RRF 融合排序
         │
         ▼
  注入 Claude Prompt
         │
         ▼
  生成带引用来源的专业回答
```

## 目录结构

```
├── config.py              # 全局配置
├── pipeline.py            # 核心流水线
├── cli.py                 # 命令行界面
├── requirements.txt
├── .env.example
├── ingest/
│   ├── loader.py          # 文档加载 (扫描+去重)
│   ├── parser.py          # 文档解析 (PDF/DOCX/Excel/TXT)
│   ├── cleaner.py         # 数据清洗
│   ├── chunker.py         # 智能分块 (按条款/章节)
│   └── embedder.py        # 向量化 (本地/OpenAI/智谱)
├── storage/
│   ├── vector_store.py    # ChromaDB 向量存储
│   └── meta_store.py      # SQLite 元数据+全文检索
├── retrieval/
│   └── retriever.py       # 混合检索 + RRF 融合
├── generation/
│   └── generator.py       # Claude 问答生成
├── api/
│   └── app.py             # FastAPI Web 服务
├── data/
│   └── raw/
│       ├── 监管文件/       # PBOC/银保监 监管规定
│       ├── 内部政策/       # 行内信贷政策
│       ├── 业务规则/       # 准入标准、产品规则
│       └── 法律法规/       # 法律、司法解释
└── scripts/
    ├── setup.sh
    └── ingest_and_test.py
```

## 快速开始

### 1. 安装

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY
```

### 3. 放入文档

将银行信贷相关文档（PDF/Word/Excel/TXT）放入对应分类目录：

```
data/raw/监管文件/     ← 监管规定、指导意见
data/raw/内部政策/     ← 行内信贷政策文件
data/raw/业务规则/     ← 准入标准、产品手册
data/raw/法律法规/     ← 法律法规、司法解释
```

### 4. 摄入文档

```bash
python cli.py ingest
```

### 5. 开始问答

```bash
# 交互式对话
python cli.py chat

# 单次提问
python cli.py ask "小微企业贷款的准入条件是什么？"

# 限定分类查询
python cli.py ask "不良贷款如何分类？" --cat 内部政策
```

### 6. 启动 Web API

```bash
python -m uvicorn api.app:app --reload --port 8000
# 访问 API 文档: http://localhost:8000/docs
```

## CLI 命令参考

| 命令 | 说明 |
|------|------|
| `python cli.py ingest` | 摄入 data/raw/ 全部文档 |
| `python cli.py ingest <路径>` | 摄入指定文件或目录 |
| `python cli.py ingest --force` | 强制重新摄入（覆盖已有） |
| `python cli.py ask "问题"` | 单次问答 |
| `python cli.py ask "问题" --no-llm` | 只检索，不调用 LLM |
| `python cli.py chat` | 多轮交互对话 |
| `python cli.py list` | 列出已索引文档 |
| `python cli.py stats` | 查看系统统计 |
| `python cli.py delete <doc_id>` | 删除文档 |

## Web API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| GET  | `/stats` | 系统统计 |
| GET  | `/documents` | 文档列表 |
| POST | `/ingest` | 上传文档 |
| POST | `/ask` | 问答（非流式） |
| POST | `/ask/stream` | 问答（SSE流式） |
| POST | `/retrieve` | 纯检索（不调LLM） |
| DELETE | `/documents/{id}` | 删除文档 |

## 核心技术选型

| 模块 | 技术 | 说明 |
|------|------|------|
| 向量数据库 | ChromaDB | 本地持久化，语义检索 |
| 关键词检索 | SQLite FTS5 | BM25 精准匹配 |
| 检索融合 | RRF (互惠排名融合) | 语义+关键词混合 |
| Embedding | sentence-transformers | 多语言中文支持，本地运行 |
| LLM | Claude Sonnet | 生成带引用来源的专业回答 |
| 文档解析 | pdfplumber + python-docx | 支持复杂版面 |
| 分块策略 | 语义感知 + 固定窗口 | 优先按条款/章节分割 |

## 典型问答示例

```
问: 小微企业信贷贷款利率如何定价？

答: 根据《公司信贷业务准入标准》第七条，贷款利率定价以当期LPR为基准：
  • 优质客户(AA及以上)：LPR + 0至30BP
  • 良好客户(A至AA-)：LPR + 30至80BP  
  • 一般客户(BBB至A-)：LPR + 80至150BP
  特殊规定：小微企业贷款利率加点不得超过同类业务平均水平的150%；
  绿色信贷客户可享受最低10BP的利率优惠。

[来源1] 公司信贷业务准入标准.md (业务规则) score=0.847
```
