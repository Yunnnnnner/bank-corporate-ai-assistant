"""
信贷知识库系统 - 全局配置
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ─── 数据目录 ───────────────────────────────────────────────
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# ─── 向量数据库 (ChromaDB) ────────────────────────────────────
CHROMA_DIR      = DATA_DIR / "chroma"
CHROMA_COLLECTION = "credit_knowledge"

# ─── 元数据库 (SQLite) ────────────────────────────────────────
SQLITE_PATH     = DATA_DIR / "metadata.db"

# ─── Embedding 配置 ──────────────────────────────────────────
# 支持: "sentence_transformers" | "openai" | "zhipu"
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
EMBEDDING_MODEL   = os.getenv(
    "EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2"   # 支持中文的多语言模型
)
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ZHIPU_API_KEY     = os.getenv("ZHIPU_API_KEY", "")

# ─── LLM 配置 (Claude) ───────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS    = 2048

# ─── 文本分割配置 ─────────────────────────────────────────────
CHUNK_SIZE        = 500        # 每块最大字符数 (信贷条款较短，适当调小)
CHUNK_OVERLAP     = 80         # 重叠字符数 (~16%, 保证跨块语义连贯)
MIN_CHUNK_SIZE    = 50         # 过滤掉太短的碎片

# ─── 检索配置 ─────────────────────────────────────────────────
TOP_K             = 6          # 召回片段数量
SCORE_THRESHOLD   = 0.3        # 最低相似度阈值 (0-1)
HYBRID_ALPHA      = 0.6        # 语义检索权重 (1-alpha = 关键词权重)

# ─── 文档解析支持格式 ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls", ".csv"}

# ─── 信贷业务分类 ─────────────────────────────────────────────
CREDIT_CATEGORIES = [
    "监管文件",    # PBOC/CBIRC/CSRC 监管规定
    "内部政策",    # 行内信贷政策、授信政策
    "业务规则",    # 准入标准、风险评级、产品规则
    "法律法规",    # 相关法律、司法解释、合规要求
]
