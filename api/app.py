"""
FastAPI Web 服务
启动: python -m uvicorn api.app:app --reload --port 8000
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, tempfile

from pipeline import CreditKnowledgePipeline
from config import CREDIT_CATEGORIES, TOP_K

app = FastAPI(
    title    = "银行信贷知识库 RAG API",
    version  = "1.0.0",
    docs_url = "/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# 单例流水线（进程级共享）
_pipeline: Optional[CreditKnowledgePipeline] = None

def get_pl() -> CreditKnowledgePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CreditKnowledgePipeline()
    return _pipeline


# ─── 数据模型 ────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:    str
    category:    Optional[str] = None
    top_k:       int           = TOP_K
    stream:      bool          = False
    chat_history: Optional[List[dict]] = None

class AskResponse(BaseModel):
    answer:  str
    sources: List[dict]
    usage:   dict

class RetrieveRequest(BaseModel):
    query:    str
    category: Optional[str] = None
    top_k:    int = TOP_K


# ─── 路由 ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def get_stats():
    return get_pl().get_stats()

@app.get("/documents")
def list_documents(category: Optional[str] = Query(None)):
    return get_pl().ms.list_documents(category=category)

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    get_pl().delete_document(doc_id)
    return {"deleted": doc_id}

@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    category: str = Query("其他"),
):
    """上传并摄入单个文档"""
    if not any(file.filename.endswith(ext) for ext in
               [".pdf", ".docx", ".txt", ".md", ".xlsx", ".csv"]):
        raise HTTPException(400, "不支持的文件格式")

    # 保存到临时目录再摄入
    tmp_dir = Path(tempfile.mkdtemp())
    cat_dir = tmp_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    tmp_fp  = cat_dir / file.filename
    with open(tmp_fp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ok = get_pl().ingest_file(tmp_fp)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if ok:
        return {"status": "success", "filename": file.filename}
    else:
        raise HTTPException(500, "文档摄入失败，请检查文件格式")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """问答接口（非流式）"""
    pl   = get_pl()
    resp = pl.ask(req.question, category=req.category,
                  top_k=req.top_k, chat_history=req.chat_history)
    return AskResponse(
        answer  = resp.answer,
        sources = resp.sources,
        usage   = {
            "prompt_tokens":     resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        },
    )

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """流式问答接口（SSE）"""
    pl                 = get_pl()
    contexts, token_gen = pl.ask_stream(
        req.question,
        category=req.category,
        top_k=req.top_k,
        chat_history=req.chat_history,
    )

    def event_stream():
        for token in token_gen:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """纯检索接口（不调用 LLM）"""
    pl       = get_pl()
    contexts = pl.retrieve_only(req.query, category=req.category, top_k=req.top_k)
    return [
        {
            "chunk_id":      c.chunk_id,
            "text":          c.text,
            "score":         round(c.score, 4),
            "filename":      c.filename,
            "category":      c.category,
            "retrieval_type": c.retrieval_type,
        }
        for c in contexts
    ]

@app.get("/categories")
def get_categories():
    return CREDIT_CATEGORIES
