"""
文本分割器 - 信贷文件专用分块策略
优先按条款/章节边界切割，保证语义完整；回退到固定窗口
"""
import re
from dataclasses import dataclass, field
from typing import List
from .parser import ParsedDocument
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class TextChunk:
    chunk_id: str          # "{doc_id}_{seq:04d}"
    doc_id: str
    category: str
    filename: str
    source_path: str
    text: str
    seq: int               # 在文档中的顺序
    metadata: dict = field(default_factory=dict)


# 信贷/法规文件常见结构标识符（按优先级排序）
SECTION_PATTERNS = [
    # 章节：第一章、第二章 …
    re.compile(r"(?=第[一二三四五六七八九十百\d]+章\s)"),
    # 条款：第一条、第二条 … 或 第1条
    re.compile(r"(?=第[一二三四五六七八九十百\d]+条\s)"),
    # 大写数字列表：一、二、三、
    re.compile(r"(?=^[一二三四五六七八九十]+、)", re.MULTILINE),
    # 数字列表：1. 2. 3.
    re.compile(r"(?=^\d+\.\s)", re.MULTILINE),
    # Markdown 标题
    re.compile(r"(?=^#{1,3}\s)", re.MULTILINE),
]


class TextChunker:
    def __init__(
        self,
        chunk_size: int    = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_size: int      = MIN_CHUNK_SIZE,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_size      = min_size

    def chunk(self, doc: ParsedDocument) -> List[TextChunk]:
        text = doc.raw_text
        if not text:
            return []

        # 1. 尝试按条款/章节边界分割
        segments = self._semantic_split(text)

        # 2. 若段落仍过长，再做固定窗口细切
        fine_segments = []
        for seg in segments:
            if len(seg) > self.chunk_size:
                fine_segments.extend(self._fixed_split(seg))
            else:
                fine_segments.append(seg)

        # 3. 过滤太短的碎片
        fine_segments = [s for s in fine_segments if len(s.strip()) >= self.min_size]

        # 4. 构建 TextChunk 列表
        chunks = []
        for i, seg in enumerate(fine_segments):
            chunk_id = f"{doc.doc_id}_{i:04d}"
            chunks.append(TextChunk(
                chunk_id    = chunk_id,
                doc_id      = doc.doc_id,
                category    = doc.category,
                filename    = doc.filename,
                source_path = doc.source_path,
                text        = seg.strip(),
                seq         = i,
                metadata    = {"page_count": doc.page_count},
            ))
        return chunks

    def _semantic_split(self, text: str) -> List[str]:
        """依次尝试各结构标识符，取能产生合理分块的那个"""
        for pattern in SECTION_PATTERNS:
            parts = [p for p in pattern.split(text) if p.strip()]
            if 2 <= len(parts) <= len(text) // (self.min_size or 1):
                # 合并过短的段落到前一块
                merged: List[str] = []
                buf = ""
                for p in parts:
                    if len(buf) + len(p) <= self.chunk_size * 2:
                        buf += p
                    else:
                        if buf:
                            merged.append(buf)
                        buf = p
                if buf:
                    merged.append(buf)
                return merged
        # 无结构标识符：按段落（双换行）分割
        paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
        return paras if paras else [text]

    def _fixed_split(self, text: str) -> List[str]:
        """滑动窗口切分（有重叠），保证不截断句子"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # 向后找句子边界（。！？\n）
                for boundary in ("。\n", "。", "！", "？", "\n"):
                    idx = text.rfind(boundary, start, end)
                    if idx != -1:
                        end = idx + len(boundary)
                        break
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap
            if start <= 0:
                break
        return chunks
