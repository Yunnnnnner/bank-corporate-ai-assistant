"""
文档加载器 - 扫描目录，识别格式，分发到解析器
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator
from config import RAW_DIR, SUPPORTED_EXTENSIONS, CREDIT_CATEGORIES


@dataclass
class RawDocument:
    path: Path
    category: str          # 信贷业务分类
    filename: str
    extension: str
    file_size: int         # bytes
    doc_id: str            # SHA256 指纹，用于去重


class DocumentLoader:
    """递归扫描 data/raw/ 目录，按分类文件夹自动标注类别"""

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = raw_dir

    def scan(self, path: Path | None = None) -> Iterator[RawDocument]:
        """扫描目录，yield RawDocument"""
        scan_root = path or self.raw_dir
        for fp in scan_root.rglob("*"):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if fp.name.startswith(("~$", ".")):   # 跳过临时/隐藏文件
                continue
            yield self._make_doc(fp)

    def load_single(self, fp: Path) -> RawDocument:
        return self._make_doc(fp)

    def _make_doc(self, fp: Path) -> RawDocument:
        category = self._detect_category(fp)
        doc_id   = self._file_hash(fp)
        return RawDocument(
            path      = fp,
            category  = category,
            filename  = fp.name,
            extension = fp.suffix.lower(),
            file_size = fp.stat().st_size,
            doc_id    = doc_id,
        )

    def _detect_category(self, fp: Path) -> str:
        """从路径推断文档分类（按子目录名匹配）"""
        parts = fp.parts
        for cat in CREDIT_CATEGORIES:
            if cat in parts:
                return cat
        return "其他"

    @staticmethod
    def _file_hash(fp: Path, chunk=65536) -> str:
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            while buf := f.read(chunk):
                h.update(buf)
        return h.hexdigest()[:16]
