"""
数据清洗器 - 去除噪音，标准化格式，过滤无效内容
针对银行信贷文件的特殊处理
"""
import re
from .parser import ParsedDocument


class DataCleaner:
    # 常见页眉/页脚模式（信贷文件特有）
    HEADER_FOOTER_PATTERNS = [
        r"第\s*\d+\s*页\s*[共/]\s*\d+\s*页",       # 第X页共Y页
        r"内部资料.*?请勿外传",
        r"版权所有.*?违者必究",
        r"[-─━]+\s*\d+\s*[-─━]+",                  # ── 1 ──
        r"confidential",
    ]
    # 无意义重复字符
    NOISE_PATTERNS = [
        r"_{3,}",           # 下划线
        r"\.{4,}",          # 省略号
        r"={3,}",           # 等号分割线
        r"\*{3,}",          # 星号
    ]

    def __init__(self):
        self._hf_re = re.compile(
            "|".join(self.HEADER_FOOTER_PATTERNS), re.IGNORECASE
        )
        self._noise_re = re.compile("|".join(self.NOISE_PATTERNS))

    def clean(self, doc: ParsedDocument) -> ParsedDocument:
        text = doc.raw_text
        text = self._remove_headers_footers(text)
        text = self._normalize_whitespace(text)
        text = self._remove_noise(text)
        text = self._normalize_punctuation(text)
        text = self._deduplicate_lines(text)
        doc.raw_text = text.strip()
        return doc

    def _remove_headers_footers(self, text: str) -> str:
        lines = text.splitlines()
        clean = [l for l in lines if not self._hf_re.search(l)]
        return "\n".join(clean)

    def _normalize_whitespace(self, text: str) -> str:
        # 多余空行 → 单空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 行内多余空格（保留缩进）
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def _remove_noise(self, text: str) -> str:
        return self._noise_re.sub("", text)

    def _normalize_punctuation(self, text: str) -> str:
        # 全半角数字/字母统一（常见于扫描件OCR结果）
        text = text.replace("，", "，").replace("。", "。")
        # 规范化百分号和利率格式
        text = re.sub(r"(\d+)\s*%", r"\1%", text)
        text = re.sub(r"(\d+)\s*‰", r"\1‰", text)
        return text

    def _deduplicate_lines(self, text: str) -> str:
        """去除完全重复的相邻段落（常见于PDF多列排版提取）"""
        lines = text.splitlines()
        seen, result = set(), []
        for line in lines:
            stripped = line.strip()
            # 短行不去重（可能是小标题）
            if len(stripped) < 10 or stripped not in seen:
                result.append(line)
                if len(stripped) >= 10:
                    seen.add(stripped)
        return "\n".join(result)
