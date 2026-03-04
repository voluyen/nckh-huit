import re
import unicodedata
from ftfy import fix_text
from typing import List

# ---------- Regex chuẩn hóa cấu trúc ----------
RE_NEW_SECTION = re.compile(
    r"^(Câu\s+\d+|Chương\s+\d+|Mục\s+\d+|\d+[.)]|[a-zA-Z][.)]|[-*•]|###)",
    re.IGNORECASE
)

RE_TABLE_MARKDOWN = re.compile(r"^\|.*\|$")
RE_SCHEMA_LINE = re.compile(r".*\(.*")          # NHANVIEN(
RE_SCHEMA_CLOSE = re.compile(r".*\).*")         # ... )
RE_SENTENCE_END = re.compile(r"[.!?:;…]$")

RE_OPERATOR = re.compile(
    r"\b(PROJECT|SELECT|INTERSECT|JOIN|DIVIDE)\b",
    re.IGNORECASE
)

# ---------- Hàm chính ----------
def clean_text(text: str) -> str:
    if not text:
        return ""

    # ===== 1. Encoding & Unicode phục hồi =====
    text = fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\xa0", " ")

    # ===== 2. Glyph → semantic token (embedding-safe) =====
    glyph_map = {
        "\uf050": " PROJECT ",
        "\uf073": " SELECT ",
        "\uf0c7": " INTERSECT ",
        "\uf0d7": " JOIN ",
        "Π": " PROJECT ",
        "σ": " SELECT ",
        "∩": " INTERSECT ",
        "÷": " DIVIDE ",
        "–": "-",
        "—": "-"
    }
    for k, v in glyph_map.items():
        text = text.replace(k, v)

    # ===== 3. Sửa lỗi dính chữ do OCR/PDF =====
    # vd: "Phú.thứ" -> "Phú. thứ"
    text = re.sub(r"([a-zA-Z0-9])\.([a-zA-Z])", r"\1. \2", text)

    # Chuẩn hóa xuống dòng
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines: List[str] = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    # ===== 4. Healing logic (có kiểm soát) =====
    paragraphs: List[str] = []
    buffer = lines[0]

    for curr in lines[1:]:
        # ---- Nhận diện ngữ cảnh ----
        is_new_section = bool(RE_NEW_SECTION.match(curr))
        is_table = (
            RE_TABLE_MARKDOWN.match(curr) or
            ("|" in curr and "|" in buffer)
        )

        schema_open = bool(RE_SCHEMA_LINE.match(buffer)) and not RE_SCHEMA_CLOSE.match(buffer)
        list_unfinished = buffer.endswith(",")
        operator_unfinished = bool(RE_OPERATOR.search(buffer)) and "(" not in buffer
        prev_sentence_end = bool(RE_SENTENCE_END.search(buffer))

        is_param_line = bool(re.match(r"^[A-Z0-9, \s=]+$", curr)) and len(curr) < 30

        if is_new_section or is_table:
            paragraphs.append(buffer)
            buffer = curr
        elif schema_open or list_unfinished or operator_unfinished or is_param_line:
            buffer += " " + curr

        elif not prev_sentence_end and curr[0].islower():
            # Văn bản bị ngắt dòng giữa câu
            buffer += " " + curr

        else:
            paragraphs.append(buffer)
            buffer = curr

    paragraphs.append(buffer)

    # ===== 5. Hậu xử lý =====
    output: List[str] = []
    for p in paragraphs:
        # Giữ nguyên bảng / schema
        if RE_TABLE_MARKDOWN.match(p) or "|" in p:
            output.append(p)
        else:
            p = re.sub(r"[ \t]+", " ", p).strip()
            if p:
                output.append(p)

    return "\n\n".join(output)
