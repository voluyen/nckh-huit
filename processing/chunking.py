import re
from typing import List, Dict


class UltimateChunker:
    """
    Research-grade semantic chunker for RAG.
    """

    def __init__(
        self,
        max_chars: int = 900,
        overlap_chars: int = 180,
        hard_max_chars: int = 1400
    ):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.hard_max_chars = hard_max_chars

        # Nhận diện heading / section
        self.re_heading = re.compile(
            r"^(Câu\s+\d+|Chương\s+\d+|Bài\s+\d+|###|\d+[.)])",
            re.IGNORECASE
        )

        # Nhận diện bảng Markdown
        self.re_table_row = re.compile(r"^\|.*\|$")

        # Tách câu (fallback)
        self.re_sentence_split = re.compile(r"(?<=[.!?:;…])\s+")

    # ==================================================
    # Utilities
    # ==================================================
    def _is_table_paragraph(self, para: str) -> bool:
        lines = para.split("\n")
        return any(self.re_table_row.match(l.strip()) for l in lines)

    def _smart_overlap(self, text: str) -> str:
        """Overlap không cắt từ, ưu tiên ngữ nghĩa cuối đoạn"""
        if len(text) <= self.overlap_chars:
            return text

        tail = text[-self.overlap_chars:]
        first_space = tail.find(" ")
        return tail[first_space:].strip() if first_space != -1 else tail

    def _estimate_tokens(self, text: str) -> int:
        # Ước lượng tốt hơn len(text)//4 nhưng vẫn rất nhanh
        return max(1, int(len(text.split()) * 1.3))

    # ==================================================
    # Core logic
    # ==================================================
    def chunk_document(self, doc: Dict) -> List[Dict]:
        text = doc.get("text", "")
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[Dict] = []

        current_heading = "General Context"
        buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            # 1. Update heading context
            if self.re_heading.match(para):
                current_heading = para.split(":")[0].strip()

            # 2. Table handling (ABSOLUTE RULE)
            if self._is_table_paragraph(para):
                if buffer:
                    chunks.append(
                        self._create_payload(
                            buffer, doc, chunk_idx, current_heading
                        )
                    )
                    chunk_idx += 1
                    buffer = ""

                chunks.append(
                    self._create_payload(
                        para, doc, chunk_idx, current_heading
                    )
                )
                chunk_idx += 1
                continue

            # 3. Paragraph quá dài → sentence fallback
            if len(para) > self.hard_max_chars:
                sentences = self.re_sentence_split.split(para)
                temp = ""

                for sent in sentences:
                    if len(temp) + len(sent) <= self.max_chars:
                        temp = f"{temp} {sent}".strip()
                    else:
                        if temp:
                            chunks.append(
                                self._create_payload(
                                    temp, doc, chunk_idx, current_heading
                                )
                            )
                            chunk_idx += 1
                            temp = self._smart_overlap(temp) + " " + sent
                        else:
                            temp = sent

                if temp:
                    chunks.append(
                        self._create_payload(
                            temp, doc, chunk_idx, current_heading
                        )
                    )
                    chunk_idx += 1
                continue

            # 4. Paragraph bình thường → gộp
            if len(buffer) + len(para) <= self.max_chars:
                buffer = f"{buffer}\n\n{para}".strip()
            else:
                if buffer:
                    chunks.append(
                        self._create_payload(
                            buffer, doc, chunk_idx, current_heading
                        )
                    )
                    chunk_idx += 1
                    buffer = self._smart_overlap(buffer) + "\n\n" + para
                else:
                    buffer = para

        # Flush cuối
        if buffer:
            chunks.append(
                self._create_payload(
                    buffer, doc, chunk_idx, current_heading
                )
            )

        return chunks

    # ==================================================
    # Payload
    # ==================================================
    def _create_payload(
        self,
        text: str,
        doc: Dict,
        idx: int,
        heading: str
    ) -> Dict:
        """
        Inject heading trực tiếp vào text (RAG-critical)
        """
        injected_text = f"[{heading}]\n{text}"

        return {
            "text": injected_text,
            "metadata": {
                "source": doc.get("source"),
                "url": doc.get("url"),
                "page": doc.get("page"),
                "heading": heading,
                "chunk_id": idx,
                "token_estimate": self._estimate_tokens(injected_text),
                "confidence": doc.get("confidence"),
                "method": doc.get("method"),
            }
        }
