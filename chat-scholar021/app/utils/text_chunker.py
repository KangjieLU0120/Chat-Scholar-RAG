import re
from typing import Any


NUMBERED_HEADING_RE = re.compile(r"^((?:[1-9]\d*)(?:\.(?:[1-9]\d*)){0,2})\.?\s+.+$")
PLAIN_HEADING_RE = re.compile(r"^(abstract|introduction|background|related work|method|methods|materials and methods|experiments|results|discussion|conclusion|conclusions|references)\b", re.I)
REFERENCE_RE = re.compile(r"^(\[?\d+\]?\s+.+|[A-Z][A-Za-z\-']+,\s+[A-Z].+\(\d{4}\).+)$")
TABLE_CAPTION_RE = re.compile(r"^(table|figure)\s+\d+", re.I)
BODYISH_WORD_RE = re.compile(r"\b(the|this|these|those|because|while|which|than|using|required|performed|shows|showed|were|was|are|is|can|could|should)\b", re.I)
MATH_SYMBOL_RE = re.compile(r"[=+\-*/^<>≤≥≈∑∫√∞±×÷µλβγδεθαπσΩω\[\](){}]")
REPLACEMENT_CHAR_RE = re.compile(r"[�□]")
EQUATION_CUE_RE = re.compile(r"\b(eq\.?|equation|formula|loss|objective|residual|defined as|where)\b", re.I)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()



def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))



def _math_symbol_score(text: str) -> int:
    return len(MATH_SYMBOL_RE.findall(text)) + len(REPLACEMENT_CHAR_RE.findall(text))



def _is_likely_equation(text: str) -> bool:
    clean = _normalize_text(text)
    if not clean:
        return False
    if TABLE_CAPTION_RE.match(clean) or clean.lower() == "references":
        return False
    lines = [line.strip() for line in clean.split("\n") if line.strip()]
    if not lines or len(lines) > 3:
        return False

    dense_math_lines = 0
    for line in lines:
        word_count = _count_words(line)
        math_score = _math_symbol_score(line)
        operator_count = len(re.findall(r"[=+\-*/^<>≤≥≈]", line))
        replacement_count = len(REPLACEMENT_CHAR_RE.findall(line))

        if replacement_count >= 2 and operator_count >= 1:
            return True
        if operator_count >= 1 and math_score >= 2 and word_count <= 10 and len(line) <= 120 and not line.endswith("."):
            dense_math_lines += 1
            continue
        if math_score >= 3 and word_count <= 8 and len(line) <= 100:
            dense_math_lines += 1
            continue
        if EQUATION_CUE_RE.search(line) and operator_count >= 1 and word_count <= 12:
            dense_math_lines += 1

    return dense_math_lines >= 1



def _is_likely_section_title(text: str) -> bool:
    clean = _normalize_text(text)
    if not clean:
        return False
    if "\n" in clean:
        return False
    if TABLE_CAPTION_RE.match(clean):
        return False
    if _is_likely_equation(clean):
        return False
    if clean.lower() == "references":
        return True
    if clean.endswith((".", ":", ";", "?", "!")):
        return False
    if "," in clean:
        return False

    word_count = _count_words(clean)
    if word_count == 0 or word_count > 14 or len(clean) > 90:
        return False

    match = NUMBERED_HEADING_RE.match(clean)
    if match:
        title_text = clean[match.end(1):].strip(" .")
        if not title_text:
            return False
        if _count_words(title_text) > 10 or len(title_text) > 70:
            return False
        if BODYISH_WORD_RE.search(title_text):
            return False
        return True

    if PLAIN_HEADING_RE.match(clean):
        return not BODYISH_WORD_RE.search(clean)

    return False



def _heading_level(title: str):
    clean = _normalize_text(title)
    if not clean:
        return None
    match = NUMBERED_HEADING_RE.match(clean)
    if match:
        return len(match.group(1).split("."))
    if PLAIN_HEADING_RE.match(clean):
        return 1
    return None



def _split_into_sentences(text: str):
    text = _normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]



def _join_sentences(sentences):
    return " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip()).strip()



def _split_long_text_by_sentences(text: str, chunk_size: int, sentence_overlap: int = 1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current and current_len + sent_len + 1 > chunk_size:
            chunks.append(_join_sentences(current))
            overlap_sents = current[-sentence_overlap:] if sentence_overlap > 0 else []
            current = overlap_sents[:]
            current_len = len(_join_sentences(current))

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(_join_sentences(current))

    return [chunk for chunk in chunks if chunk]



def _chunk_single_text(text: str, chunk_size: int, overlap: int):
    text = _normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    final_chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())
                current_chunk = ""
            final_chunks.extend(_split_long_text_by_sentences(para, chunk_size, sentence_overlap=1))
            continue

        candidate = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    return final_chunks



def _infer_chunk_type(text: str, base_type: str = "paragraph") -> str:
    clean = _normalize_text(text)
    if not clean:
        return base_type
    if base_type in {"table_like", "caption", "reference", "title", "equation_like"}:
        return base_type
    if clean.lower() == "references" or REFERENCE_RE.match(clean):
        return "reference"
    if _is_likely_equation(clean):
        return "equation_like"
    if _is_likely_section_title(clean):
        return "title"
    return base_type or "paragraph"



def _split_block_into_units(block: dict[str, Any]) -> list[dict[str, Any]]:
    text = _normalize_text(block.get("text", ""))
    if not text:
        return []

    page = block.get("page")
    block_type = block.get("block_type", "paragraph")

    if block_type in {"table_like", "caption", "reference", "equation_like"}:
        return [{
            "text": text,
            "unit_type": block_type,
            "page": page,
            "contains_math": block_type == "equation_like",
        }]

    if block_type == "title" and _is_likely_section_title(text):
        return [{
            "text": text,
            "unit_type": "title",
            "page": page,
            "contains_math": False,
        }]

    units = []
    current_lines = []

    def flush_paragraph():
        nonlocal current_lines
        paragraph = _normalize_text("\n".join(current_lines))
        if paragraph:
            units.append({
                "text": paragraph,
                "unit_type": "paragraph",
                "page": page,
                "contains_math": _is_likely_equation(paragraph),
            })
        current_lines = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _is_likely_section_title(line):
            flush_paragraph()
            units.append({
                "text": line,
                "unit_type": "title",
                "page": page,
                "contains_math": False,
            })
        elif _is_likely_equation(line):
            flush_paragraph()
            units.append({
                "text": line,
                "unit_type": "equation_like",
                "page": page,
                "contains_math": True,
            })
        else:
            current_lines.append(line)

    flush_paragraph()

    if not units:
        units.append({
            "text": text,
            "unit_type": "paragraph",
            "page": page,
            "contains_math": _is_likely_equation(text),
        })

    return units



def _update_title_path(title_path: list[str], title_text: str) -> list[str]:
    level = _heading_level(title_text)
    clean_title = _normalize_text(title_text)
    if not clean_title:
        return title_path[:]
    if level is None:
        return [clean_title]

    new_path = title_path[: max(0, level - 1)]
    while len(new_path) < level - 1:
        new_path.append("")
    new_path.append(clean_title)
    return [item for item in new_path if item]



def _build_chunks_from_pages(pages: list[dict[str, Any]], chunk_size: int = 500, sentence_overlap: int = 1):
    all_chunks = []
    chunk_id = 0
    current_title_path: list[str] = []

    for page_item in pages:
        page_num = page_item.get("page")
        blocks = page_item.get("blocks", [])

        units = []
        if blocks:
            for block in blocks:
                prepared = dict(block)
                prepared["page"] = page_num
                units.extend(_split_block_into_units(prepared))
        else:
            units.append({
                "text": page_item.get("text", ""),
                "unit_type": "paragraph",
                "page": page_num,
                "contains_math": False,
            })

        for unit in units:
            text = _normalize_text(unit.get("text", ""))
            if not text:
                continue

            unit_type = unit.get("unit_type", "paragraph")
            contains_math = bool(unit.get("contains_math", unit_type == "equation_like" or _is_likely_equation(text)))

            if unit_type == "title" and _is_likely_section_title(text):
                current_title_path = _update_title_path(current_title_path, text)
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "page": page_num,
                    "page_start": page_num,
                    "page_end": page_num,
                    "section": current_title_path[-1] if current_title_path else text,
                    "chunk_type": "title",
                    "title_path": current_title_path[:],
                    "contains_math": False,
                })
                chunk_id += 1
                continue

            if len(text) > chunk_size and unit_type == "paragraph":
                sub_chunks = _split_long_text_by_sentences(text, chunk_size=chunk_size, sentence_overlap=sentence_overlap)
            else:
                sub_chunks = [text]

            for sub_text in sub_chunks:
                clean_sub_text = _normalize_text(sub_text)
                if not clean_sub_text:
                    continue
                chunk_type = _infer_chunk_type(clean_sub_text, unit_type)
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": clean_sub_text,
                    "page": page_num,
                    "page_start": page_num,
                    "page_end": page_num,
                    "section": current_title_path[-1] if current_title_path else None,
                    "chunk_type": chunk_type,
                    "title_path": current_title_path[:],
                    "contains_math": contains_math or chunk_type == "equation_like",
                })
                chunk_id += 1

    return all_chunks



def split_text_into_chunks(data, chunk_size=500, overlap=100, return_metadata=False):
    all_chunks = []
    chunk_id = 0

    if isinstance(data, str):
        chunks = _chunk_single_text(data, chunk_size, overlap)
        if return_metadata:
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "page": None,
                    "page_start": None,
                    "page_end": None,
                    "chunk_id": chunk_id,
                    "section": None,
                    "chunk_type": "equation_like" if _is_likely_equation(chunk) else "paragraph",
                    "title_path": [],
                    "contains_math": _is_likely_equation(chunk),
                })
                chunk_id += 1
            return all_chunks
        return chunks

    if isinstance(data, list):
        chunks = _build_chunks_from_pages(data, chunk_size=chunk_size, sentence_overlap=1)
        if return_metadata:
            return chunks
        return [c["text"] for c in chunks]

    return []
