import re

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from PyPDF2 import PdfReader


NUMBERED_HEADING_RE = re.compile(r"^((?:[1-9]\d*)(?:\.(?:[1-9]\d*)){0,2})\.?\s+.+$")
PLAIN_HEADING_RE = re.compile(r"^(abstract|introduction|background|related work|method|methods|materials and methods|experiments|results|discussion|conclusion|conclusions|references)\b", re.I)
TABLE_CAPTION_RE = re.compile(r"^(table|figure)\s+\d+", re.I)
REFERENCE_RE = re.compile(r"^(\[?\d+\]?\s+.+|[A-Z][A-Za-z\-']+,\s+[A-Z].+\(\d{4}\).+)$")
BODYISH_WORD_RE = re.compile(r"\b(the|this|these|those|because|while|which|than|using|required|performed|shows|showed|were|was|are|is|can|could|should)\b", re.I)

MATH_SYMBOL_RE = re.compile(r"[=+\-*/^<>≤≥≈∑∫√∞±×÷µλβγδεθαπσΩω\[\](){}]")
ASCII_MATH_LINE_RE = re.compile(r"^[A-Za-z0-9_\s=+\-*/^.,()\[\]{}<>|:]+$")
REPLACEMENT_CHAR_RE = re.compile(r"[�□]")
EQUATION_CUE_RE = re.compile(r"\b(eq\.?|equation|formula|loss|objective|residual|defined as|where)\b", re.I)


def _clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line is not None).strip()



def _clean_equation_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*([=+\-*/^<>≤≥≈])\s*", r" \1 ", text)
    text = re.sub(r"\s+([,)\]}>])", r"\1", text)
    text = re.sub(r"([(\[{<])\s+", r"\1", text)
    text = re.sub(r"\n{2,}", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines).strip()



def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))



def _looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False

    digit_heavy_lines = sum(1 for line in lines if len(re.findall(r"\d", line)) >= 3)
    short_lines = sum(1 for line in lines if len(line) <= 60)
    multi_space_lines = sum(1 for line in lines if re.search(r"\S\s{2,}\S", line))

    return (
        digit_heavy_lines >= 2
        and (short_lines >= max(2, len(lines) // 2) or multi_space_lines >= 1)
    )



def _looks_like_reference(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if clean.lower() == "references":
        return False
    return bool(REFERENCE_RE.match(clean))



def _math_symbol_score(text: str) -> int:
    return len(MATH_SYMBOL_RE.findall(text)) + len(REPLACEMENT_CHAR_RE.findall(text))



def _looks_like_equation(text: str, font_size: float = 0.0) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if TABLE_CAPTION_RE.match(clean) or _looks_like_reference(clean):
        return False
    if _looks_like_heading(clean, font_size=font_size):
        return False

    lines = [line.strip() for line in clean.split("\n") if line.strip()]
    if not lines:
        return False

    dense_math_lines = 0
    cue_lines = 0

    for line in lines:
        word_count = _count_words(line)
        math_score = _math_symbol_score(line)
        alpha_count = len(re.findall(r"[A-Za-z]", line))
        digit_count = len(re.findall(r"\d", line))
        operator_count = len(re.findall(r"[=+\-*/^<>≤≥≈]", line))
        replacement_count = len(REPLACEMENT_CHAR_RE.findall(line))
        compact_ascii_math = bool(ASCII_MATH_LINE_RE.match(line)) and operator_count >= 1 and word_count <= 8

        if replacement_count >= 2 and operator_count >= 1:
            return True

        if EQUATION_CUE_RE.search(line):
            cue_lines += 1

        if (
            operator_count >= 1
            and (math_score >= 2 or compact_ascii_math)
            and word_count <= 10
            and len(line) <= 120
            and not line.endswith(".")
        ):
            dense_math_lines += 1
            continue

        if (
            math_score >= 3
            and word_count <= 8
            and alpha_count <= max(8, word_count * 3)
            and len(line) <= 100
        ):
            dense_math_lines += 1
            continue

        if (
            digit_count >= 2
            and operator_count >= 1
            and word_count <= 10
            and len(line) <= 80
        ):
            dense_math_lines += 1

    if dense_math_lines >= 1 and len(lines) <= 3:
        return True
    if dense_math_lines >= 1 and cue_lines >= 1:
        return True
    return False



def _looks_like_heading(text: str, font_size: float = 0.0) -> bool:
    clean = text.strip()
    if not clean:
        return False

    lines = [line.strip() for line in clean.split("\n") if line.strip()]
    line_count = len(lines)
    single_line = lines[0] if lines else clean
    word_count = _count_words(single_line)

    if line_count > 1:
        return False
    if word_count == 0 or word_count > 14 or len(single_line) > 90:
        return False
    if single_line.endswith((".", ":", ";", "?", "!")):
        return False
    if "," in single_line:
        return False
    if BODYISH_WORD_RE.search(single_line) and not PLAIN_HEADING_RE.match(single_line):
        return False

    match = NUMBERED_HEADING_RE.match(single_line)
    if match:
        title_text = single_line[match.end(1):].strip(" .")
        if not title_text:
            return False
        if _count_words(title_text) > 10 or len(title_text) > 70:
            return False
        if BODYISH_WORD_RE.search(title_text):
            return False
        return True

    if PLAIN_HEADING_RE.match(single_line):
        return True

    if font_size >= 13.5 and word_count <= 8 and single_line == single_line.title():
        return True

    return False



def _classify_block(text: str, font_size: float = 0.0) -> str:
    clean = text.strip()
    if not clean:
        return "paragraph"

    if TABLE_CAPTION_RE.match(clean):
        return "caption"
    if _looks_like_table(clean):
        return "table_like"
    if clean.lower() == "references" or _looks_like_reference(clean):
        return "reference"
    if _looks_like_equation(clean, font_size=font_size):
        return "equation_like"
    if _looks_like_heading(clean, font_size=font_size):
        return "title"
    return "paragraph"



def _extract_with_pymupdf(file_path: str):
    pages = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        raw_blocks = page_dict.get("blocks", [])
        parsed_blocks = []

        for raw_block in raw_blocks:
            if "lines" not in raw_block:
                continue

            lines_text = []
            font_sizes = []
            raw_line_texts = []

            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                span_texts = []
                for span in spans:
                    span_text = span.get("text", "")
                    if span_text:
                        span_texts.append(span_text)
                    span_size = span.get("size")
                    if isinstance(span_size, (int, float)):
                        font_sizes.append(float(span_size))

                joined_line = " ".join(part.strip() for part in span_texts if part.strip()).strip()
                tight_line = "".join(part for part in span_texts if part).strip()
                candidate_line = joined_line
                if _looks_like_equation(tight_line) or REPLACEMENT_CHAR_RE.search(tight_line):
                    candidate_line = tight_line or joined_line
                if candidate_line:
                    raw_line_texts.append(candidate_line)

            provisional_text = "\n".join(raw_line_texts)
            font_size = round(max(font_sizes), 2) if font_sizes else 0.0
            provisional_type = _classify_block(provisional_text, font_size=font_size)

            for line_text in raw_line_texts:
                if provisional_type == "equation_like" or _looks_like_equation(line_text, font_size=font_size):
                    cleaned_line = _clean_equation_text(line_text)
                else:
                    cleaned_line = _clean_text(line_text)
                if cleaned_line:
                    lines_text.append(cleaned_line)

            block_text = (
                _clean_equation_text("\n".join(lines_text))
                if provisional_type == "equation_like"
                else _clean_text("\n".join(lines_text))
            )
            if not block_text:
                continue

            bbox = raw_block.get("bbox", [0, 0, 0, 0])
            x0, y0, x1, y1 = bbox
            block_type = _classify_block(block_text, font_size=font_size)

            parsed_blocks.append({
                "x0": x0,
                "y0": y0,
                "bbox": [x0, y0, x1, y1],
                "font_size": font_size,
                "block_type": block_type,
                "text": block_text,
                "contains_math": block_type == "equation_like" or bool(_looks_like_equation(block_text, font_size=font_size)),
            })

        parsed_blocks.sort(key=lambda b: (round(b["y0"], 1), round(b["x0"], 1)))
        page_text = _clean_text("\n\n".join(block["text"] for block in parsed_blocks))

        if parsed_blocks or page_text:
            pages.append({
                "page": page_num,
                "blocks": parsed_blocks,
                "text": page_text,
            })

    doc.close()
    return pages



def _extract_with_pypdf2(file_path: str):
    pages = []
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = _clean_text(page.extract_text() or "")
        if not page_text:
            continue
        block_type = "equation_like" if _looks_like_equation(page_text) else "paragraph"
        pages.append({
            "page": page_num,
            "blocks": [{
                "x0": 0,
                "y0": 0,
                "bbox": [0, 0, 0, 0],
                "font_size": 0.0,
                "block_type": block_type,
                "text": page_text,
                "contains_math": block_type == "equation_like",
            }],
            "text": page_text,
        })

    return pages



def extract_pages_from_pdf(file_path: str):
    try:
        if fitz is not None:
            return _extract_with_pymupdf(file_path)
        return _extract_with_pypdf2(file_path)
    except Exception as e:
        print("PDF Read Error:", e)
        return []



def extract_text_from_pdf(file_path: str):
    pages = extract_pages_from_pdf(file_path)
    return "\n\n".join(page.get("text", "") for page in pages if page.get("text"))
