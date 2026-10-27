import json
import re
import statistics
from pathlib import Path

COMPLIANCE_RAW_DIR = Path("data/raw/compliance")
COMPLIANCE_OUT_FILE = Path("data/processed/compliance_chunks.jsonl")
COMPLIANCE_MAX_CHARS = 1000

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
ANCHOR = re.compile(r"\s*\{\s*#.*?\}\s*$")
PAGE_MARKER = re.compile(r"^\s*<!--\s*Page\s+(\d+)\s*-->\s*$")
NJ_SECTION = re.compile(r"^\s*(13:69[ODE]-\d+\.\d+[A-Z]?)\s*$", re.IGNORECASE)
GLI_CHAPTER = re.compile(r"^\s*Chapter\s+(\d+)\s*:\s*(.*?)\s*$", re.IGNORECASE)
GLI_SECTION = re.compile(r"^\s*(\d+(?:\.\d+)+)(?:\s+(.*?))?\s*$")
CLAUSE_MARKER = re.compile(r"^\s*(\([A-Za-z0-9]+\)|\d{1,2}\.|[ivxlcdm]{1,6}\.)\s*$", re.IGNORECASE)
NJ_SUBCHAPTER = re.compile(r"^\s*SUBCHAPTER\s+\d+\.\s+.*$", re.IGNORECASE)
PAGE_FOOTER = re.compile(r"^\s*-\s*\d+\s*-\s*$")
IDENTIFIER = re.compile(
    r"^\s*(?:(?:13:69[ODE]-)?\d+(?:\.\d+)+(?:[A-Z])?|RTS\s*\d+[A-Z]?|GLI-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def _page_from_line(line, current_page):
    match = PAGE_MARKER.match(line)
    if match:
        return int(match.group(1))
    return current_page


def _nonempty_after(lines, index):
    for next_index in range(index + 1, len(lines)):
        if lines[next_index].strip():
            return next_index, lines[next_index].strip()
    return None, ""


def _paragraph_blocks(lines, page_by_line, start, end):
    blocks = []
    current_lines = []
    start_page = None
    end_page = None
    clause_stack = []

    def flush():
        nonlocal current_lines, start_page, end_page
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append({
                "text": text,
                "page_start": start_page,
                "page_end": end_page,
                "clause_path": list(clause_stack),
            })
        current_lines = []
        start_page = None
        end_page = None

    for line_index in range(start, end):
        line = lines[line_index]
        page_match = PAGE_MARKER.match(line)
        if page_match or PAGE_FOOTER.match(line):
            continue

        marker = CLAUSE_MARKER.match(line)
        if marker and current_lines:
            flush()
        if marker:
            label = marker.group(1)
            if label.startswith("("):
                level = 1
            elif label.endswith(".") and label[:-1].isdigit():
                level = 2
            else:
                level = 3
            clause_stack = clause_stack[:level - 1] + [label]

        if not line.strip():
            if current_lines:
                flush()
            continue
        if start_page is None:
            start_page = page_by_line[line_index]
        end_page = page_by_line[line_index]
        current_lines.append(line.rstrip())

    flush()
    return blocks


def _pack_compliance_blocks(blocks, max_chars):
    packed = []
    current = []
    current_size = 0

    def flush():
        nonlocal current, current_size
        if current:
            packed.append(current)
        current = []
        current_size = 0

    for block in blocks:
        block_size = len(block["text"])
        if current and current_size + block_size + 2 > max_chars:
            flush()
        current.append(block)
        current_size += block_size + (2 if len(current) > 1 else 0)
        if block_size > max_chars:
            flush()
    flush()
    return packed


def _split_long_block(block, max_chars):
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero.")
    if len(block["text"]) <= max_chars:
        return [block]

    sentences = re.split(r"(?<=[.!?;])\s+", block["text"].strip())
    continuation_prefix = (
        f"Continuation of clause {''.join(block['clause_path'])}: "
        if block["clause_path"]
        else "Continuation: "
    )
    piece_limit = max_chars - len(continuation_prefix)
    if piece_limit < 1:
        raise ValueError("max_chars is too small for the continuation label.")

    pieces = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        word_pieces = []
        current_words = ""
        for word in words:
            if len(word) > piece_limit:
                if current_words:
                    word_pieces.append(current_words)
                    current_words = ""
                word_pieces.extend(
                    word[offset:offset + piece_limit]
                    for offset in range(0, len(word), piece_limit)
                )
            elif current_words and len(current_words) + len(word) + 1 > piece_limit:
                word_pieces.append(current_words)
                current_words = word
            else:
                current_words = f"{current_words} {word}".strip()
        if current_words:
            word_pieces.append(current_words)

        for word_piece in word_pieces:
            if current and len(current) + len(word_piece) + 1 > piece_limit:
                pieces.append(current)
                current = word_piece
            else:
                current = f"{current} {word_piece}".strip()
    if current:
        pieces.append(current)

    split_blocks = []
    for index, piece in enumerate(pieces):
        if index and block["clause_path"]:
            prefix = f"Continuation of clause {''.join(block['clause_path'])}: "
        elif index:
            prefix = "Continuation: "
        else:
            prefix = ""
        split_blocks.append({
            **block,
            "text": prefix + piece,
            "continuation": index > 0,
        })
    return split_blocks


def _format_clause_path(clause_path):
    return "".join(f"({label.strip('() .')})" for label in clause_path)


def _format_clause_paths(clause_paths):
    paths = [tuple(label.strip("() .").lower() for label in path) for path in clause_paths]
    leaf_paths = [
        path
        for path in paths
        if not any(
            len(other) > len(path) and other[:len(path)] == path
            for other in paths
        )
    ]
    return ", ".join(
        _format_clause_path(path) for path in dict.fromkeys(leaf_paths)
    )


def _build_compliance_chunks(path, relative_path, jurisdiction, regulation_type, source_url):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    page_by_line = []
    current_page = None
    for line in lines:
        current_page = _page_from_line(line, current_page)
        page_by_line.append(current_page)

    if jurisdiction == "nj_dge":
        sections = _parse_nj_sections(lines, page_by_line)
    elif jurisdiction == "gli":
        sections = _parse_gli_sections(lines, page_by_line)
    else:
        sections = _parse_markdown_sections(lines, page_by_line)

    chunks = []
    source_name = path.stem
    for section_index, section in enumerate(sections):
        section_blocks = []
        for block in section["blocks"]:
            section_blocks.extend(_split_long_block(block, COMPLIANCE_MAX_CHARS))
        packed_blocks = _pack_compliance_blocks(
            section_blocks, COMPLIANCE_MAX_CHARS
        )
        for index, blocks in enumerate(packed_blocks):
            body = "\n\n".join(block["text"] for block in blocks)
            clause_paths = []
            for block in blocks:
                if block["clause_path"] and block["clause_path"] not in clause_paths:
                    clause_paths.append(block["clause_path"])
            page_starts = [block["page_start"] for block in blocks if block["page_start"] is not None]
            page_ends = [block["page_end"] for block in blocks if block["page_end"] is not None]
            section_label = section["section_id"] or "section"
            chunk_id = f"{relative_path}#{section_index}-{section_label}-{index}"
            heading_path = section["heading_path"]
            citation = section["section_id"] or " > ".join(heading_path)
            if jurisdiction == "gli" and section["section_id"]:
                citation = f"GLI-19 {citation}"
            if clause_paths:
                citation += f" {_format_clause_paths(clause_paths)}"
            if len(body) > COMPLIANCE_MAX_CHARS:
                raise ValueError(
                    f"Chunk {chunk_id} exceeds {COMPLIANCE_MAX_CHARS} characters."
                )

            chunks.append({
                "chunk_id": chunk_id,
                "doc": "compliance",
                "file": relative_path,
                "source_doc": source_name,
                "source_url": source_url,
                "heading_path": heading_path,
                "section_id": section["section_id"],
                "clause_path": clause_paths[-1] if clause_paths else [],
                "clause_paths": clause_paths,
                "citation": citation,
                "jurisdiction": jurisdiction,
                "regulation_type": regulation_type,
                "page_start": min(page_starts) if page_starts else None,
                "page_end": max(page_ends) if page_ends else None,
                "text": body,
                "char_count": len(body),
                "oversize": len(body) > COMPLIANCE_MAX_CHARS,
                "continuation": any(block.get("continuation", False) for block in blocks),
            })
    return chunks


def _parse_nj_sections(lines, page_by_line):
    sections = []
    section_starts = []
    for index, line in enumerate(lines):
        if NJ_SECTION.match(line):
            section_starts.append(index)

    context = ["New Jersey Administrative Code"]
    section_starts.append(len(lines))
    previous_start = 0
    for section_index in range(len(section_starts) - 1):
        start = section_starts[section_index]
        end = section_starts[section_index + 1]
        section_id = NJ_SECTION.match(lines[start]).group(1).upper()
        for line in lines[previous_start:start]:
            if line.strip().upper().startswith("CHAPTER "):
                context = [line.strip()]
            elif NJ_SUBCHAPTER.match(line):
                context = context[:1] + [line.strip()]

        title_index, title = _nonempty_after(lines, start)
        if title_index is not None and not IDENTIFIER.match(title):
            heading = f"{section_id} {title}"
            body_start = title_index + 1
        else:
            heading = section_id
            body_start = start + 1
        body_blocks = _paragraph_blocks(lines, page_by_line, body_start, end)
        sections.append({
            "heading_path": context + [heading],
            "section_id": section_id,
            "blocks": body_blocks,
        })
        previous_start = end
    return sections


def _parse_gli_sections(lines, page_by_line):
    sections = []
    starts = []
    body_start_page = 5
    for line_index, line in enumerate(lines):
        if page_by_line[line_index] is not None and page_by_line[line_index] < body_start_page:
            continue
        chapter_match = GLI_CHAPTER.match(line)
        section_match = GLI_SECTION.match(line)
        if chapter_match:
            starts.append((line_index, "chapter", chapter_match.group(1), chapter_match.group(2).strip()))
        elif section_match:
            title = (section_match.group(2) or "").strip()
            if not title:
                title_index, title = _nonempty_after(lines, line_index)
            else:
                title_index = line_index
            if title_index is not None and title_index < len(lines):
                starts.append((line_index, "section", section_match.group(1), title))
    starts.append((len(lines), "end", "", ""))

    chapter_titles = {}
    section_titles = {}
    for entry_index in range(len(starts) - 1):
        start, kind, section_id, title = starts[entry_index]
        end = starts[entry_index + 1][0]
        if kind == "chapter":
            chapter_titles[section_id] = f"Chapter {section_id}: {title}"
            section_titles = {
                key: value for key, value in section_titles.items()
                if key.split(".")[0] != section_id
            }
            body_start = start + 1
            heading_path = [chapter_titles[section_id]]
        elif kind == "section":
            title_index = start if title else _nonempty_after(lines, start)[0]
            body_start = title_index + 1 if title_index is not None else start + 1
            chapter_number = section_id.split(".")[0]
            heading_path = [chapter_titles.get(chapter_number, f"Chapter {chapter_number}")]
            segments = section_id.split(".")
            for prefix_length in range(1, len(segments)):
                prefix = ".".join(segments[:prefix_length])
                if prefix in section_titles:
                    heading_path.append(f"{prefix} {section_titles[prefix]}")
            heading_path.append(f"{section_id} {title}")
            section_titles[section_id] = title
        else:
            continue
        blocks = _paragraph_blocks(lines, page_by_line, body_start, end)
        sections.append({
            "heading_path": heading_path,
            "section_id": section_id,
            "blocks": blocks,
        })
    return sections


def _parse_markdown_sections(lines, page_by_line):
    sections = []
    stack = []
    current_start = 0
    current_path = []

    def flush(end):
        if not current_path:
            return
        blocks = _paragraph_blocks(lines, page_by_line, current_start, end)
        if blocks:
            section_id = None
            for item in reversed(current_path):
                match = re.search(
                    r"\bRTS\s+(?:(?:requirement|aim)\s+)?(\d+[A-Z]?)\b|"
                    r"\b13:69[ODE]-\d+\.\d+[A-Z]?\b",
                    item,
                    re.IGNORECASE,
                )
                if match:
                    section_id = (
                        f"RTS {match.group(1)}"
                        if match.group(1)
                        else match.group(0).strip()
                    )
                    break
            sections.append({
                "heading_path": list(current_path),
                "section_id": section_id,
                "blocks": blocks,
            })

    for line_index, line in enumerate(lines):
        if PAGE_MARKER.match(line):
            continue
        heading_match = HEADING.match(line)
        if not heading_match:
            continue
        if stack:
            flush(line_index)
        level = len(heading_match.group(1))
        title = ANCHOR.sub("", heading_match.group(2)).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current_path = [heading for _, heading in stack]
        current_start = line_index + 1
    flush(len(lines))
    return sections


def ingest_compliance():
    COMPLIANCE_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    source_urls = {
        "gli": (
            "https://gaminglabs.com/gli-standards/",
            "https://raw.githubusercontent.com/alexbotov/rgs/main/docs/"
            "GLI-19-Interactive-Gaming-Systems-v3.0.pdf",
        ),
        "nj_dge": (
            "https://www.nj.gov/oag/ge/regulations.html",
        ),
        "ukgc": (
            "https://www.gamblingcommission.gov.uk/standards/"
            "remote-gambling-and-software-technical-standards/",
        ),
    }

    source_paths = sorted(COMPLIANCE_RAW_DIR.rglob("*.md"))
    if not source_paths:
        raise FileNotFoundError(
            f"No Markdown sources found under {COMPLIANCE_RAW_DIR}"
        )
    for path in source_paths:
        if path.name.startswith("_"):
            continue
        jurisdiction = path.relative_to(COMPLIANCE_RAW_DIR).parts[0]
        if jurisdiction not in source_urls:
            raise ValueError(f"Unknown compliance jurisdiction: {jurisdiction}")
        regulation_type = "administrative_code" if jurisdiction == "nj_dge" else "technical_standard"
        relative_path = path.relative_to(COMPLIANCE_RAW_DIR).as_posix()
        if jurisdiction == "ukgc":
            source_url = source_urls[jurisdiction][0] + path.stem
        elif jurisdiction == "gli":
            source_url = source_urls[jurisdiction][1]
        elif jurisdiction == "nj_dge":
            source_url = source_urls[jurisdiction][0]
        file_chunks = _build_compliance_chunks(
            path, relative_path, jurisdiction, regulation_type, source_url
        )
        if not file_chunks:
            raise ValueError(f"No chunks were created from {path}")
        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(f"No compliance chunks were created from {COMPLIANCE_RAW_DIR}")

    with COMPLIANCE_OUT_FILE.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = [chunk["char_count"] for chunk in chunks]
    print(
        f"{len(chunks)} compliance chunks | "
        f"min {min(sizes)} | median {statistics.median(sizes)} | max {max(sizes)}"
    )
    print("By jurisdiction:")
    for jurisdiction in sorted({chunk["jurisdiction"] for chunk in chunks}):
        subset = [chunk for chunk in chunks if chunk["jurisdiction"] == jurisdiction]
        print(f"  {jurisdiction}: {len(subset)} chunks")
    return chunks


def main():
    ingest_compliance()


if __name__ == "__main__":
    main()