from pathlib import Path

import pymupdf

PDFS = [
    (
        Path("data/raw/compliance/gli/GLI-19-v3.0.pdf"),
        Path("data/raw/compliance/gli/GLI-19-v3.0.md"),
        "GLI-19 v3.0 Interactive Gaming Systems",
    ),
    (
        Path("data/raw/compliance/nj_dge/NJAC_13_69O_Internet_Gaming.pdf"),
        Path("data/raw/compliance/nj_dge/NJAC_13_69O_Internet_Gaming.md"),
        "New Jersey Administrative Code 13:69O Internet and Mobile Gaming",
    ),
    (
        Path("data/raw/compliance/nj_dge/NJAC_13_69D_Internal_Controls.pdf"),
        Path("data/raw/compliance/nj_dge/NJAC_13_69D_Internal_Controls.md"),
        "New Jersey Administrative Code 13:69D Accounting and Internal Controls",
    ),
    (
        Path("data/raw/compliance/nj_dge/NJAC_13_69E_Gaming_Equipment.pdf"),
        Path("data/raw/compliance/nj_dge/NJAC_13_69E_Gaming_Equipment.md"),
        "New Jersey Administrative Code 13:69E Gaming Equipment",
    ),
]


def extract_pdf_to_md(pdf_path: Path, md_path: Path, title: str):
    print(f"Extracting {pdf_path.name} -> {md_path.name}...")
    lines = [f"# {title}\n"]

    with pymupdf.open(pdf_path) as document:
        page_count = len(document)
        for page_idx, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            lines.append(f"\n<!-- Page {page_idx} -->\n")
            lines.append(text)

    content = "\n".join(lines)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(content, encoding="utf-8")
    chars = len(content)
    kb = chars / 1024
    print(f"  Saved {md_path.name} ({page_count} pages, {chars:,} chars, {kb:.1f} KB)")


def main():
    print("=== Extracting Compliance PDFs to Structured Markdown ===")
    for pdf_path, md_path, title in PDFS:
        extract_pdf_to_md(pdf_path, md_path, title)
    print("Extraction complete.")


if __name__ == "__main__":
    main()
