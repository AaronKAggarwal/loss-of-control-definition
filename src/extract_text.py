"""PDF text extraction using PyMuPDF (fitz)."""

import os
import sys

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF. Returns full text as a single string.

    Pages joined with double newlines to preserve document structure
    without losing page boundaries.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_all_papers(
    papers_config: list,
    papers_dir: str,
    output_dir: str,
    force: bool = False,
) -> dict:
    """Extract text from all papers in the registry and save to output_dir.

    Args:
        papers_config: List of paper dicts from papers.yaml (each has id, filename).
        papers_dir: Directory containing the source PDFs.
        output_dir: Directory to write {paper_id}.txt files.
        force: If True, re-extract even if output file already exists.

    Returns:
        Dict of {paper_id: {"chars": int, "pages": int, "skipped": bool, "warning": str|None}}.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats = {}

    for paper in papers_config:
        paper_id = paper["id"]
        pdf_path = os.path.join(papers_dir, paper["filename"])
        out_path = os.path.join(output_dir, f"{paper_id}.txt")

        # Idempotent: skip if output already exists unless force=True
        if os.path.exists(out_path) and not force:
            print(f"  [SKIP] {paper_id} — output already exists")
            stats[paper_id] = {"chars": 0, "pages": 0, "skipped": True, "warning": None}
            continue

        if not os.path.exists(pdf_path):
            print(f"  [ERROR] {paper_id} — PDF not found: {pdf_path}")
            stats[paper_id] = {"chars": 0, "pages": 0, "skipped": False, "warning": "PDF not found"}
            continue

        # Open once: get page count and extract text in the same pass
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)

        char_count = len(text)

        # Heuristic: fewer than 200 chars per page suggests a scanned PDF
        # with little or no OCR text. 200 is conservative — a typical
        # academic page has 2000-3000 chars.
        warning = None
        if page_count > 0 and (char_count / page_count) < 200:
            warning = f"Low text density ({char_count / page_count:.0f} chars/page) — may be a scanned PDF"
            print(f"  [WARN] {paper_id} — {warning}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  [OK] {paper_id} — {char_count:,} chars, {page_count} pages")
        stats[paper_id] = {"chars": char_count, "pages": page_count, "skipped": False, "warning": warning}

    return stats


if __name__ == "__main__":
    # CLI test: pass a PDF path as argument to verify extraction works.
    # Usage: uv run python -m src.extract_text /path/to/some.pdf
    #    or: uv run python src/extract_text.py /path/to/some.pdf
    if len(sys.argv) < 2:
        print("Usage: python src/extract_text.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    text = extract_text_from_pdf(pdf_path)
    char_count = len(text)

    print(f"Pages:      {page_count}")
    print(f"Characters: {char_count:,}")
    print(f"Chars/page: {char_count / page_count:.0f}" if page_count > 0 else "Chars/page: N/A")
    print(f"\n--- First 500 characters ---\n{text[:500]}")
