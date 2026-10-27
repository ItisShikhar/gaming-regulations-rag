from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
UKGC_ROOT = (
    "https://www.gamblingcommission.gov.uk/standards/"
    "remote-gambling-and-software-technical-standards"
)
UKGC_SECTION_THREE = f"{UKGC_ROOT}/3-remote-gambling-and-software-technical-standards"
UKGC_OUTPUT_DIR = Path("data/raw/compliance/ukgc")
PDF_TARGETS = [
    (
        "https://raw.githubusercontent.com/alexbotov/rgs/main/docs/"
        "GLI-19-Interactive-Gaming-Systems-v3.0.pdf",
        Path("data/raw/compliance/gli/GLI-19-v3.0.pdf"),
    ),
    (
        "https://www.nj.gov/oag/ge/docs/Regulations/CHAPTER69O.pdf",
        Path("data/raw/compliance/nj_dge/NJAC_13_69O_Internet_Gaming.pdf"),
    ),
    (
        "https://www.nj.gov/oag/ge/docs/Regulations/CHAPTER69D.pdf",
        Path("data/raw/compliance/nj_dge/NJAC_13_69D_Internal_Controls.pdf"),
    ),
    (
        "https://www.nj.gov/oag/ge/docs/Regulations/CHAPTER69E.pdf",
        Path("data/raw/compliance/nj_dge/NJAC_13_69E_Gaming_Equipment.pdf"),
    ),
]


def fetch_html(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def download_pdfs():
    for url, path in PDF_TARGETS:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Expected a PDF response from {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        print(f"Downloaded {path} ({path.stat().st_size:,} bytes)")


def discover_ukgc_pages():
    urls = {UKGC_SECTION_THREE}
    for page_url in (UKGC_ROOT, UKGC_SECTION_THREE):
        soup = BeautifulSoup(fetch_html(page_url), "html.parser")
        for link in soup.find_all("a", href=True):
            candidate = urljoin(page_url, link["href"]).split("#", 1)[0].split("?", 1)[0]
            parsed = urlparse(candidate)
            if (
                parsed.netloc == "www.gamblingcommission.gov.uk"
                and parsed.path.startswith(f"{urlparse(UKGC_ROOT).path}/")
                and candidate != UKGC_ROOT
            ):
                urls.add(candidate)
    return sorted(urls)


def _extract_table_rows(table):
    rows = []
    for row in table.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    caption = table.find("caption")
    if caption:
        rows.insert(0, caption.get_text(" ", strip=True))
    return rows


def extract_ukgc_text(url):
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    main = soup.find("main")
    if main is None:
        raise ValueError(f"Could not find main content at {url}")

    for element in main(["script", "style", "nav", "footer", "button", "form"]):
        element.decompose()

    lines = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        if element.find_parent("table") is not None:
            continue
        if element.name == "p" and element.find_parent("li") is not None:
            continue
        if element.name == "table":
            rows = _extract_table_rows(element)
            if rows:
                lines.extend(rows)
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name == "li":
            lines.append(f"- {text}")
        else:
            heading_level = int(element.name[1]) if element.name.startswith("h") else 0
            lines.append(f"{'#' * heading_level} {text}".strip())

    content = "\n\n".join(lines).strip()
    if not content:
        raise ValueError(f"No readable main content at {url}")
    return content


def download_ukgc_pages():
    UKGC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for url in discover_ukgc_pages():
        slug = urlparse(url).path.rstrip("/").split("/")[-1]
        path = UKGC_OUTPUT_DIR / f"{slug}.md"
        content = extract_ukgc_text(url)
        path.write_text(content, encoding="utf-8")
        print(f"Saved {path} ({len(content):,} characters)")


def main():
    download_pdfs()
    download_ukgc_pages()
    print("Source download complete.")


if __name__ == "__main__":
    main()
