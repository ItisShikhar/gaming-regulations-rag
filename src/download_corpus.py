import json
import re
import time
import unicodedata
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
INDIA_OUTPUT_DIR = Path("data/raw/compliance/india")
INDIACODE_API = "https://indiacode.gov.in/server/api"
INDIA_CODE_TERMS = (
    "gambling",
    "gaming",
    "online games",
    "lottery",
    "casino",
    "prize competition",
    "betting",
)
INDIA_LAW_TITLE = re.compile(
    r"\b(gambl\w*|gaming|online game\w*|lotter\w*|casino\w*|"
    r"prize compet\w*|betting|wager\w*)\b",
    re.IGNORECASE,
)
INDIA_IT_INTERMEDIARY_RULES = re.compile(
    r"information technology.*intermediary.*guidelines.*rules",
    re.IGNORECASE,
)
INDIA_EXCLUDED_TITLES = re.compile(
    r"\b(prevention of dangerous activities|anti[- ]social|"
    r"gst|goods and services tax|tax on services)\b",
    re.IGNORECASE,
)
INDIA_UNION_TERRITORIES = {
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
}
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
INDIA_DRAFT_RULES_NOTICE = (
    "https://www.meity.gov.in/static/uploads/2025/10/"
    "275f784cc6401f46214a02107e941b91.pdf"
)


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


def _metadata_value(metadata, key, default=""):
    values = metadata.get(key, [])
    for value in values:
        if value.get("value"):
            return str(value["value"])
    return default


def _safe_slug(text):
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug or "unknown"


def _jurisdiction_record(community):
    name = community["name"]
    if name == "CENTRAL":
        return {
            "jurisdiction": "india_central",
            "jurisdiction_name": "India (Central)",
            "jurisdiction_type": "central",
            "country": "India",
        }
    is_union_territory = name in INDIA_UNION_TERRITORIES
    category = "ut" if is_union_territory else "state"
    return {
        "jurisdiction": f"india_{category}_{_safe_slug(name).replace('-', '_')}",
        "jurisdiction_name": name,
        "jurisdiction_type": "union_territory" if is_union_territory else "state",
        "country": "India",
    }


def _india_code_pages(session, query, scope):
    endpoint = f"{INDIACODE_API}/discover/search/objects"
    page_number = 0
    total_pages = 1
    while page_number < total_pages:
        response = session.get(
            endpoint,
            params={
                "query": query,
                "scope": scope,
                "size": 100,
                "page": page_number,
            },
            timeout=45,
        )
        response.raise_for_status()
        search_result = response.json()["_embedded"]["searchResult"]
        total_pages = search_result["page"]["totalPages"]
        yield from search_result.get("_embedded", {}).get("objects", [])
        page_number += 1


def _india_document_type(jurisdiction_type, title):
    title_lower = title.lower()
    if "draft" in title_lower:
        suffix = "draft_rule"
    elif re.search(r"\b(rules|regulations)\b", title_lower):
        suffix = "rule"
    elif "notification" in title_lower or "order" in title_lower:
        suffix = "notification"
    else:
        suffix = "act"
    return f"{jurisdiction_type}_{suffix}"


def _india_source_candidates(session, community):
    jurisdiction = _jurisdiction_record(community)
    candidates = {}
    for term in INDIA_CODE_TERMS:
        for result in _india_code_pages(session, term, community["uuid"]):
            item = result.get("_embedded", {}).get("indexableObject", {})
            metadata = item.get("metadata", {})
            title = _metadata_value(metadata, "dc.title", item.get("name", "")).strip()
            act_id = _metadata_value(metadata, "dc.identifier.act_id")
            section_id = _metadata_value(metadata, "dc.identifier.section_id")
            if not title or not act_id or section_id:
                continue
            is_online_gaming_intermediary_rules = bool(
                INDIA_IT_INTERMEDIARY_RULES.search(title)
            )
            if (
                not INDIA_LAW_TITLE.search(title)
                and not is_online_gaming_intermediary_rules
            ) or INDIA_EXCLUDED_TITLES.search(title):
                continue

            item_uuid = item.get("uuid") or item.get("id")
            handle = item.get("handle")
            if not item_uuid or not handle:
                continue
            document = {
                **jurisdiction,
                "title": title,
                "handle": handle,
                "item_uuid": item_uuid,
                "act_id": act_id,
                "act_year": _metadata_value(metadata, "dc.date.act_year"),
                "enactment_date": _metadata_value(metadata, "dc.date.enact_date"),
                "repealed_status": _metadata_value(
                    metadata, "dc.identifier.repealed", "unknown"
                ),
                "document_status": (
                    "repealed"
                    if _metadata_value(metadata, "dc.identifier.repealed").lower() == "true"
                    else "not_marked_repealed"
                ),
                "legal_effect_status": "not_verified",
                "commencement_status": "not_verified",
                "source_url": f"https://indiacode.gov.in/handle/{handle}",
                "api_url": f"{INDIACODE_API}/core/items/{item_uuid}",
                "source_system": "India Code",
            }
            if handle == "123456789/496509":
                document["official_gazette_url"] = (
                    "https://egazette.gov.in/WriteReadData/2025/265635.pdf"
                )
                document["official_publication_date"] = "2025-08-22"
                document["commencement_date"] = "2026-05-01"
                document["commencement_notification_reference"] = (
                    "S.O. 1994 (E), dated 2026-04-22, Gazette of India, "
                    "Extraordinary, Part II, Section 3(ii)"
                )
                document["legal_effect_status"] = (
                    "India Code consolidated text records the commencement date. "
                    "The commencement notification is not a separate corpus source."
                )
            document["regulation_type"] = _india_document_type(
                document["jurisdiction_type"], title
            )
            candidates[item_uuid] = document
        time.sleep(0.05)
    return list(candidates.values())


def _india_text_from_item(session, item_uuid, include_original=True):
    item_url = f"{INDIACODE_API}/core/items/{item_uuid}"
    item_response = session.get(item_url, timeout=45)
    item_response.raise_for_status()
    item = item_response.json()
    bundles_url = item.get("_links", {}).get("bundles", {}).get("href")
    if not bundles_url:
        raise ValueError(f"IndiaCode item {item_uuid} has no bundle link.")
    bundles_response = session.get(bundles_url, params={"size": 100}, timeout=45)
    bundles_response.raise_for_status()
    bundles = bundles_response.json().get("_embedded", {}).get("bundles", [])
    bundle_by_name = {bundle.get("name"): bundle for bundle in bundles}

    text_bytes = None
    text_bundle = bundle_by_name.get("TEXT")
    if text_bundle:
        bitstreams_url = text_bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if bitstreams_url:
            response = session.get(bitstreams_url, params={"size": 100}, timeout=45)
            response.raise_for_status()
            bitstreams = response.json().get("_embedded", {}).get("bitstreams", [])
            if bitstreams:
                content_url = bitstreams[0].get("_links", {}).get("content", {}).get("href")
                if content_url:
                    content_response = session.get(content_url, timeout=60)
                    content_response.raise_for_status()
                    text_bytes = content_response.content

    original_bytes = None
    original_bundle = bundle_by_name.get("ORIGINAL") if include_original else None
    if original_bundle:
        bitstreams_url = original_bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if bitstreams_url:
            response = session.get(bitstreams_url, params={"size": 100}, timeout=45)
            response.raise_for_status()
            bitstreams = response.json().get("_embedded", {}).get("bitstreams", [])
            if bitstreams:
                content_url = bitstreams[0].get("_links", {}).get("content", {}).get("href")
                if content_url:
                    content_response = session.get(content_url, timeout=60)
                    content_response.raise_for_status()
                    original_bytes = content_response.content

    text = ""
    if text_bytes:
        text = text_bytes.decode("utf-8", errors="replace")
    elif original_bytes and original_bytes.startswith(b"%PDF"):
        import pymupdf

        with pymupdf.open(stream=original_bytes, filetype="pdf") as document:
            page_text = []
            for page_number, page in enumerate(document, start=1):
                page_text.append(f"<!-- Page {page_number} -->\n{page.get_text('text')}")
            text = "\n".join(page_text)
    if not text.strip():
        raise ValueError(f"IndiaCode item {item_uuid} has no extractable text.")
    return text, original_bytes


def _format_india_code_markdown(title, raw_text):
    lines = [f"# {title}", ""]
    current_page = 0
    statute_started = False
    in_contents = False
    enacting_context = ""
    contents_lines = []
    contents_section_numbers = set()
    is_rules_document = bool(re.search(r"\b(rules|regulations)\b", title, re.IGNORECASE))

    for raw_line in raw_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            if lines[-1] != "":
                lines.append("")
            continue
        page_match = re.fullmatch(r"\d{1,3}", line)
        if page_match and int(line) == current_page + 1:
            current_page += 1
            if not in_contents:
                lines.extend(["", f"<!-- Page {current_page} -->", ""])
            continue
        if re.search(
            r"\b(arrangement of sections|table of contents|table of provisions)\b",
            line,
            re.IGNORECASE,
        ):
            in_contents = True
            statute_started = False
            enacting_context = ""
            continue
        if in_contents:
            contents_lines.append(line)
            listed_section = re.match(r"^(\d+[A-Z]?)\.\s+", line, re.IGNORECASE)
            if listed_section:
                section_number = listed_section.group(1).upper()
                if section_number in contents_section_numbers:
                    in_contents = False
                    statute_started = True
                    contents_lines.clear()
                    contents_section_numbers.clear()
                else:
                    contents_section_numbers.add(section_number)
            enacting_context = f"{enacting_context} {line}"[-1000:]
            if re.search(
                r"\b(enacted|enacts)\b.{0,500}\b(as follows|following)\b|"
                r"\b(promulgate[sd]?|make[sd]?|made)\b.{0,400}"
                r"\b(following|as follows)\b",
                enacting_context,
                re.IGNORECASE,
            ):
                in_contents = False
                statute_started = True
            else:
                if in_contents:
                    continue
            contents_lines.clear()
            contents_section_numbers.clear()
        enacting_context = f"{enacting_context} {line}"[-1000:]
        if re.search(
            r"\b(enacted|enacts)\b.{0,500}\b(as follows|following)\b|"
            r"\bfollowing rules\b|"
            r"\bmake the following rules\b",
            enacting_context,
            re.IGNORECASE,
        ):
            statute_started = True
        if not in_contents and is_rules_document and re.match(r"^1\.\s", line):
            statute_started = True

        section_match = re.match(r"^(\d+[A-Z]?)\.\s+(.+)$", line, re.IGNORECASE)
        if (
            section_match
            and section_match.group(1) == "1"
            and re.match(r"^\s*\(1\)", section_match.group(2))
        ):
            statute_started = True
        if re.match(r"^\d+\.\s+For Statement of Objects", line, re.IGNORECASE):
            section_match = None
        if statute_started and section_match:
            lines.extend(["", f"## Section {section_match.group(1).upper()}", ""])
        lines.append(line)
    if in_contents and contents_lines:
        lines.extend(["", "## Contents", "", *contents_lines])
    return "\n".join(lines).strip() + "\n"


def download_india_code_sources():
    INDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    community_response = session.get(
        f"{INDIACODE_API}/core/communities",
        params={"size": 100},
        timeout=45,
    )
    community_response.raise_for_status()
    communities = community_response.json().get("_embedded", {}).get("communities", [])
    if not communities:
        raise ValueError("IndiaCode returned no state or central communities.")

    catalog = {
        "source_system": "India Code",
        "api_base": INDIACODE_API,
        "search_terms": list(INDIA_CODE_TERMS),
        "coverage_note": (
            "A no-title-match status means the configured India Code title searches "
            "found no candidate. It does not mean that the jurisdiction has no gaming law."
        ),
        "jurisdictions": [],
        "documents": [],
        "errors": [],
    }
    for community in communities:
        jurisdiction = _jurisdiction_record(community)
        try:
            documents = _india_source_candidates(session, community)
            jurisdiction_record = {
                **jurisdiction,
                "community_handle": community.get("handle"),
                "community_uuid": community.get("uuid") or community.get("id"),
                "source_status": (
                    "title_matches_found_in_indiacode_search"
                    if documents
                    else "no_title_match_in_indiacode_search"
                ),
                "candidate_count": len(documents),
            }
            catalog["jurisdictions"].append(jurisdiction_record)
            for document in documents:
                folder = INDIA_OUTPUT_DIR / jurisdiction["jurisdiction"]
                folder.mkdir(parents=True, exist_ok=True)
                handle_suffix = _safe_slug(document["handle"].split("/")[-1])
                file_stem = f"india-code-{handle_suffix}"
                markdown_path = folder / f"{file_stem}.md"
                metadata_path = folder / f"{file_stem}.source.json"
                try:
                    raw_text, original_bytes = _india_text_from_item(
                        session, document["item_uuid"]
                    )
                    markdown_path.write_text(
                        _format_india_code_markdown(document["title"], raw_text),
                        encoding="utf-8",
                    )
                    document["file"] = markdown_path.relative_to(
                        Path("data/raw/compliance")
                    ).as_posix()
                    document["has_original_pdf"] = bool(
                        original_bytes and original_bytes.startswith(b"%PDF")
                    )
                    if document["has_original_pdf"]:
                        pdf_path = folder / f"{file_stem}.pdf"
                        pdf_path.write_bytes(original_bytes)
                        document["pdf_file"] = pdf_path.relative_to(
                            Path("data/raw/compliance")
                        ).as_posix()
                    metadata_path.write_text(
                        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    catalog["documents"].append(document)
                    print(
                        f"Saved {document['jurisdiction_name']}: "
                        f"{document['title']}"
                    )
                except (requests.RequestException, ValueError, OSError) as error:
                    catalog["errors"].append({
                        "jurisdiction": jurisdiction["jurisdiction"],
                        "handle": document["handle"],
                        "title": document["title"],
                        "error": str(error),
                    })
        except (requests.RequestException, ValueError, KeyError) as error:
            catalog["errors"].append({
                "jurisdiction": jurisdiction["jurisdiction"],
                "error": str(error),
            })

    catalog_path = INDIA_OUTPUT_DIR / "source_catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failures = len(catalog["errors"])
    print(
        f"IndiaCode: {len(catalog['documents'])} documents from "
        f"{len(catalog['jurisdictions'])} jurisdictions, {failures} errors."
    )
    if failures:
        raise RuntimeError(
            f"IndiaCode source collection had {failures} errors. "
            f"See {catalog_path} for details."
        )


def download_india_draft_notice():
    url = INDIA_DRAFT_RULES_NOTICE
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Expected a PDF response from {url}")

    folder = INDIA_OUTPUT_DIR / "india_central"
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / "meity-draft-online-gaming-rules-notice-2025.pdf"
    markdown_path = folder / "meity-draft-online-gaming-rules-notice-2025.md"
    metadata_path = folder / "meity-draft-online-gaming-rules-notice-2025.source.json"
    pdf_path.write_bytes(response.content)

    import pymupdf

    with pymupdf.open(pdf_path) as document:
        pages = [
            f"<!-- Page {page_number} -->\n{page.get_text('text')}"
            for page_number, page in enumerate(document, start=1)
        ]
    title = "MeitY notice on the Draft Promotion and Regulation of Online Gaming Rules, 2025"
    markdown_path.write_text(f"# {title}\n\n" + "\n\n".join(pages), encoding="utf-8")
    metadata = {
        "jurisdiction": "india_central",
        "jurisdiction_name": "India (Central)",
        "jurisdiction_type": "central",
        "country": "India",
        "regulation_type": "central_draft_notice",
        "document_status": "draft_notice",
        "legal_effect_status": "draft_notice",
        "commencement_status": "not_applicable",
        "title": title,
        "source_system": "Ministry of Electronics and Information Technology",
        "source_url": url,
        "published_date": "2025-10-02",
        "file": markdown_path.relative_to(Path("data/raw/compliance")).as_posix(),
        "pdf_file": pdf_path.relative_to(Path("data/raw/compliance")).as_posix(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    catalog_path = INDIA_OUTPUT_DIR / "source_catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    else:
        catalog = {
            "source_system": "India Code",
            "jurisdictions": [],
            "documents": [],
            "errors": [],
        }
    catalog["documents"] = [
        document
        for document in catalog["documents"]
        if document.get("file") != metadata["file"]
    ]
    catalog["documents"].append(metadata)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved central draft-rule notice: {pdf_path}")


def main():
    download_pdfs()
    download_ukgc_pages()
    download_india_code_sources()
    download_india_draft_notice()
    print("Source download complete.")


if __name__ == "__main__":
    main()
