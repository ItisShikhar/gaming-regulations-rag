import argparse
import json
import pickle
import re
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from index import (
    BM25_TEXT_VERSION,
    BM25_FILE,
    COLLECTION,
    INDEX_TEXT_VERSION,
    INDEX_MANIFEST_FILE,
    INDIA_JURISDICTIONS,
    INDIA_JURISDICTION_NAMES,
    INDIA_JURISDICTION_TYPES,
    MODEL_NAME,
    MODEL_REVISION,
    QDRANT_PATH,
    TOKENIZER_VERSION,
    corpus_fingerprint,
    load_chunks,
    tokenize,
)

RRF_K = 60
DEPTH = 100
BASE_JURISDICTIONS = ("ukgc", "nj_dge", "gli")
JURISDICTIONS = BASE_JURISDICTIONS + INDIA_JURISDICTIONS
REGULATION_TYPES = (
    "technical_standard",
    "administrative_code",
    "central_act",
    "central_rule",
    "central_notification",
    "central_draft_notice",
    "state_act",
    "state_rule",
    "state_notification",
    "union_territory_act",
    "union_territory_rule",
    "union_territory_notification",
)
STATUTE_REFERENCE = re.compile(
    r"\b(13:69[ODE]-\d+\.\d+[A-Z]?)(?:\([A-Z0-9]+\))*",
    re.IGNORECASE,
)
RTS_REFERENCE = re.compile(
    r"\bRTS\s+(?:(?:requirement|aim)\s+)?(\d+[A-Z]?)\b",
    re.IGNORECASE,
)
SECTION_REFERENCE = re.compile(
    r"\b(?:section|§)\s*(\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
JURISDICTION_ALIASES = {
    "ukgc": re.compile(r"\b(?:ukgc|uk|united kingdom|britain)\b|rts\s*\d", re.IGNORECASE),
    "nj_dge": re.compile(
        r"\b(?:new jersey|nj dge|njac)\b|13:69[ode](?:-|\b)",
        re.IGNORECASE,
    ),
    "gli": re.compile(r"\bgli(?:\s*-\s*\d+|\s+\d+)\b", re.IGNORECASE),
}
INDIA_REGION_ALIASES = {
    code: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for code, name in INDIA_JURISDICTION_NAMES.items()
    if code != "india_central"
}

_MODEL = None
_CLIENT = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QdrantClient(path=QDRANT_PATH)
    return _CLIENT


def close_search_client():
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None


def _validate_filters(filters):
    normalized = {}
    for key, value in (filters or {}).items():
        if key == "jurisdiction":
            allowed = JURISDICTIONS
        elif key == "regulation_type":
            allowed = REGULATION_TYPES
        elif key == "jurisdiction_type":
            allowed = tuple(sorted(set(INDIA_JURISDICTION_TYPES.values())))
        elif key == "country":
            allowed = ("India",)
        else:
            raise ValueError(f"Unsupported compliance filter: {key}")
        if value not in allowed:
            raise ValueError(f"Invalid {key} {value!r}. Choose from {', '.join(allowed)}.")
        normalized[key] = value
    return normalized


def _validate_limit(k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")


def _qdrant_filter(filters):
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in _validate_filters(filters).items()
    ]
    return Filter(must=conditions) if conditions else None


def infer_jurisdictions(query):
    india_regions = [
        jurisdiction
        for jurisdiction, pattern in INDIA_REGION_ALIASES.items()
        if pattern.search(query)
    ]
    if india_regions:
        return india_regions
    if re.search(r"\bIndia\b|\bIndian\b", query, re.IGNORECASE):
        return list(INDIA_JURISDICTIONS)
    return [
        jurisdiction
        for jurisdiction, pattern in JURISDICTION_ALIASES.items()
        if pattern.search(query)
    ]


@lru_cache(maxsize=1)
def _assert_index_current():
    if not INDEX_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "The compliance index manifest is missing. Run `python src/index.py`."
        )

    with INDEX_MANIFEST_FILE.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)

    chunks = _load_chunks()
    fingerprint = corpus_fingerprint(chunks)
    if manifest.get("corpus_fingerprint") != fingerprint:
        raise ValueError(
            "The compliance index does not match the current chunks. "
            "Run `python src/ingest.py` and `python src/index.py` again."
        )
    if manifest.get("embedding_model") != MODEL_NAME:
        raise ValueError(
            "The compliance index uses a different embedding model. "
            "Run `python src/index.py` again."
        )
    if manifest.get("embedding_revision") != MODEL_REVISION:
        raise ValueError(
            "The compliance index uses a different embedding model revision. "
            "Run `python src/index.py` again."
        )
    if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
        raise ValueError(
            "The compliance index uses a different tokenizer. "
            "Run `python src/index.py` again."
        )
    if manifest.get("index_text_version") != INDEX_TEXT_VERSION:
        raise ValueError(
            "The compliance index uses a different embedding text format. "
            "Run `python src/index.py` again."
        )
    if manifest.get("bm25_text_version") != BM25_TEXT_VERSION:
        raise ValueError(
            "The compliance index uses a different BM25 text format. "
            "Run `python src/index.py` again."
        )
    collection_name = manifest.get("collection")
    if not isinstance(collection_name, str) or not (
        collection_name == COLLECTION
        or collection_name.startswith(f"{COLLECTION}_")
    ):
        raise ValueError("The compliance index manifest has an invalid collection name.")
    point_count = _get_client().count(collection_name, exact=True).count
    if point_count != len(chunks) or manifest.get("point_count") != len(chunks):
        raise ValueError(
            f"The compliance index has {point_count} points for {len(chunks)} chunks. "
            "Run `python src/index.py` again."
        )
    return collection_name


@lru_cache(maxsize=1)
def _load_bm25():
    _assert_index_current()
    with BM25_FILE.open("rb") as input_file:
        saved = pickle.load(input_file)
    if (
        not isinstance(saved, dict)
        or saved.get("tokenizer_version") != TOKENIZER_VERSION
        or saved.get("index_text_version") != INDEX_TEXT_VERSION
        or saved.get("bm25_text_version") != BM25_TEXT_VERSION
    ):
        raise ValueError(
            "The compliance BM25 index is stale or incompatible. "
            "Run `python src/index.py` again."
        )
    return saved["bm25"], saved["chunk_ids"], saved.get("corpus_fingerprint")


@lru_cache(maxsize=1)
def _load_chunks():
    return load_chunks()


def _assert_bm25_alignment(chunks, bm25_chunk_ids, bm25_fingerprint):
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if (
        chunk_ids != bm25_chunk_ids
        or corpus_fingerprint(chunks) != bm25_fingerprint
    ):
        raise ValueError(
            "The compliance BM25 index does not match its chunk file. "
            "Run `python src/index.py` again."
        )


def _matches_filters(chunk, filters):
    return all(chunk.get(key) == value for key, value in filters.items())


def _exact_reference_ids(query, chunks, filters):
    statute_references = [
        (
            re.sub(r"\s+", "", match.group(1)).lower(),
            tuple(
                label.lower()
                for label in re.findall(
                    r"\(([A-Z0-9]+)\)", match.group(0), re.IGNORECASE
                )
            ),
        )
        for match in STATUTE_REFERENCE.finditer(query)
    ]
    rts_matches = [
        f"RTS {match.group(1)}".lower()
        for match in RTS_REFERENCE.finditer(query)
    ]
    section_matches = [
        match.group(1) for match in SECTION_REFERENCE.finditer(query)
    ]
    query_mentions_gli = bool(re.search(r"\bGLI-?\d+\b", query, re.IGNORECASE))

    if not (statute_references or rts_matches or section_matches):
        return []

    exact_ids = []
    matched_statutes = set()
    matched_rts = set()
    matched_sections = set()
    for index, chunk in enumerate(chunks):
        if not _matches_filters(chunk, filters):
            continue
        section_id = chunk.get("section_id") or ""
        chunk_clause_paths = {
            tuple(label.strip("() .").lower() for label in clause_path)
            for clause_path in chunk.get("clause_paths", [])
        }
        matched_reference = False
        for reference in statute_references:
            if reference in matched_statutes:
                continue
            section_reference, clause_reference = reference
            if (
                section_id.lower() == section_reference
                and (
                    not clause_reference
                    or clause_reference in chunk_clause_paths
                )
            ):
                matched_statutes.add(reference)
                matched_reference = True
        for reference in rts_matches:
            if reference not in matched_rts and section_id.lower() == reference:
                matched_rts.add(reference)
                matched_reference = True
        for reference in section_matches:
            if (
                reference not in matched_sections
                and section_id == reference
                and (not query_mentions_gli or chunk.get("jurisdiction") == "gli")
            ):
                matched_sections.add(reference)
                matched_reference = True
        if matched_reference:
            exact_ids.append(index)
    return exact_ids


def _dense_ids(query, k, filters=None):
    collection_name = _assert_index_current()
    vector = _get_model().encode(query, normalize_embeddings=True).tolist()
    result = _get_client().query_points(
        collection_name,
        query=vector,
        query_filter=_qdrant_filter(filters),
        limit=k,
    )
    return [point.id for point in result.points]


def _bm25_ids(query, k, filters=None):
    bm25, chunk_ids, fingerprint = _load_bm25()
    chunks = _load_chunks()
    _assert_bm25_alignment(chunks, chunk_ids, fingerprint)
    scores = bm25.get_scores(tokenize(query))
    eligible = [
        index for index, chunk in enumerate(chunks)
        if _matches_filters(chunk, filters or {})
    ]
    ranked = sorted(eligible, key=lambda index: (-scores[index], index))
    return ranked[:k]


def _rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]


def _chunk_ids_to_chunks(chunk_ids):
    chunks = _load_chunks()
    return [chunks[int(chunk_id)] for chunk_id in chunk_ids]


def _apply_post_filter(chunk_ids, filters):
    chunks = _load_chunks()
    return [
        chunk_id
        for chunk_id in chunk_ids
        if _matches_filters(chunks[int(chunk_id)], filters)
    ]


def _hybrid_ids(query, k, filters=None, mode="pre"):
    normalized_filters = _validate_filters(filters)
    if mode == "pre":
        dense = _dense_ids(query, DEPTH, normalized_filters)
        sparse = _bm25_ids(query, DEPTH, normalized_filters)
    elif mode == "post":
        dense = _apply_post_filter(
            _dense_ids(query, DEPTH), normalized_filters
        )
        sparse = _apply_post_filter(
            _bm25_ids(query, DEPTH), normalized_filters
        )
    else:
        raise ValueError("Hybrid mode must be 'pre' or 'post'.")

    fused = _rrf([dense, sparse], k=DEPTH)
    exact = _exact_reference_ids(query, _load_chunks(), normalized_filters)
    if mode == "post":
        candidate_ids = set(fused)
        exact = [chunk_id for chunk_id in exact if chunk_id in candidate_ids]
    if exact:
        fused = exact + [chunk_id for chunk_id in fused if chunk_id not in exact]
    return fused[:k]


def dense_search(query, k=5, filters=None, mode="pre"):
    _validate_limit(k)
    if mode not in ("pre", "post"):
        raise ValueError("Filter mode must be 'pre' or 'post'.")
    selected_filters = _validate_filters(filters)
    depth = DEPTH if mode == "post" else k
    ids = _dense_ids(
        query,
        depth,
        None if mode == "post" else selected_filters,
    )
    if mode == "post":
        ids = _apply_post_filter(ids, selected_filters)
    return _chunk_ids_to_chunks(ids[:k])


def bm25_search(query, k=5, filters=None, mode="pre"):
    _validate_limit(k)
    if mode not in ("pre", "post"):
        raise ValueError("Filter mode must be 'pre' or 'post'.")
    selected_filters = _validate_filters(filters)
    ids = _bm25_ids(
        query,
        DEPTH if mode == "post" else k,
        None if mode == "post" else selected_filters,
    )
    if mode == "post":
        ids = _apply_post_filter(ids, selected_filters)
    return _chunk_ids_to_chunks(ids[:k])


def hybrid_search(query, k=5, filters=None, mode="pre"):
    _validate_limit(k)
    return _chunk_ids_to_chunks(_hybrid_ids(query, k, filters, mode))


def multi_query_search(query, k=5, filters=None, jurisdictions=None):
    _validate_limit(k)
    selected_filters = _validate_filters(filters)
    if jurisdictions is None:
        if "jurisdiction" in selected_filters:
            jurisdictions = [selected_filters["jurisdiction"]]
        else:
            inferred = infer_jurisdictions(query)
            jurisdictions = inferred or list(BASE_JURISDICTIONS)
    elif not jurisdictions:
        raise ValueError("Pass at least one jurisdiction to multi-query search.")

    if "jurisdiction" in selected_filters:
        jurisdictions = [selected_filters["jurisdiction"]]
    for jurisdiction in jurisdictions:
        _validate_filters({"jurisdiction": jurisdiction})

    rankings = []
    for jurisdiction in jurisdictions:
        route_filters = {**selected_filters, "jurisdiction": jurisdiction}
        rankings.append(_hybrid_ids(query, k, route_filters, "pre"))
    return _chunk_ids_to_chunks(_rrf(rankings, k))


def compliance_search(query, k=5, filters=None, mode="pre"):
    _validate_limit(k)
    selected_filters = _validate_filters(filters)
    if mode == "multi_query":
        return multi_query_search(query, k, selected_filters)

    inferred = infer_jurisdictions(query)
    if "jurisdiction" not in selected_filters and len(inferred) == 1:
        selected_filters["jurisdiction"] = inferred[0]
    return hybrid_search(query, k, selected_filters, mode)


def compare_modes(query, k=5, filters=None):
    return {
        "pre": compliance_search(query, k, filters, "pre"),
        "post": compliance_search(query, k, filters, "post"),
        "multi_query": compliance_search(query, k, filters, "multi_query"),
    }


def main():
    parser = argparse.ArgumentParser(description="Search the local compliance indexes.")
    parser.add_argument("query", help="Question or search phrase.")
    parser.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid", "pre", "post", "multi_query", "compare"),
        default="hybrid",
        help="Compare pre-filtering, post-filtering, and per-jurisdiction routing.",
    )
    parser.add_argument("--jurisdiction", choices=JURISDICTIONS)
    parser.add_argument("--regulation-type", choices=REGULATION_TYPES)
    parser.add_argument(
        "--jurisdiction-type",
        choices=tuple(sorted(set(INDIA_JURISDICTION_TYPES.values()))),
    )
    parser.add_argument("--country", choices=("India",))
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k must be at least 1")

    filters = {}
    if args.jurisdiction:
        filters["jurisdiction"] = args.jurisdiction
    if args.regulation_type:
        filters["regulation_type"] = args.regulation_type
    if args.jurisdiction_type:
        filters["jurisdiction_type"] = args.jurisdiction_type
    if args.country:
        filters["country"] = args.country

    if args.mode == "compare":
        results = compare_modes(args.query, args.k, filters)
    elif args.mode == "multi_query":
        results = {
            "multi_query": multi_query_search(args.query, args.k, filters)
        }
    elif args.mode == "dense":
        results = {"dense": dense_search(args.query, args.k, filters)}
    elif args.mode == "bm25":
        results = {"bm25": bm25_search(args.query, args.k, filters)}
    else:
        mode = args.mode if args.mode in ("pre", "post") else "pre"
        results = {
            args.mode: hybrid_search(args.query, args.k, filters, mode)
        }

    for mode_name, chunks in results.items():
        print(f"\n{mode_name} ({len(chunks)} results)")
        for chunk in chunks:
            page = (
                f" pages {chunk['page_start']}-{chunk['page_end']}"
                if chunk.get("page_start") is not None
                else ""
            )
            print(
                f"  [{chunk['jurisdiction']}] {chunk['citation']}"
                f"{page} - {chunk['file']}"
            )

if __name__ == "__main__":
    try:
        main()
    finally:
        close_search_client()
