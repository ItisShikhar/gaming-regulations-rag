import hashlib
import json
import os
import pickle
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/compliance_chunks.jsonl")
INDEX_CHUNKS_FILE = Path("data/processed/compliance_index_chunks.jsonl")
BM25_FILE = Path("data/processed/compliance_bm25.pkl")
INDEX_MANIFEST_FILE = Path("data/processed/compliance_index.json")
QDRANT_PATH = "data/qdrant-compliance"
COLLECTION = "compliance"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBED_WITH_HEADING = True
TOKENIZER_VERSION = 3
INDEX_TEXT_VERSION = 1
MAX_INPUT_TOKENS = 220
SECTION_REFERENCE = re.compile(
    r"(?<![A-Z0-9])(?:13:69[ODE]-\d+\.\d+[A-Z]?(?:\([A-Z0-9]+\))*|"
    r"RTS\s*(?:(?:requirement|aim)\s+)?\d+[A-Z]?|GLI-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
SECTION_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+){1,4}(?![\w.])")
WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CLAUSE_MARKER = re.compile(
    r"^\s*(\([A-Za-z0-9]+\)|\d{1,2}\.|[ivxlcdm]{1,6}\.)\s*$",
    re.IGNORECASE,
)


def tokenize(text):
    normalized = text.lower().replace("–", "-")
    identifiers = []
    for match in SECTION_REFERENCE.finditer(normalized):
        identifier = re.sub(r"\s+", "", match.group(0)).replace("-", "")
        identifier = re.sub(r"^rts(?:requirement|aim)", "rts", identifier)
        identifiers.append(identifier)
    identifiers.extend(
        match.group(0) for match in SECTION_NUMBER.finditer(normalized)
    )
    words = [token.lower() for token in WORD.findall(normalized)]
    return words + identifiers


def load_chunks():
    with INDEX_CHUNKS_FILE.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def load_source_chunks():
    with CHUNKS_FILE.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def corpus_fingerprint(chunks):
    serialized = json.dumps(
        chunks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def index_text(chunk):
    parts = []
    if EMBED_WITH_HEADING:
        parts.extend(chunk.get("heading_path", []))
    for key in ("jurisdiction", "regulation_type", "section_id"):
        if chunk.get(key):
            parts.append(str(chunk[key]))
    parts.append(chunk["text"])
    return "\n".join(parts)


def _token_count(tokenizer, chunk, text):
    encoding = tokenizer(
        index_text({**chunk, "text": text}),
        add_special_tokens=True,
        truncation=False,
    )
    return len(encoding["input_ids"])


def _format_clause_paths(clause_paths):
    normalized_paths = [
        tuple(label.strip("() .").lower() for label in path)
        for path in clause_paths
    ]
    leaf_paths = [
        path
        for path in normalized_paths
        if not any(
            len(other) > len(path) and other[:len(path)] == path
            for other in normalized_paths
        )
    ]
    return ", ".join(
        "".join(f"({label})" for label in path)
        for path in dict.fromkeys(leaf_paths)
    )


def split_chunks_for_token_limit(chunks, tokenizer, max_tokens):
    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than zero.")

    indexed_chunks = []
    for chunk in chunks:
        if _token_count(tokenizer, chunk, chunk["text"]) <= max_tokens:
            indexed_chunks.append(chunk)
            continue

        continuation = (
            f"Continuation of clause {''.join(chunk.get('clause_path', []))}: "
            if chunk.get("clause_path")
            else "Continuation: "
        )
        parts = []
        current = ""
        current_clause_paths = []
        active_clause_path = []

        def prefix():
            return "" if not parts else continuation

        def fits(text):
            return _token_count(tokenizer, chunk, prefix() + text) <= max_tokens

        def add_clause_path():
            path = list(active_clause_path)
            if path and path not in current_clause_paths:
                current_clause_paths.append(path)

        def flush():
            nonlocal current, current_clause_paths
            if current:
                parts.append((current, current_clause_paths))
                current = ""
                current_clause_paths = []

        segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?;])\s+|\n+", chunk["text"].strip())
            if segment.strip()
        ]
        for segment in segments:
            marker = CLAUSE_MARKER.fullmatch(segment)
            if marker:
                label = marker.group(1)
                level = (
                    1 if label.startswith("(")
                    else 2 if label.endswith(".") and label[:-1].isdigit()
                    else 3
                )
                active_clause_path = active_clause_path[:level - 1] + [label]

            candidate = f"{current} {segment}".strip()
            if fits(candidate):
                current = candidate
                add_clause_path()
                continue

            flush()
            if fits(segment):
                current = segment
                add_clause_path()
                continue

            for word in segment.split():
                candidate = f"{current} {word}".strip()
                if fits(candidate):
                    current = candidate
                    add_clause_path()
                    continue
                flush()
                if fits(word):
                    current = word
                    add_clause_path()
                    continue

                word_part = ""
                for character in word:
                    candidate = word_part + character
                    if fits(candidate):
                        word_part = candidate
                    else:
                        if not word_part:
                            raise ValueError(
                                f"Chunk metadata alone exceeds the {max_tokens}-token limit."
                            )
                        current = word_part
                        add_clause_path()
                        flush()
                        word_part = character
                if word_part:
                    current = word_part
                    add_clause_path()
                    flush()

        flush()
        if not parts:
            raise ValueError(f"Could not split chunk {chunk['chunk_id']}.")

        for part_index, (part, part_clause_paths) in enumerate(parts):
            text = ("" if part_index == 0 else continuation) + part
            if not part_clause_paths:
                part_clause_paths = chunk.get("clause_paths", [])
            clause_path = part_clause_paths[-1] if part_clause_paths else []
            citation = chunk.get("section_id") or " > ".join(
                chunk.get("heading_path", [])
            )
            if chunk.get("jurisdiction") == "gli" and chunk.get("section_id"):
                citation = f"GLI-19 {citation}"
            if clause_path:
                citation += f" {_format_clause_paths(part_clause_paths)}"
            part_chunk = {
                **chunk,
                "chunk_id": f"{chunk['chunk_id']}-part-{part_index}",
                "text": text,
                "char_count": len(text),
                "clause_path": clause_path,
                "clause_paths": part_clause_paths,
                "citation": citation,
                "continuation": chunk.get("continuation", False) or part_index > 0,
            }
            if _token_count(tokenizer, chunk, text) > max_tokens:
                raise ValueError(
                    f"Chunk {part_chunk['chunk_id']} exceeds {max_tokens} tokens."
                )
            indexed_chunks.append(part_chunk)
    return indexed_chunks


def build_index():
    chunks = load_source_chunks()
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_FILE}")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique before indexing.")
    for chunk in chunks:
        for field in ("text", "jurisdiction", "regulation_type"):
            if not chunk.get(field):
                raise ValueError(f"Chunk {chunk['chunk_id']} has no {field}.")

    print(f"{len(chunks)} compliance chunks loaded")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    if MAX_INPUT_TOKENS > model.max_seq_length:
        raise ValueError(
            f"MAX_INPUT_TOKENS ({MAX_INPUT_TOKENS}) exceeds the model limit "
            f"({model.max_seq_length})."
        )
    chunks = split_chunks_for_token_limit(
        chunks,
        model.tokenizer,
        MAX_INPUT_TOKENS,
    )
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Token-bounded chunk IDs must be unique.")
    fingerprint = corpus_fingerprint(chunks)
    print(f"{len(chunks)} token-bounded chunks")
    texts = [index_text(chunk) for chunk in chunks]
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )

    embedding_size = model.get_embedding_dimension()
    if embedding_size is None:
        raise ValueError("Could not determine the model's embedding dimension")

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    BM25_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_chunks_file = INDEX_CHUNKS_FILE.with_suffix(".jsonl.tmp")
    temp_bm25_file = BM25_FILE.with_suffix(".pkl.tmp")
    temp_manifest_file = INDEX_MANIFEST_FILE.with_suffix(".json.tmp")
    try:
        with temp_chunks_file.open("w", encoding="utf-8") as output_file:
            for chunk in chunks:
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        with temp_bm25_file.open("wb") as output_file:
            pickle.dump({
                "bm25": bm25,
                "chunk_ids": chunk_ids,
                "corpus_fingerprint": fingerprint,
                "index_text_version": INDEX_TEXT_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
            }, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())

        manifest = {
            "collection": COLLECTION,
            "corpus_fingerprint": fingerprint,
            "embedding_model": MODEL_NAME,
            "embedding_revision": MODEL_REVISION,
            "index_text_version": INDEX_TEXT_VERSION,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "point_count": len(chunks),
            "tokenizer_version": TOKENIZER_VERSION,
        }
        with temp_manifest_file.open("w", encoding="utf-8") as output_file:
            json.dump(manifest, output_file, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        client = QdrantClient(path=QDRANT_PATH)
        try:
            if client.collection_exists(COLLECTION):
                client.delete_collection(COLLECTION)
            client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(
                    size=embedding_size, distance=Distance.COSINE
                ),
            )
            for start in range(0, len(points), 256):
                client.upsert(COLLECTION, points=points[start:start + 256])

            point_count = client.count(COLLECTION).count
            if point_count != len(chunks):
                raise RuntimeError(
                    f"Indexed {point_count} points, expected {len(chunks)}."
                )
            print("qdrant points:", point_count)
        finally:
            client.close()
        os.replace(temp_chunks_file, INDEX_CHUNKS_FILE)
        os.replace(temp_bm25_file, BM25_FILE)
        os.replace(temp_manifest_file, INDEX_MANIFEST_FILE)
    finally:
        if temp_chunks_file.exists():
            temp_chunks_file.unlink()
        if temp_bm25_file.exists():
            temp_bm25_file.unlink()
        if temp_manifest_file.exists():
            temp_manifest_file.unlink()


def main():
    build_index()


if __name__ == "__main__":
    main()