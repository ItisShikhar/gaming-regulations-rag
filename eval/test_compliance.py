import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_corpus import (
    _extract_table_rows,
    _format_india_code_markdown,
    _india_document_type,
)
from index import (
    INDIA_JURISDICTIONS,
    bm25_text,
    bm25_texts,
    corpus_fingerprint,
    remove_orphaned_compliance_collections,
    tokenize,
)
from index import index_text, split_chunks_for_token_limit
from ingest import (
    COMPLIANCE_MAX_CHARS,
    _build_compliance_chunks,
    _parse_gli_sections,
    _split_long_block,
)
from search import (
    _assert_bm25_alignment,
    _matches_filters,
    _exact_reference_ids,
    _rrf,
    _validate_filters,
    infer_jurisdictions,
)
from run_eval import first_hit_rank


class ComplianceChunkingTests(unittest.TestCase):
    def test_new_jersey_chunks_keep_section_and_clause_paths(self):
        source = (
            "CHAPTER 69O\n"
            "SUBCHAPTER 1. GENERAL PROVISIONS\n"
            "13:69O-1.2\n"
            "General requirements\n"
            "(a)\n"
            "Account holders must register before wagering.\n"
            "(e)\n"
            "The system must determine the patron's location.\n"
            "1.\n"
            "Mobile gaming must occur within an approved facility.\n"
            "13:69O-1.3\n"
            "Account requirements\n"
            "(a)\n"
            "The operator must maintain an account record.\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "temporary-compliance-source.md"
            path.write_text(source, encoding="utf-8")
            chunks = _build_compliance_chunks(
                path,
                "nj_dge/temporary-compliance-source.md",
                "nj_dge",
                "administrative_code",
                "https://example.invalid/rules",
            )

        location_chunks = [chunk for chunk in chunks if chunk["section_id"] == "13:69O-1.2"]
        self.assertTrue(location_chunks)
        self.assertEqual(location_chunks[0]["jurisdiction"], "nj_dge")
        self.assertEqual(location_chunks[0]["regulation_type"], "administrative_code")
        self.assertIn(["(e)", "1."], location_chunks[0]["clause_paths"])
        self.assertIn("13:69O-1.2", location_chunks[0]["citation"])
        self.assertIn("(e)(1)", location_chunks[0]["citation"])
        self.assertNotIn("(e) (e)(1)", location_chunks[0]["citation"])
        self.assertIn("page_start", location_chunks[0])
        self.assertEqual(
            _exact_reference_ids(
                "N.J.A.C. 13:69O-1.2(e)(1)",
                location_chunks,
                {"jurisdiction": "nj_dge"},
            ),
            [0],
        )

    def test_gli_numbered_sections_keep_parent_hierarchy(self):
        lines = [
            "<!-- Page 5 -->",
            "Chapter 3: Random Number Generator Requirements",
            "3.2",
            "General RNG Requirements",
            "All outcomes must meet the applicable standard.",
            "3.2.1",
            "Source Code Review",
            "The laboratory reviews source code.",
        ]
        page = None
        page_by_line = []
        from ingest import _page_from_line

        for line in lines:
            page = _page_from_line(line, page)
            page_by_line.append(page)

        sections = _parse_gli_sections(lines, page_by_line)
        child = next(section for section in sections if section["section_id"] == "3.2.1")
        self.assertEqual(
            child["heading_path"],
            [
                "Chapter 3: Random Number Generator Requirements",
                "3.2 General RNG Requirements",
                "3.2.1 Source Code Review",
            ],
        )

    def test_long_clause_splits_only_at_sentence_boundaries(self):
        block = {
            "text": "(a) " + ("First requirement applies. " * 65),
            "page_start": 12,
            "page_end": 12,
            "clause_path": ["(a)"],
        }
        pieces = _split_long_block(block, COMPLIANCE_MAX_CHARS)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(pieces[0]["text"].startswith("(a)"))
        self.assertTrue(pieces[1]["text"].startswith("Continuation of clause (a):"))
        self.assertTrue(all(piece["clause_path"] == ["(a)"] for piece in pieces))
        self.assertTrue(all(piece["page_start"] == 12 for piece in pieces))
        self.assertTrue(all(len(piece["text"]) <= COMPLIANCE_MAX_CHARS for piece in pieces))

    def test_long_block_without_sentence_breaks_stays_within_limit(self):
        block = {
            "text": "x" * 2500,
            "page_start": 12,
            "page_end": 12,
            "clause_path": [],
        }

        pieces = _split_long_block(block, COMPLIANCE_MAX_CHARS)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece["text"]) <= COMPLIANCE_MAX_CHARS for piece in pieces))

    def test_repeated_rts_section_ids_have_unique_chunk_ids(self):
        source = (
            "# Remote standards\n"
            "## RTS 7 - Random outcomes\n"
            "### RTS aim 7\n"
            "The aim describes fair outcomes.\n"
            "#### RTS aim 7\n"
            "This nested heading repeats the same section identifier.\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "standards.md"
            path.write_text(source, encoding="utf-8")
            chunks = _build_compliance_chunks(
                path,
                "ukgc/standards.md",
                "ukgc",
                "technical_standard",
                "https://example.invalid/standards",
            )

        repeated_sections = [
            chunk for chunk in chunks if chunk["section_id"] == "RTS 7"
        ]
        self.assertEqual(len(repeated_sections), 2)
        self.assertEqual(
            len({chunk["chunk_id"] for chunk in repeated_sections}),
            len(repeated_sections),
        )

    def test_index_chunks_respect_the_model_token_limit(self):
        class WordTokenizer:
            def __call__(self, text, **kwargs):
                tokens = re.findall(r"\w+|[^\w\s]", text)
                return {"input_ids": ["start", *tokens, "end"]}

        chunks = [
            {
                "chunk_id": "test-section",
                "text": (
                    "(a)\n"
                    + "Requirement applies. " * 15
                    + "\n(b)\n"
                    + "Another rule applies. " * 15
                ),
                "heading_path": ["Chapter 1", "Section 1.1"],
                "section_id": "1.1",
                "citation": "1.1",
                "jurisdiction": "ukgc",
                "regulation_type": "technical_standard",
                "clause_path": [],
                "continuation": False,
            }
        ]
        tokenizer = WordTokenizer()

        split_chunks = split_chunks_for_token_limit(chunks, tokenizer, 20)

        self.assertGreater(len(split_chunks), 1)
        self.assertEqual(
            len({chunk["chunk_id"] for chunk in split_chunks}),
            len(split_chunks),
        )
        self.assertTrue(
            all(
                len(tokenizer(index_text(chunk))["input_ids"]) <= 20
                for chunk in split_chunks
            )
        )
        self.assertTrue(split_chunks[1]["text"].startswith("Continuation:"))
        self.assertTrue(any(["(a)"] in chunk["clause_paths"] for chunk in split_chunks))
        self.assertTrue(any(["(b)"] in chunk["clause_paths"] for chunk in split_chunks))
        self.assertTrue(any("(a)" in chunk["citation"] for chunk in split_chunks))
        self.assertTrue(any("(b)" in chunk["citation"] for chunk in split_chunks))


class ComplianceRetrievalTests(unittest.TestCase):
    def test_index_startup_removes_only_unregistered_compliance_stores(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            collection_root = root / "collection"
            current_store = collection_root / "compliance_current"
            stale_store = collection_root / "compliance_old"
            unrelated_store = collection_root / "another"
            for path in (current_store, stale_store, unrelated_store):
                path.mkdir(parents=True)
            (root / "meta.json").write_text(
                json.dumps({"collections": {"compliance_current": {}}}),
                encoding="utf-8",
            )

            with patch("index.QDRANT_PATH", str(root)):
                remove_orphaned_compliance_collections()

            self.assertTrue(current_store.exists())
            self.assertFalse(stale_store.exists())
            self.assertTrue(unrelated_store.exists())

    def test_comparison_evaluation_requires_every_gold_section(self):
        golds = [
            {"jurisdiction": "ukgc", "section_id": "RTS 7A"},
            {"jurisdiction": "gli", "section_id": "3.2"},
        ]
        results = [
            {"jurisdiction": "ukgc", "section_id": "RTS 7A"},
        ]

        self.assertIsNone(first_hit_rank(results, golds))
        results.append({"jurisdiction": "gli", "section_id": "3.2"})
        self.assertEqual(first_hit_rank(results, golds), 2)

    def test_tokenizer_preserves_statutory_and_standard_identifiers(self):
        tokens = tokenize(
            "N.J.A.C. 13:69O-1.2(e)(1); UKGC RTS requirement 14A; GLI-19 section 3.2.1"
        )

        self.assertIn("13:69o1.2(e)(1)", tokens)
        self.assertIn("rts14a", tokens)
        self.assertIn("gli19", tokens)
        self.assertIn("3.2.1", tokens)
        self.assertIn("13", tokens)
        self.assertIn("69o", tokens)
        self.assertIn("definition", tokenize("How do rules define a gaming term?"))
        self.assertIn("definition", tokenize("Definitions list gaming terms."))

    def test_bm25_text_avoids_repeating_long_parent_titles(self):
        text = bm25_text({
            "section_id": "2",
            "heading_path": [
                "The Tamil Nadu Prohibition of Online Gambling and Regulation of Online Games Act, 2022",
                "Section 2",
            ],
            "text": "Online gambling means online wagering or betting.",
        })

        self.assertIn("Section 2", text)
        self.assertIn("Online gambling means", text)
        self.assertNotIn("Tamil Nadu Prohibition", text)

    def test_bm25_excludes_unsectioned_front_matter_for_sectioned_documents(self):
        chunks = [
            {
                "file": "india/tn/act.md",
                "section_id": None,
                "heading_path": ["Tamil Nadu Gaming Act"],
                "title": "Tamil Nadu Gaming Act",
                "text": "Tamil Nadu Gaming Act Arrangement of Sections.",
            },
            {
                "file": "india/tn/act.md",
                "section_id": "2",
                "heading_path": ["Tamil Nadu Gaming Act", "Section 2"],
                "title": "Tamil Nadu Gaming Act",
                "text": "Online gambling means online wagering.",
            },
        ]

        sparse_text = bm25_texts(chunks)

        self.assertEqual(sparse_text[0], "")
        self.assertIn("Online gambling", sparse_text[1])

    def test_index_fingerprint_detects_changed_chunk_content(self):
        first_chunks = [{"chunk_id": "chunk-1", "text": "old source text"}]
        changed_chunks = [{"chunk_id": "chunk-1", "text": "new source text"}]

        self.assertNotEqual(
            corpus_fingerprint(first_chunks),
            corpus_fingerprint(changed_chunks),
        )
        with self.assertRaises(ValueError):
            _assert_bm25_alignment(
                changed_chunks,
                ["chunk-1"],
                corpus_fingerprint(first_chunks),
            )

    def test_metadata_filters_are_allowlisted_and_exact(self):
        self.assertEqual(
            _validate_filters({
                "jurisdiction": "nj_dge",
                "regulation_type": "administrative_code",
            }),
            {
                "jurisdiction": "nj_dge",
                "regulation_type": "administrative_code",
            },
        )
        with self.assertRaises(ValueError):
            _validate_filters({"jurisdiction": "other"})
        with self.assertRaises(ValueError):
            _validate_filters({"file": "anything"})
        self.assertEqual(
            _validate_filters({
                "jurisdiction": "india_state_andhra_pradesh",
                "jurisdiction_type": "state",
                "regulation_type": "state_act",
                "country": "India",
            }),
            {
                "jurisdiction": "india_state_andhra_pradesh",
                "jurisdiction_type": "state",
                "regulation_type": "state_act",
                "country": "India",
            },
        )
        self.assertTrue(
            _matches_filters(
                {"jurisdiction": "ukgc", "regulation_type": "technical_standard"},
                {"jurisdiction": "ukgc"},
            )
        )
        self.assertFalse(
            _matches_filters(
                {"jurisdiction": "gli", "regulation_type": "technical_standard"},
                {"jurisdiction": "ukgc"},
            )
        )

    def test_reciprocal_rank_fusion_combines_independent_rankings(self):
        fused = _rrf([[2, 1], [1, 3]], 3)

        self.assertEqual(fused[0], 1)
        self.assertEqual(set(fused), {1, 2, 3})

    def test_multi_query_routing_detects_named_jurisdictions(self):
        routes = infer_jurisdictions(
            "Compare UKGC RTS 14A with N.J.A.C. 13:69O and GLI-19."
        )

        self.assertEqual(routes, ["ukgc", "nj_dge", "gli"])

    def test_jurisdiction_router_detects_india_and_named_states(self):
        state_routes = infer_jurisdictions(
            "Compare Andhra Pradesh and Tamil Nadu online gaming laws."
        )
        all_india_routes = infer_jurisdictions("Compare Indian state gaming laws.")

        self.assertEqual(
            state_routes,
            ["india_state_andhra_pradesh", "india_state_tamil_nadu"],
        )
        self.assertEqual(len(all_india_routes), len(INDIA_JURISDICTIONS))
        self.assertEqual(all_india_routes[0], "india_central")

    def test_explicit_citations_are_promoted_only_for_matching_metadata(self):
        chunks = [
            {
                "section_id": "3.2.1",
                "citation": "GLI-19 3.2.1",
                "jurisdiction": "gli",
            },
            {
                "section_id": "3.4.1",
                "citation": "GLI-19 3.4.1",
                "jurisdiction": "gli",
            },
            {
                "section_id": "13:69O-1.2",
                "citation": "13:69O-1.2(e)(1)",
                "jurisdiction": "nj_dge",
                "clause_paths": [["(e)", "1."]],
            },
            {
                "section_id": "RTS 7A",
                "citation": "RTS 7A",
                "jurisdiction": "ukgc",
            },
        ]

        self.assertEqual(
            _exact_reference_ids(
                "GLI-19 section 3.2.1",
                chunks,
                {"jurisdiction": "gli"},
            ),
            [0],
        )
        self.assertEqual(
            _exact_reference_ids(
                "N.J.A.C. 13:69O-1.2(e)(1)",
                chunks,
                {"jurisdiction": "nj_dge"},
            ),
            [2],
        )
        self.assertEqual(
            _exact_reference_ids(
                "N.J.A.C. 13:69O-1.2(e)(2)",
                chunks,
                {"jurisdiction": "nj_dge"},
            ),
            [],
        )
        self.assertEqual(
            _exact_reference_ids(
                "UKGC RTS requirement 7A",
                chunks,
                {"jurisdiction": "ukgc"},
            ),
            [3],
        )
        self.assertEqual(
            _exact_reference_ids(
                "India Gaming Act section 5",
                [{
                    "section_id": "5",
                    "citation": "Example Gaming Act, section 5",
                    "jurisdiction": "india_state_odisha",
                    "clause_paths": [],
                }],
                {"jurisdiction": "india_state_odisha"},
            ),
            [0],
        )
        self.assertEqual(
            _exact_reference_ids(
                "GLI-19 section 3.2.1",
                [chunks[0], {**chunks[0], "chunk_id": "part-2"}],
                {"jurisdiction": "gli"},
            ),
            [0],
        )

    def test_ukgc_table_extraction_keeps_headers_and_cells(self):
        table = BeautifulSoup(
            "<table><caption>RNG requirements</caption>"
            "<tr><th>Topic</th><th>Rule</th></tr>"
            "<tr><td>Output</td><td>Unpredictable</td></tr></table>",
            "html.parser",
        ).table

        self.assertEqual(
            _extract_table_rows(table),
            [
                "RNG requirements",
                "| Topic | Rule |",
                "| Output | Unpredictable |",
            ],
        )

    def test_india_code_text_becomes_section_and_page_markers(self):
        source = (
            "1\n"
            "Prefatory note.\n"
            "ARRANGEMENT OF SECTIONS\n"
            "1. Short title and commencement.\n"
            "2. Definitions.\n"
            "This Act is hereby enacted by the Legislature\n"
            "in this State as follows:\n"
            "1. For Statement of Objects and Reasons, see the Gazette.\n"
            "1. (1) This Act may be called the Example Gaming Act.\n"
            "2\n"
            "2. Whoever operates a common gaming house commits an offence.\n"
        )

        markdown = _format_india_code_markdown("Example Gaming Act", source)

        self.assertIn("<!-- Page 1 -->", markdown)
        self.assertIn("## Section 1", markdown)
        self.assertIn("## Section 2", markdown)
        self.assertEqual(markdown.count("## Section 1"), 1)
        self.assertNotIn("ARRANGEMENT OF SECTIONS", markdown)

    def test_india_code_table_of_contents_ends_at_repeated_section_number(self):
        source = (
            "ARRANGEMENT OF SECTIONS\n"
            "1. Short title and commencement.\n"
            "2. Definitions.\n"
            "THE EXAMPLE GAMING ACT\n"
            "1. (1) This Act may be called the Example Gaming Act.\n"
            "2. Gaming means the operation of an online game.\n"
        )

        markdown = _format_india_code_markdown("Example Gaming Act", source)

        self.assertEqual(markdown.count("## Section 1"), 1)
        self.assertIn("## Section 2", markdown)
        self.assertNotIn("ARRANGEMENT OF SECTIONS", markdown)
        self.assertIn("Gaming means the operation", markdown)
        self.assertIn("(1) This Act may be called", markdown)

    def test_india_rule_type_does_not_match_singular_act_title(self):
        self.assertEqual(
            _india_document_type("state", "The Meghalaya Regulation of Gaming Act, 2021"),
            "state_act",
        )
        self.assertEqual(
            _india_document_type("state", "The Meghalaya Regulation of Gaming Rules, 2021"),
            "state_rule",
        )

    def test_india_source_chunks_keep_country_and_law_metadata(self):
        source = (
            "# Andhra Pradesh Gaming Act\n\n"
            "## Section 2\n\n"
            "Gaming means playing a game for prizes.\n"
        )
        metadata = {
            "country": "India",
            "jurisdiction_name": "Andhra Pradesh",
            "jurisdiction_type": "state",
            "title": "The Andhra Pradesh Gaming Act",
            "source_system": "India Code",
            "handle": "123456789/1",
            "act_year": "1974",
            "document_status": "not_marked_repealed",
            "legal_effect_status": "not_verified",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "andhra-pradesh.md"
            path.write_text(source, encoding="utf-8")
            chunks = _build_compliance_chunks(
                path,
                "india/india_state_andhra_pradesh/andhra-pradesh.md",
                "india_state_andhra_pradesh",
                "state_act",
                "https://indiacode.gov.in/handle/123456789/1",
                metadata,
            )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_id"], "2")
        self.assertEqual(chunks[0]["citation"], "The Andhra Pradesh Gaming Act, section 2")
        self.assertEqual(chunks[0]["country"], "India")
        self.assertEqual(chunks[0]["jurisdiction_name"], "Andhra Pradesh")
        self.assertEqual(chunks[0]["legal_effect_status"], "not_verified")


if __name__ == "__main__":
    unittest.main()
