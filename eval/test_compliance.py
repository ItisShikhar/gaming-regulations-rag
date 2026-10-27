import re
import sys
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from download_corpus import _extract_table_rows
from index import corpus_fingerprint, tokenize
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


if __name__ == "__main__":
    unittest.main()
