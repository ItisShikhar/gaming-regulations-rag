import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from index import load_chunks
from search import close_search_client, compliance_search

QUESTIONS = Path("eval/compliance_questions.jsonl")
K = 5
FETCH = 20
RETRIEVERS = ("pre", "post", "multi_query")


def load_questions():
    with QUESTIONS.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def is_hit(chunk, gold):
    if chunk.get("jurisdiction") != gold["jurisdiction"]:
        return False
    if chunk.get("section_id") != gold["section_id"]:
        return False
    gold_clause_path = gold.get("clause_path")
    return not gold_clause_path or gold_clause_path in chunk.get("clause_paths", [])


def unique_sections(chunks, k):
    seen, output = set(), []
    for chunk in chunks:
        key = (
            chunk["file"],
            chunk.get("section_id"),
            tuple(chunk.get("clause_path", [])),
        )
        if key not in seen:
            seen.add(key)
            output.append(chunk)
    return output[:k]


def first_hit_rank(results, golds):
    gold_ranks = []
    for gold in golds:
        rank = next(
            (
                rank
                for rank, chunk in enumerate(results, start=1)
                if is_hit(chunk, gold)
            ),
            None,
        )
        if rank is None:
            return None
        gold_ranks.append(rank)
    return max(gold_ranks) if gold_ranks else None


def check_gold_labels(questions, chunks):
    for question in questions:
        for gold in question["gold"]:
            if not any(is_hit(chunk, gold) for chunk in chunks):
                print(
                    f"WARNING: question {question['id']} gold label "
                    f"matches no chunk: {gold}"
                )


def main():
    questions = load_questions()
    if not questions:
        raise ValueError(f"No evaluation questions found in {QUESTIONS}.")
    chunks = load_chunks()
    check_gold_labels(questions, chunks)
    answerable = [question for question in questions if question["gold"]]
    if not answerable:
        raise ValueError("The evaluation set has no answerable questions.")
    print(
        f"compliance: {len(questions)} questions, "
        f"{len(answerable)} answerable, Recall@{K}\n"
    )

    stats = {
        mode: {"rr": [], "hit": [], "by_type": defaultdict(list)}
        for mode in RETRIEVERS
    }
    misses = defaultdict(list)

    for question in answerable:
        for mode in RETRIEVERS:
            results = unique_sections(
                compliance_search(
                    question["question"],
                    FETCH,
                    question.get("filters"),
                    mode,
                ),
                K,
            )
            rank = first_hit_rank(results, question["gold"])
            stats[mode]["rr"].append(1 / rank if rank else 0.0)
            stats[mode]["hit"].append(1 if rank else 0)
            stats[mode]["by_type"][question["type"]].append(1 if rank else 0)
            if rank is None:
                misses[mode].append(question["id"])

    print(f"{'retriever':<14}{'Recall@' + str(K):>10}{'MRR':>8}")
    for mode, result in stats.items():
        recall = sum(result["hit"]) / len(result["hit"])
        mean_reciprocal_rank = sum(result["rr"]) / len(result["rr"])
        print(f"{mode:<14}{recall:>10.3f}{mean_reciprocal_rank:>8.3f}")

    print("\nRecall by question type")
    types = sorted({question["type"] for question in answerable})
    print(f"{'retriever':<14}" + "".join(f"{kind:>14}" for kind in types))
    for mode, result in stats.items():
        row = ""
        for kind in types:
            hits = result["by_type"][kind]
            row += (
                f"{sum(hits) / len(hits):>14.2f}"
                if hits
                else f"{'-':>14}"
            )
        print(f"{mode:<14}{row}")

    print("\nMissed question ids")
    for mode in RETRIEVERS:
        print(f"{mode:<14}{misses[mode]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_search_client()
