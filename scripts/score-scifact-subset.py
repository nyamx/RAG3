"""Score the fixed 16-query BEIR SciFact subset used in the RAG3 smoke benchmark.

Input JSON maps each method name to a mapping of query ID -> ranked corpus IDs.
The qrels come from the official SciFact test.tsv. This subset is not the full
5,183-document / 300-query BEIR benchmark.
"""

import argparse
import json
import math
from pathlib import Path


QRELS = {
    "1": ["31715818"],
    "3": ["14717500"],
    "5": ["13734012"],
    "13": ["1606628"],
    "36": ["5152028", "11705328"],
    "42": ["18174210"],
    "48": ["13734012"],
    "49": ["5953485"],
    "50": ["12580014"],
    "51": ["45638119"],
    "53": ["45638119"],
    "54": ["49556906"],
    "56": ["4709641"],
    "57": ["4709641"],
    "70": ["5956380", "4414547"],
    "72": ["6076903"],
}


def score_query(ranked, relevant):
    relevant = set(relevant)
    ranked = ranked[:10]
    if len(ranked) != len(set(ranked)):
        raise ValueError("A document appears more than once in the top 10")
    hit_ranks = [rank for rank, doc_id in enumerate(ranked, 1) if doc_id in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in hit_ranks)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    average_precision = sum(hit_number / rank for hit_number, rank in enumerate(hit_ranks, 1)) / len(relevant)
    return {
        "hit_ranks": hit_ranks,
        "ndcg@10": dcg / idcg,
        "recall@10": len(hit_ranks) / len(relevant),
        "map@10": average_precision,
    }


def score_method(run):
    if set(run) != set(QRELS):
        raise ValueError(f"Expected query IDs {sorted(QRELS)}, received {sorted(run)}")
    by_query = {qid: score_query([str(doc) for doc in run[qid]], relevant) for qid, relevant in QRELS.items()}
    count = len(by_query)
    return {
        "query_count": count,
        "ndcg@10": sum(row["ndcg@10"] for row in by_query.values()) / count,
        "recall@10": sum(row["recall@10"] for row in by_query.values()) / count,
        "map@10": sum(row["map@10"] for row in by_query.values()) / count,
        "by_query": by_query,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="JSON mapping method -> query ID -> ranked corpus IDs")
    args = parser.parse_args()
    data = json.loads(args.run.read_text(encoding="utf-8"))
    print(json.dumps({method: score_method(run) for method, run in data.items()}, indent=2))


if __name__ == "__main__":
    main()
