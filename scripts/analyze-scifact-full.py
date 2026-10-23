"""Compare saved full SciFact runs with paired query-level uncertainty."""

import json
import random
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "scifact_full_20261022"
benchmark = runpy.run_path(str(ROOT / "scripts" / "benchmark-scifact-full.py"))
_, queries, qrels = benchmark["load_dataset"]()
score = benchmark["score"]
run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
rows = {mode: score(ranking, qrels)["by_query"] for mode, ranking in run.items()}
query_ids = list(qrels)
rng = random.Random(20261022)
comparisons = {}
for baseline in ("bm25", "dense"):
    diffs = [rows["hybrid"][qid]["ndcg@10"] - rows[baseline][qid]["ndcg@10"]
             for qid in query_ids]
    samples = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
                     for _ in range(10000))
    rescued = [qid for qid in query_ids
               if not rows[baseline][qid]["hit_ranks"] and rows["hybrid"][qid]["hit_ranks"]]
    lost = [qid for qid in query_ids
            if rows[baseline][qid]["hit_ranks"] and not rows["hybrid"][qid]["hit_ranks"]]
    comparisons[f"hybrid_vs_{baseline}"] = {
        "ndcg@10_mean_difference": sum(diffs) / len(diffs),
        "paired_bootstrap_95pct_ci": [samples[249], samples[9749]],
        "query_wins": sum(x > 1e-12 for x in diffs),
        "query_losses": sum(x < -1e-12 for x in diffs),
        "query_ties": sum(abs(x) <= 1e-12 for x in diffs),
        "rescued_misses": len(rescued),
        "new_misses": len(lost),
        "rescued_query_ids": rescued,
        "lost_query_ids": lost,
    }

subset_ids = list(runpy.run_path(str(ROOT / "scripts" / "score-scifact-subset.py"))["QRELS"])
subset = {mode: {metric: sum(rows[mode][qid][metric] for qid in subset_ids) / len(subset_ids)
                 for metric in ("ndcg@10", "recall@10", "map@10")}
          for mode in rows}
result = {"comparisons": comparisons, "first_16_queries_full_corpus": subset}
(OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
