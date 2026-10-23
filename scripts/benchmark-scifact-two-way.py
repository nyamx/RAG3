"""Score semantic + BM25 RRF on the fixed 16-query SciFact smoke subset."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.retrieval import search

COLLECTION = "rag3_beir_scifact_subset_20261021"
QUERIES = {
    "1": "0-dimensional biomaterials show inductive properties.",
    "3": "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.",
    "5": "1/2000 in UK have abnormal PrP positivity.",
    "13": "5% of perinatal mortality is due to low birth weight.",
    "36": "A deficiency of vitamin B12 increases blood levels of homocysteine.",
    "42": "A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.",
    "48": "A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.",
    "49": "ADAR1 binds to Dicer to cleave pre-miRNA.",
    "50": "AIRE is expressed in some skin tumors.",
    "51": "ALDH1 expression is associated with better breast cancer outcomes.",
    "53": "ALDH1 expression is associated with poorer prognosis in breast cancer.",
    "54": "AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs.",
    "56": "APOE4 expression in iPSC-derived neurons increases AlphaBeta production and tau phosphorylation causing GABA neuron degeneration.",
    "57": "APOE4 expression in iPSC-derived neurons increases AlphaBeta production and tau phosphorylation, delaying GABA neuron degeneration.",
    "70": "Activation of PPM1D suppresses p53 function.",
    "72": "Activator-inhibitor pairs are provided dorsally by Admpchordin.",
}


def main() -> None:
    score_method = runpy.run_path(str(ROOT / "scripts" / "score-scifact-subset.py"))["score_method"]
    ranked = {}
    durations = {}
    for qid, query in QUERIES.items():
        started = time.perf_counter()
        result = search(query, collection=COLLECTION, semantic=True, bm25=True,
                        rank_fusion=True, top_k=10, candidate_k=30)
        durations[qid] = time.perf_counter() - started
        ranked[qid] = [str(hit["title"]).removeprefix("BEIR SciFact ")
                       for hit in result["hits"]]
        print(f"query {qid}: {durations[qid]:.3f}s", flush=True)
    score = score_method(ranked)
    path = ROOT / "artifacts" / "analysis" / "scifact_subset_two_way_20261021_run.json"
    path.write_text(json.dumps({"semantic_bm25": ranked}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: score[key] for key in ("ndcg@10", "recall@10", "map@10")},
                     indent=2))
    print(f"run={path}")


if __name__ == "__main__":
    main()
