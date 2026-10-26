"""
Évaluation du retriever sur le jeu de test (data/eval/testset.jsonl), sans appel au LLM de réponse.

Pour chaque cas qui attend une réponse, on vérifie si les passages récupérés dans Pinecone
proviennent des pages où se trouve la réponse de référence :
- hit@k (taux de réussite) : proportion de cas où au moins une page attendue figure dans les k premiers résultats
- MRR@k (Mean Reciprocal Rank) : moyenne de 1/rang du premier bon résultat (1 si en tête, 0,5 si 2e... 0 si absent)
- doc_hit@k : même chose au niveau du document (le bon PDF, quelle que soit la page)

Les questions de suivi sont d'abord reformulées en question autonome, comme dans l'API
(seul appel LLM du script, un par question de suivi).

Usage (depuis la racine du projet) :
    python scripts/eval_retrieval.py              # k = 3, 5 et 10
    python scripts/eval_retrieval.py --k 3 5      # valeurs de k au choix
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "eval" / "testset.jsonl"
RESULTS_DIR = ROOT / "data" / "eval" / "results"

# Le code du serveur s'importe depuis serveur/ (imports du type "from modules... import ...")
sys.path.insert(0, str(ROOT / "serveur"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "serveur" / ".env")

from modules.embeddings import EMBEDDING_PROVIDER  # noqa: E402
from modules.llm import get_llm_chain  # noqa: E402
from modules.load_vectorstore import PINECONE_INDEX_NAME, embed_model, index  # noqa: E402
from modules.query_handlers import format_sources  # noqa: E402
from modules.retriever import PineconeRetriever  # noqa: E402


def evaluate_case(retrieved: list[dict], expected: list[dict], k: int) -> dict:
    """Calcule hit, reciprocal rank et doc_hit pour un cas, sur les k premiers résultats."""
    expected_pages = {(s["file"], s["page"]) for s in expected}
    expected_files = {s["file"] for s in expected}
    top = retrieved[:k]

    rank = next((i for i, r in enumerate(top, start=1) if (r["file"], r["page"]) in expected_pages), None)
    return {
        "hit": rank is not None,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "doc_hit": any(r["file"] in expected_files for r in top),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10], help="Nombres de résultats évalués")
    args = parser.parse_args()
    max_k = max(args.k)

    cases = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Seuls les cas qui attendent une réponse et ont des sources de référence évaluent le retriever
    cases = [c for c in cases if c["comportement_attendu"] == "repondre" and c["sources"]]

    retriever = PineconeRetriever(index=index, embed_model=embed_model, top_k=max_k)
    # Première étape de la chaîne de l'API : reformulation de la question (sautée s'il n'y a pas d'historique)
    standalone_step = get_llm_chain(retriever).first

    details = []
    start = time.time()
    for case in cases:
        standalone = standalone_step.invoke({"question": case["question"], "history": case["history"]})["standalone_question"]
        # format_sources : même numérotation des pages que les citations de l'API
        # (on retire seulement le dédoublonnage par page en gardant l'ordre des résultats)
        docs = retriever.invoke(standalone)
        retrieved = [{"file": s["file"], "page": s["page"], "score": s["score"]} for d in docs for s in format_sources([d])]

        details.append({
            "id": case["id"],
            "categorie": case["categorie"],
            "question": case["question"],
            "standalone_question": standalone,
            "expected": case["sources"],
            "retrieved": retrieved,
            "metrics": {f"k={k}": evaluate_case(retrieved, case["sources"], k) for k in args.k},
        })

    # Agrégation : moyenne globale et par catégorie
    def aggregate(rows, k):
        n = len(rows)
        return {
            "n": n,
            "hit": round(sum(r["metrics"][f"k={k}"]["hit"] for r in rows) / n, 3),
            "mrr": round(sum(r["metrics"][f"k={k}"]["reciprocal_rank"] for r in rows) / n, 3),
            "doc_hit": round(sum(r["metrics"][f"k={k}"]["doc_hit"] for r in rows) / n, 3),
        }

    categories = sorted({d["categorie"] for d in details})
    summary = {
        f"k={k}": {
            "global": aggregate(details, k),
            "par_categorie": {cat: aggregate([d for d in details if d["categorie"] == cat], k) for cat in categories},
        }
        for k in args.k
    }

    # Affichage
    print(f"\nÉvaluation du retriever — {len(details)} cas — embeddings : {EMBEDDING_PROVIDER} — index : {PINECONE_INDEX_NAME}")
    print(f"{'k':>4} | {'hit@k':>6} | {'MRR@k':>6} | {'doc_hit@k':>9}")
    for k in args.k:
        g = summary[f"k={k}"]["global"]
        print(f"{k:>4} | {g['hit']:>6.1%} | {g['mrr']:>6.3f} | {g['doc_hit']:>9.1%}")

    k_ref = args.k[0]
    print(f"\nPar catégorie (k={k_ref}) :")
    for cat, m in summary[f"k={k_ref}"]["par_categorie"].items():
        print(f"  {cat:<16} n={m['n']:<3} hit={m['hit']:.1%}  MRR={m['mrr']:.3f}  doc_hit={m['doc_hit']:.1%}")

    misses = [d for d in details if not d["metrics"][f"k={max_k}"]["hit"]]
    print(f"\nCas sans la bonne page dans les {max_k} premiers résultats ({len(misses)}) :")
    for d in misses:
        print(f"  - {d['id']} : attendu {[(s['file'], s['page']) for s in d['expected']]}")
        print(f"    obtenu  {[(r['file'], r['page']) for r in d['retrieved'][:3]]}...")

    # Sauvegarde des résultats détaillés (versionnés, pour comparer les expériences dans le temps)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "embedding_provider": EMBEDDING_PROVIDER,
            "index": PINECONE_INDEX_NAME,
            "chunk_size": 500,
            "chunk_overlap": 100,
            "k": args.k,
        },
        "duree_s": round(time.time() - start, 1),
        "summary": summary,
        "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats détaillés : {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
