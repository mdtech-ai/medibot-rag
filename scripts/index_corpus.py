"""
Indexe dans Pinecone tous les PDF du corpus (data/raw/), en réutilisant la logique d'indexation du serveur.

Les identifiants des vecteurs sont dérivés du nom de fichier ({nom}-{numéro de chunk}).
Le script est reprenable : un document déjà présent dans l'index est sauté, ce qui permet de relancer
le script après une interruption (quota, panne réseau) sans tout recalculer.

Usage (depuis la racine du projet, après scripts/download_corpus.py) :
    python scripts/index_corpus.py            # indexe uniquement les documents absents de l'index
    python scripts/index_corpus.py --force    # réindexe tous les documents (remplace les vecteurs existants)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# Le code du serveur s'importe depuis serveur/ (imports du type "from modules... import ...")
sys.path.insert(0, str(ROOT / "serveur"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "serveur" / ".env")

from modules.load_vectorstore import index, index_pdf  # noqa: E402


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun PDF dans {RAW_DIR} : lance d'abord scripts/download_corpus.py")
        return

    force = "--force" in sys.argv
    start = time.time()
    total_chunks = 0
    indexed = 0
    for i, pdf in enumerate(pdf_files, start=1):
        # Un document est considéré comme indexé si son premier chunk existe dans l'index
        # (les vecteurs d'un document ne sont envoyés qu'une fois tous ses embeddings calculés)
        if not force and index.fetch(ids=[f"{pdf.stem}-0"]).vectors:
            print(f"\n[{i}/{len(pdf_files)}] {pdf.name} : déjà indexé, ignoré (--force pour réindexer)")
            continue
        print(f"\n[{i}/{len(pdf_files)}] {pdf.name}")
        total_chunks += index_pdf(str(pdf))
        indexed += 1

    print(f"\n{indexed} documents, {total_chunks} chunks indexés en {time.time() - start:.0f} s")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
