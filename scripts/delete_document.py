"""
Supprime de l'index Pinecone tous les vecteurs d'un document.

Les identifiants des vecteurs ont la forme {nom du fichier sans extension}-{numéro de chunk} :
on retrouve donc tous les chunks d'un document grâce à ce préfixe.

Usage (depuis la racine du projet) :
    python scripts/delete_document.py DIABETES
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Le code du serveur s'importe depuis serveur/ (imports du type "from modules... import ...")
sys.path.insert(0, str(ROOT / "serveur"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "serveur" / ".env")

from modules.load_vectorstore import index  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage : python scripts/delete_document.py <nom du fichier sans extension>")
        sys.exit(1)

    prefix = f"{sys.argv[1]}-"  # Le tiret évite de supprimer aussi un document dont le nom commencerait pareil
    ids = [vector_id for page in index.list(prefix=prefix) for vector_id in page]
    if not ids:
        print(f"Aucun vecteur trouvé avec le préfixe « {prefix} »")
        return

    print(f"{len(ids)} vecteurs trouvés (ex. : {', '.join(ids[:3])}...)")
    if input("Confirmer la suppression ? (o/n) ").strip().lower() != "o":
        print("Suppression annulée")
        return

    for start in range(0, len(ids), 1000):  # Pinecone accepte au plus 1000 identifiants par requête de suppression
        index.delete(ids=ids[start:start + 1000])
    print(f"{len(ids)} vecteurs supprimés")


if __name__ == "__main__":
    main()
