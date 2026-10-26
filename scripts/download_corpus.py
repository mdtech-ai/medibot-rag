"""
Télécharge les documents du corpus listés dans data/sources.csv vers data/raw/.

Les PDF ne sont pas versionnés dans git (droits d'auteur des organismes, taille du dépôt) :
ce script permet de reconstruire le corpus à l'identique.

Usage (depuis la racine du projet) :
    python scripts/download_corpus.py
"""
import csv
from pathlib import Path

import requests
import truststore

# Vérification des certificats SSL par le magasin de certificats du système (Windows, macOS...) plutôt que par celui de Python.
# Nécessaire pour has-sante.fr, dont le serveur n'envoie pas sa chaîne de certificats complète : le système sait
# récupérer le certificat intermédiaire manquant, Python non. La vérification reste active (pas de verify=False).
truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = ROOT / "data" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"


def download(url: str, destination: Path) -> int:
    """Télécharge un fichier par blocs et vérifie qu'il s'agit bien d'un PDF. Renvoie sa taille en octets."""
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0 (medibot-rag)"}) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type:
            raise ValueError(f"Type inattendu : {content_type}")
        tmp = destination.with_suffix(".part")  # Fichier temporaire : un téléchargement interrompu ne laisse pas de PDF corrompu
        with open(tmp, "wb") as f:
            for block in response.iter_content(chunk_size=64 * 1024):
                f.write(block)
        tmp.replace(destination)
    return destination.stat().st_size


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_FILE, encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    errors = 0
    for source in sources:
        destination = RAW_DIR / source["fichier"]
        if destination.exists():
            print(f"[déjà présent] {destination.name}")
            continue
        try:
            size = download(source["url"], destination)
            print(f"[téléchargé]   {destination.name} ({size / 1_000_000:.1f} Mo)")
        except Exception as e:
            errors += 1
            print(f"[ERREUR]       {destination.name} : {e}")

    print(f"\n{len(sources) - errors}/{len(sources)} documents disponibles dans {RAW_DIR}")


if __name__ == "__main__":
    main()
