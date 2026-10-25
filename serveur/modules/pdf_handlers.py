import shutil
from pathlib import Path
from fastapi import UploadFile

# Répertoire pour stocker les fichiers PDF téléchargés : serveur/uploaded_docs (indépendant du dossier depuis lequel le serveur est lancé)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploaded_docs"

def save_uploaded_files(files: list[UploadFile]) -> list[str]:
    UPLOAD_DIR.mkdir(exist_ok=True)  # Crée le répertoire s'il n'existe pas déjà
    file_paths = []  # Liste pour stocker les chemins des fichiers PDF téléchargés
    for file in files:
        save_path = UPLOAD_DIR / Path(file.filename).name  # .name : ignore un éventuel chemin dans le nom envoyé par le client
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)  # Copie le contenu du fichier téléchargé par blocs (sans tout charger en mémoire)
        file_paths.append(str(save_path))  # Ajoute le chemin du fichier sauvegardé à la liste

    return file_paths  # Retourne la liste des chemins des fichiers PDF sauvegardés
