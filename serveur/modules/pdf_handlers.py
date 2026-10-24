import os 
import shutil 
from fastapi import UploadFile
import tempfile  # tempfile pour sauvegarder temporairement les fichiers PDF téléchargés dans un espace de stockage temporaire avant de les traiter.

UPLOAD_DIR = "./uploaded_docs"  # Répertoire pour stocker les fichiers PDF téléchargés

def save_uploaded_files(files:list(UploadFile)) -> list[str]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)  # Crée le répertoire s'il n'existe pas déjà
    file_paths = []  # Liste pour stocker les chemins des fichiers PDF téléchargés
    for file in files:
            temp_path = os.path.join(UPLOAD_DIR, file.filename)  # Chemin complet pour sauvegarder le fichier
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)  # Copie le contenu du fichier téléchargé dans le fichier temporaire
            file_paths.append(temp_path)  # Ajoute le chemin du fichier sauvegardé à la liste
    
    return file_paths  # Retourne la liste des chemins des fichiers PDF sauvegardés
