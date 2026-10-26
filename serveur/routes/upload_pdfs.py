from fastapi import APIRouter, UploadFile, File
from typing import List
from modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger

router = APIRouter() # Initialisation du routeur FastAPI pour gérer les routes liées au téléchargement de fichiers PDF

@router.post("/upload_pdfs")
# def (et non async def) : l'indexation est bloquante (lecture des PDF, embeddings, upsert).
# FastAPI exécute alors la route dans un thread séparé, et le serveur continue de répondre aux autres requêtes.
def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Endpoint pour le téléchargement de fichiers PDF.
    Cette route permet aux utilisateurs de soumettre des documents PDF à l'application pour traitement.
    Le fichier sera reçu via une requête POST, et le serveur pourra ensuite extraire le texte du PDF pour l'analyse.
    """

    try:
        logger.info(f"Réception de {len(files)} fichiers PDF pour téléchargement.")
        # Appel de la fonction load_vectorstore pour traiter les fichiers PDF téléchargés
        load_vectorstore(files)
        logger.info("Documents ajoutés à la base de données vectorielle avec succès.")
        return JSONResponse(content={"message": "Fichiers traités et ajoutés à la base de données vectorielle avec succès."}, status_code=200)
    except Exception as e:
        logger.exception(f"Erreur lors du téléchargement des fichiers PDF : {e}")
        return JSONResponse(content={"error": "Erreur lors du téléchargement des fichiers PDF."}, status_code=500)