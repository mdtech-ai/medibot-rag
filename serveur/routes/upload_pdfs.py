import os
import secrets
from typing import List

from fastapi import APIRouter, File, Header, UploadFile
from fastapi.responses import JSONResponse
from logger import logger
from modules.load_vectorstore import load_vectorstore

router = APIRouter() # Initialisation du routeur FastAPI pour gérer les routes liées au téléchargement de fichiers PDF

# Jeton d'administration : l'ajout de documents modifie l'index partagé par tous les utilisateurs.
# Sans protection, n'importe qui pourrait indexer des documents trompeurs ou épuiser les quotas sur une démo publique.
# Route désactivée si ADMIN_TOKEN n'est pas défini.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


@router.post("/upload_pdfs")
# def (et non async def) : l'indexation est bloquante (lecture des PDF, embeddings, upsert).
# FastAPI exécute alors la route dans un thread séparé, et le serveur continue de répondre aux autres requêtes.
def upload_pdfs(files: List[UploadFile] = File(...), x_admin_token: str = Header("", description="Jeton d'administration")):
    """
    Ajoute des documents PDF au corpus (réservé à l'administrateur : en-tête X-Admin-Token).
    Les fichiers sont enregistrés, nettoyés, découpés, vectorisés puis indexés dans Pinecone.
    """
    # compare_digest : comparaison en temps constant (ne révèle pas, par la durée de la réponse, combien de caractères sont justes)
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        logger.warning("Tentative d'ajout de documents refusée (jeton d'administration absent ou invalide)")
        return JSONResponse(content={"error": "Ajout de documents non autorisé."}, status_code=403)

    try:
        logger.info(f"Réception de {len(files)} fichiers PDF pour téléchargement.")
        # Appel de la fonction load_vectorstore pour traiter les fichiers PDF téléchargés
        load_vectorstore(files)
        logger.info("Documents ajoutés à la base de données vectorielle avec succès.")
        return JSONResponse(content={"message": "Fichiers traités et ajoutés à la base de données vectorielle avec succès."}, status_code=200)
    except Exception as e:
        logger.exception(f"Erreur lors du téléchargement des fichiers PDF : {e}")
        return JSONResponse(content={"error": "Erreur lors du téléchargement des fichiers PDF."}, status_code=500)
