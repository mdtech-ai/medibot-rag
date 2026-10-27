# Les bibliothèques nécessaires
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain, stream_chain
from modules.load_vectorstore import index, embed_model
from modules.retriever import PineconeRetriever
from modules.observability import tracing_config
from schemas import AskRequest, AskResponse
from logger import logger

router = APIRouter()  # Initialisation du routeur FastAPI pour gérer les routes liées aux questions posées par les utilisateurs

# Construits une seule fois au démarrage du serveur, puis réutilisés pour chaque question
# (avant : connexion Pinecone, modèle d'embedding et chaîne recréés à chaque requête, ce qui ajoutait de la latence)
retriever = PineconeRetriever(index=index, embed_model=embed_model, top_k=3)
chain = get_llm_chain(retriever)

@router.post("/ask/", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        logger.info(f"Requête de l'utilisateur : {request.question}")
        history = [message.model_dump() for message in request.history]
        result = await query_chain(chain, request.question, history, tracing_config("ask", request.session_id))

        logger.info("Requête traitée avec succès")
        return result

    except Exception:
        logger.exception("Erreur du traitement de la question")
        # Message générique pour le client : le détail technique reste dans les journaux (pas de fuite d'informations internes)
        return JSONResponse(status_code=500, content={"error": "Erreur lors du traitement de la question."})


@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """
    Version streaming de /ask/ : la réponse est envoyée au fil de la génération (format Server-Sent Events).
    Événements : "sources" (citations), "token" (morceau de texte), "done" (fin) ou "error".
    """
    logger.info(f"Requête de l'utilisateur (streaming) : {request.question}")
    return StreamingResponse(
        stream_chain(
            chain,
            request.question,
            [message.model_dump() for message in request.history],
            tracing_config("ask-stream", request.session_id),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # Le navigateur ou un proxy ne doit pas mettre le flux en cache
            "X-Accel-Buffering": "no",  # Désactive la mise en tampon des proxys (ex. Nginx) qui retarderait les tokens
        },
    )
