"""
Observabilité avec Langfuse : chaque question posée à MediBot produit une « trace » consultable dans l'interface Langfuse.

Une trace contient toutes les étapes de la chaîne LangChain, avec leurs entrées, sorties et durées :
reformulation de la question, recherche dans Pinecone (passages et scores), génération (prompt complet,
réponse, tokens consommés). On peut ainsi comprendre une mauvaise réponse a posteriori : échec de la
recherche ? passage mal utilisé ? reformulation ratée ?

Langfuse est facultatif : sans les clés LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY, le traçage est
simplement désactivé et l'API fonctionne normalement.
"""
import os

from logger import logger

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if LANGFUSE_ENABLED:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    logger.info(f"Observabilité Langfuse activée ({os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')})")
else:
    logger.info("Observabilité Langfuse désactivée (clés LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY absentes)")


def tracing_config(run_name: str, session_id: str | None = None, tags: list[str] | None = None) -> dict:
    """
    Configuration LangChain à passer à invoke/ainvoke/astream (argument config) pour tracer l'exécution.

    - run_name   : nom de la trace dans Langfuse (ex. « ask », « ask-stream »)
    - session_id : identifiant de la conversation, fourni par le client : Langfuse regroupe alors
                   toutes les questions d'une même conversation
    - tags       : étiquettes pour filtrer les traces (ex. « evaluation » pour les scripts d'évaluation)
    """
    config = {"run_name": run_name}
    if not LANGFUSE_ENABLED:
        return config

    metadata = {"langfuse_tags": tags or []}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    # Un handler par requête : il est léger, et l'état d'une trace n'est jamais partagé entre deux requêtes simultanées
    config["callbacks"] = [CallbackHandler()]
    config["metadata"] = metadata
    return config


def shutdown_tracing():
    """Envoie les traces encore en mémoire avant l'arrêt du serveur (elles sont envoyées par lots, en arrière-plan)."""
    if LANGFUSE_ENABLED:
        get_client().shutdown()
