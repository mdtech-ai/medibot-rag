"""
Reranking : reclassement des passages récupérés par un modèle cross-encoder.

La recherche vectorielle (bi-encoder, modèle E5) compare deux vecteurs calculés séparément : celui de la question
et celui du passage. C'est rapide (les passages sont vectorisés une seule fois, à l'indexation), mais approximatif.
Un cross-encoder lit la question ET le passage ensemble, dans un même appel au modèle : il juge bien plus finement
si le passage répond à la question, mais il est trop lent pour comparer la question à tout le corpus.

D'où la recherche en deux temps :
1. Pinecone renvoie les FETCH_K passages les plus proches (rapide, large) ;
2. le cross-encoder les reclasse et on garde les top_k meilleurs (précis, sur peu de passages).

Modèle : BAAI/bge-reranker-v2-m3, multilingue (dont le français), open source, exécuté localement.
Désactivable avec RERANKER_MODEL=none dans .env (pour comparer avec et sans reranking).
"""
import os

from logger import logger

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
FETCH_K = int(os.getenv("RERANKER_FETCH_K", "20"))  # Nombre de passages demandés à Pinecone avant reclassement


def get_reranker():
    """Charge le cross-encoder (sur la carte graphique si disponible), ou renvoie None si le reranking est désactivé."""
    if RERANKER_MODEL.lower() in ("", "none"):
        logger.info("Reranking désactivé (RERANKER_MODEL=none)")
        return None

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Chargement du reranker {RERANKER_MODEL} ({device})")
    # Pleine précision (float32) : mesuré sur Quadro T2000, 0,39 s pour 20 paires contre 1,59 s en float16
    # (demi-précision mal accélérée sur cette carte), pour des scores identiques ; mémoire : 2,3 Go contre 1,2 Go
    return CrossEncoder(
        RERANKER_MODEL,
        device=device,
        max_length=512,  # Question + passage : les passages font au plus 500 caractères, largement sous 512 tokens
    )


# Chargé une seule fois au démarrage, comme le modèle d'embedding
reranker = get_reranker()
