import json
import re
from logger import logger

EXCERPT_LENGTH = 200  # Nombre de caractères du passage renvoyés au client


def format_sources(docs) -> list[dict]:
    """
    Transforme les documents récupérés en citations lisibles : nom du fichier, page, extrait et score.
    Une seule citation par couple (fichier, page) : les documents arrivent triés par score décroissant,
    on garde donc le passage le plus pertinent de chaque page.
    """
    sources = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata
        # Garde uniquement le nom du fichier, que le chemin stocké soit Windows (\) ou Unix (/)
        file = re.split(r"[\\/]", metadata.get("source", ""))[-1]

        # page_label = numéro affiché dans le PDF ; à défaut, page (qui commence à 0) + 1
        page_label = str(metadata.get("page_label", ""))
        if page_label.isdigit():
            page = int(page_label)
        elif metadata.get("page") is not None:
            page = int(metadata["page"]) + 1
        else:
            page = None

        if (file, page) in seen:
            continue
        seen.add((file, page))

        text = " ".join(doc.page_content.split())  # Normalise les espaces et retours à la ligne
        excerpt = text if len(text) <= EXCERPT_LENGTH else text[:EXCERPT_LENGTH].rstrip() + "…"
        score = metadata.get("score")

        sources.append({
            "file": file,
            "page": page,
            "excerpt": excerpt,
            "score": round(score, 3) if score is not None else None,
        })
    return sources


async def query_chain(chain, question: str, history: list[dict] | None = None, config: dict | None = None):
    try:
        logger.debug(f"Exécution de la chaîne pour la requête utilisateur : {question} (historique : {len(history or [])} messages)")
        # ainvoke (version asynchrone de invoke) : le serveur reste disponible pour les autres requêtes pendant l'appel au LLM
        result = await chain.ainvoke({"question": question, "history": history or []}, config=config)
        response = {
            "response": result["answer"],
            "standalone_question": result["standalone_question"],
            "sources": format_sources(result["source_documents"]),
        }
        logger.debug(f"Résultat de la chaîne : {response}")
        return response
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution de la chaîne LLM : {e}")
        raise  # Relance l'exception pour que la route renvoie une erreur 500 au lieu de None


def sse_event(event: str, data) -> str:
    """Formate un événement Server-Sent Events : une ligne "event", une ligne "data" (JSON), puis une ligne vide."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_chain(chain, question: str, history: list[dict] | None = None, config: dict | None = None):
    """
    Exécute la chaîne en streaming et produit des événements SSE :
    - "sources" : les citations, dès que la recherche dans Pinecone est terminée (avant la génération)
    - "token"   : chaque morceau de texte généré par le LLM
    - "done"    : fin de la réponse
    - "error"   : en cas d'erreur (le code HTTP 200 est déjà parti, l'erreur est donc signalée dans le flux)
    """
    try:
        logger.debug(f"Exécution de la chaîne en streaming pour la requête utilisateur : {question} (historique : {len(history or [])} messages)")
        # astream renvoie la sortie de la chaîne par morceaux : d'abord les clés intermédiaires (question, standalone_question,
        # source_documents...), puis {"answer": "..."} pour chaque token, grâce au StrOutputParser qui transmet le texte au fil de l'eau
        sources_sent = False
        async for chunk in chain.astream({"question": question, "history": history or []}, config=config):
            if "source_documents" in chunk and not sources_sent:
                sources_sent = True
                yield sse_event("sources", format_sources(chunk["source_documents"]))
            if chunk.get("answer"):
                yield sse_event("token", {"text": chunk["answer"]})
        yield sse_event("done", {})
    except Exception as e:
        logger.exception(f"Erreur lors du streaming de la chaîne LLM : {e}")
        yield sse_event("error", {"error": "Erreur lors du traitement de la question."})
