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


async def query_chain(chain, user_input:str):
    try:
        logger.debug(f"Exécution de la chaîne pour la requête utilisateur : {user_input}")
        # ainvoke (version asynchrone de invoke) : le serveur reste disponible pour les autres requêtes pendant l'appel au LLM
        result = await chain.ainvoke(user_input)
        response = {
            "response": result["answer"],
            "sources": format_sources(result["source_documents"]),
        }
        logger.debug(f"Résultat de la chaîne : {response}")
        return response
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution de la chaîne LLM : {e}")
        raise  # Relance l'exception pour que la route renvoie une erreur 500 au lieu de None
