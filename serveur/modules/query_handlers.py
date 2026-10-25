from logger import logger

async def query_chain(chain, user_input:str):
    try:
        logger.debug(f"Exécution de la chaîne pour la requête utilisateur : {user_input}")
        # ainvoke (version asynchrone de invoke) : le serveur reste disponible pour les autres requêtes pendant l'appel au LLM
        result = await chain.ainvoke(user_input)
        response = {
            "response": result["answer"],
            "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
        }
        logger.debug(f"Résultat de la chaîne : {response}")
        return response
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution de la chaîne LLM : {e}")
        raise  # Relance l'exception pour que la route renvoie une erreur 500 au lieu de None
