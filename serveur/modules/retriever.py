# Retriever Pinecone : transforme la question en vecteur et récupère les chunks les plus proches dans l'index
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class PineconeRetriever(BaseRetriever):
    # BaseRetriever est un modèle Pydantic : les attributs doivent être déclarés comme champs
    index: Any  # Index Pinecone (pinecone.Index)
    embed_model: Embeddings  # Même modèle d'embedding que celui utilisé pour l'indexation
    top_k: int = 3  # Nombre de chunks renvoyés

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_vector = self.embed_model.embed_query(query)  # Vectorisation de la question
        res = self.index.query(vector=query_vector, top_k=self.top_k, include_metadata=True)

        # Le texte du chunk est stocké dans les métadonnées sous la clé "text" (voir load_vectorstore.py)
        return [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata={**match["metadata"], "score": match["score"]},  # Score de similarité utile pour le débogage
            )
            for match in res["matches"]
        ]
