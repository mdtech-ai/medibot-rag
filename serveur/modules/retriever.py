# Retriever Pinecone : transforme la question en vecteur et récupère les chunks les plus proches dans l'index,
# puis (si un reranker est fourni) les reclasse avec un cross-encoder (voir modules/reranker.py)
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class PineconeRetriever(BaseRetriever):
    # BaseRetriever est un modèle Pydantic : les attributs doivent être déclarés comme champs
    index: Any  # Index Pinecone (pinecone.Index)
    embed_model: Embeddings  # Même modèle d'embedding que celui utilisé pour l'indexation
    top_k: int = 3  # Nombre de chunks renvoyés
    reranker: Any = None  # Cross-encoder (sentence_transformers.CrossEncoder) ; None = pas de reclassement
    fetch_k: int = 20  # Nombre de chunks demandés à Pinecone avant reclassement (ignoré sans reranker)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_vector = self.embed_model.embed_query(query)  # Vectorisation de la question
        k = max(self.fetch_k, self.top_k) if self.reranker else self.top_k
        res = self.index.query(vector=query_vector, top_k=k, include_metadata=True)

        # Le texte du chunk est stocké dans les métadonnées sous la clé "text" (voir load_vectorstore.py)
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata={**match["metadata"], "score": match["score"]},  # Score de similarité utile pour le débogage
            )
            for match in res["matches"]
        ]
        if not self.reranker or not docs:
            return docs

        # Reclassement : le cross-encoder note chaque paire (question, passage), titre du document compris
        pairs = [(query, f"{doc.metadata.get('title', '')}\n{doc.page_content}".strip()) for doc in docs]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.metadata["vector_score"] = doc.metadata["score"]  # Score de la recherche vectorielle, conservé pour comparaison
            doc.metadata["score"] = float(score)  # Score du reranker (0 à 1) : c'est lui qui est affiché dans les citations
        docs.sort(key=lambda doc: doc.metadata["score"], reverse=True)
        return docs[:self.top_k]
