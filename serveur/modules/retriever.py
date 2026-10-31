# Retriever Pinecone : transforme la question en vecteur et récupère les chunks les plus proches dans l'index,
# puis (si un reranker est fourni) les reclasse avec un cross-encoder (voir modules/reranker.py),
# puis (si window > 0) élargit chaque chunk retenu à ses voisins immédiats dans le document
import os
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

# Nombre de chunks voisins ajoutés de chaque côté des chunks retenus (API et évaluation des réponses)
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1"))


def merge_overlapping(texts: list[str]) -> str:
    """
    Recolle des chunks consécutifs en retirant leur chevauchement (chunk_overlap) :
    la fin d'un chunk est répétée au début du suivant, on ne la garde qu'une fois.
    """
    merged = texts[0]
    for text in texts[1:]:
        overlap = next((n for n in range(min(len(merged), len(text)), 0, -1) if merged.endswith(text[:n])), 0)
        merged += text[overlap:] if overlap >= 20 else "\n" + text  # Moins de 20 caractères communs : coïncidence
    return merged


class PineconeRetriever(BaseRetriever):
    # BaseRetriever est un modèle Pydantic : les attributs doivent être déclarés comme champs
    index: Any  # Index Pinecone (pinecone.Index)
    embed_model: Embeddings  # Même modèle d'embedding que celui utilisé pour l'indexation
    top_k: int = 3  # Nombre de chunks renvoyés
    reranker: Any = None  # Cross-encoder (sentence_transformers.CrossEncoder) ; None = pas de reclassement
    fetch_k: int = 20  # Nombre de chunks demandés à Pinecone avant reclassement (ignoré sans reranker)
    window: int = 0  # Nombre de chunks voisins ajoutés de chaque côté de chaque chunk retenu (0 = aucun)
    min_score: float = 0.0  # Seuil du reranker : en dessous, pour le meilleur passage, aucun passage n'est renvoyé

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_vector = self.embed_model.embed_query(query)  # Vectorisation de la question
        k = max(self.fetch_k, self.top_k) if self.reranker else self.top_k
        res = self.index.query(vector=query_vector, top_k=k, include_metadata=True)

        # Le texte du chunk est stocké dans les métadonnées sous la clé "text" (voir load_vectorstore.py)
        docs = [
            Document(
                id=match["id"],  # Identifiant « {fichier}-{numéro du chunk} » : permet de retrouver les voisins
                page_content=match["metadata"].get("text", ""),
                metadata={**match["metadata"], "score": match["score"]},  # Score de similarité utile pour le débogage
            )
            for match in res["matches"]
        ]
        if self.reranker and docs:
            # Reclassement : le cross-encoder note chaque paire (question, passage), titre du document compris
            pairs = [(query, f"{doc.metadata.get('title', '')}\n{doc.page_content}".strip()) for doc in docs]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(docs, scores):
                doc.metadata["vector_score"] = doc.metadata["score"]  # Score de la recherche vectorielle, conservé pour comparaison
                doc.metadata["score"] = float(score)  # Score du reranker (0 à 1) : c'est lui qui est affiché dans les citations
            docs.sort(key=lambda doc: doc.metadata["score"], reverse=True)
            # Barrière de pertinence : question manifestement hors du corpus -> contexte vide, le LLM répond qu'il n'a rien trouvé
            if docs[0].metadata["score"] < self.min_score:
                return []
        docs = docs[:self.top_k]
        return self._expand_with_neighbors(docs) if self.window > 0 and docs else docs

    def _expand_with_neighbors(self, docs: List[Document]) -> List[Document]:
        """
        Élargissement du contexte : avec des chunks de 500 caractères, une réponse est souvent coupée entre deux chunks
        (ex. la liste ABCDE du mélanome, dont « D comme Diamètre » est dans le chunk suivant). On ajoute donc à chaque
        chunk retenu ses voisins dans le document ; deux chunks retenus proches sont fusionnés en un seul passage.
        """
        # Identifiants des chunks à récupérer : {fichier}-{i - window} à {fichier}-{i + window}
        windows = []  # (fichier, numéros des chunks de la fenêtre, document retenu)
        for doc in docs:
            stem, _, number = doc.id.rpartition("-")
            i = int(number)
            windows.append((stem, set(range(max(0, i - self.window), i + self.window + 1)), doc))

        # Deux fenêtres d'un même fichier qui se touchent ou se chevauchent : fusionnées, au rang du meilleur chunk
        merged = []
        for stem, numbers, doc in windows:
            for group in merged:
                if group[0] == stem and min(numbers) <= max(group[1]) + 1 and min(group[1]) <= max(numbers) + 1:
                    group[1].update(numbers)
                    break
            else:
                merged.append((stem, numbers, doc))

        ids = [f"{stem}-{n}" for stem, numbers, _ in merged for n in numbers]
        fetched = self.index.fetch(ids=ids).vectors  # Les identifiants inexistants (début ou fin du document) sont ignorés
        expanded = []
        for stem, numbers, doc in merged:
            texts = [fetched[f"{stem}-{n}"].metadata["text"] for n in sorted(numbers) if f"{stem}-{n}" in fetched]
            # Les métadonnées (fichier, page, score) restent celles du chunk retenu : c'est lui qui est cité
            expanded.append(Document(id=doc.id, page_content=merge_overlapping(texts) if texts else doc.page_content,
                                     metadata=doc.metadata))
        return expanded
