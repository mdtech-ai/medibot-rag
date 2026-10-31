"""
Tests du retriever (modules/retriever.py) avec de faux index, modèle d'embedding et reranker :
reclassement, seuil de pertinence, élargissement aux chunks voisins et recollage sans chevauchement.
"""
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings
from modules.retriever import PineconeRetriever, merge_overlapping

# Document fictif découpé en 5 chunks qui se chevauchent de 30 caractères, comme le fait RecursiveCharacterTextSplitter
FULL_TEXT = (
    "La règle ABCDE aide à repérer un mélanome. A comme Asymétrie de la lésion. B comme Bords irréguliers et mal définis. "
    "C comme Couleur non homogène. D comme Diamètre supérieur à 6 millimètres. E comme Évolution rapide de la taille."
)
CHUNKS = [FULL_TEXT[max(0, i - 30):i + 50] for i in range(0, len(FULL_TEXT), 50)]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class FakeIndex:
    """Faux index Pinecone : query renvoie les chunks dans l'ordre donné, fetch les renvoie par identifiant."""

    def __init__(self, order):
        self.order = order
        self.vectors = {
            f"melanome-{i}": SimpleNamespace(metadata={"text": text, "source": "melanome.pdf", "page": 0, "title": "Mélanome"})
            for i, text in enumerate(CHUNKS)
        }

    def query(self, vector, top_k, include_metadata):
        matches = [{"id": f"melanome-{i}", "score": 0.8, "metadata": self.vectors[f"melanome-{i}"].metadata} for i in self.order]
        return {"matches": matches[:top_k]}

    def fetch(self, ids):
        return SimpleNamespace(vectors={i: self.vectors[i] for i in ids if i in self.vectors})


class FakeReranker:
    """Faux cross-encoder : score fixé par chunk, pour contrôler le classement."""

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text

    def predict(self, pairs):
        return [next((s for t, s in self.scores_by_text.items() if t in passage), 0.0) for _, passage in pairs]


def test_merge_overlapping_retire_le_chevauchement():
    assert merge_overlapping(CHUNKS) == FULL_TEXT


def test_merge_overlapping_sans_chevauchement_conserve_les_deux_textes():
    assert merge_overlapping(["Premier passage.", "Second passage."]) == "Premier passage.\nSecond passage."


def test_sans_reranker_renvoie_les_top_k_de_pinecone():
    retriever = PineconeRetriever(index=FakeIndex([2, 0, 1]), embed_model=FakeEmbeddings(), top_k=2)
    docs = retriever.invoke("question")
    assert [doc.id for doc in docs] == ["melanome-2", "melanome-0"]


def test_le_reranker_reclasse_les_passages():
    reranker = FakeReranker({"Diamètre": 0.95, "Asymétrie": 0.40})
    retriever = PineconeRetriever(index=FakeIndex([0, 1, 2, 3]), embed_model=FakeEmbeddings(), top_k=1, reranker=reranker)
    docs = retriever.invoke("taille suspecte ?")
    assert "Diamètre" in docs[0].page_content
    assert docs[0].metadata["score"] == 0.95
    assert docs[0].metadata["vector_score"] == 0.8  # Score vectoriel conservé


def test_seuil_de_pertinence_renvoie_un_contexte_vide():
    reranker = FakeReranker({})  # Tous les passages à 0 : question hors du corpus
    retriever = PineconeRetriever(index=FakeIndex([0, 1]), embed_model=FakeEmbeddings(), top_k=2, reranker=reranker, min_score=0.03)
    assert retriever.invoke("Comment traite-t-on une migraine ?") == []


def test_elargissement_aux_voisins():
    # Le chunk retenu ne contient que le début de la liste ; ses voisins complètent la réponse
    retriever = PineconeRetriever(index=FakeIndex([2]), embed_model=FakeEmbeddings(), top_k=1, window=1)
    docs = retriever.invoke("question")
    assert len(docs) == 1
    assert docs[0].page_content == merge_overlapping(CHUNKS[1:4])
    assert docs[0].id == "melanome-2"  # Métadonnées du chunk retenu (celui qui est cité)


def test_deux_fenetres_proches_sont_fusionnees():
    retriever = PineconeRetriever(index=FakeIndex([1, 2]), embed_model=FakeEmbeddings(), top_k=2, window=1)
    docs = retriever.invoke("question")
    assert len(docs) == 1
    assert docs[0].page_content == merge_overlapping(CHUNKS[0:4])


def test_elargissement_en_debut_et_fin_de_document():
    retriever = PineconeRetriever(index=FakeIndex([0, len(CHUNKS) - 1]), embed_model=FakeEmbeddings(), top_k=2, window=1)
    docs = retriever.invoke("question")
    assert docs[0].page_content == merge_overlapping(CHUNKS[0:2])
    assert docs[1].page_content == merge_overlapping(CHUNKS[-2:])
