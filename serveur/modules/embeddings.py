# Choix du modèle d'embedding, partagé par l'indexation (load_vectorstore.py) et la recherche (retriever.py).
# Les documents et les questions doivent impérativement être vectorisés avec le même modèle :
# changer de modèle impose de réindexer tout le corpus, dans un index Pinecone dédié.
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

# "local" (par défaut) : modèle open source exécuté sur la machine, gratuit et sans quota
# "gemini" : API Google, limitée à 1000 embeddings par jour avec l'offre gratuite
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_DIMENSION = 768  # Les deux modèles produisent des vecteurs de 768 dimensions


def get_embed_model() -> Embeddings:
    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",  # Modèle d'embedding multilingue de Google (prend en charge le français)
            output_dimensionality=EMBEDDING_DIMENSION,  # Doit correspondre à la dimension de l'index Pinecone
        )

    # Import local : PyTorch n'est chargé que si ce modèle est utilisé
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",  # Modèle multilingue (dont le français) conçu pour la recherche de passages
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},  # Carte graphique si disponible
        # Les modèles E5 ont été entraînés avec des préfixes : "passage: " pour les documents, "query: " pour les questions.
        # Les oublier dégrade nettement la qualité de la recherche.
        # normalize_embeddings : vecteurs de norme 1, adaptés à la similarité cosinus de l'index
        encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True, "batch_size": 32},
        query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
    )
