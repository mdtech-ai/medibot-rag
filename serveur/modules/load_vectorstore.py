# Module de gestion du chargement du vectorstore pour l'application MedicalAIAssistant
import os 
import time 
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.embeddings import EMBEDDING_DIMENSION, get_embed_model
from modules.pdf_cleaning import clean_pages
from modules.pdf_handlers import save_uploaded_files
from logger import logger


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1" # Région de Pinecone (AWS par défaut: version gratuite US East 1)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # Nom de l'index Pinecone pour stocker les vecteurs (défini dans .env)

# Initialisation de Pinecone avec la clé d'API 
pc = Pinecone(api_key=PINECONE_API_KEY) 
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)  # Spécification du serveur Pinecone (AWS par défaut)
existing_indexes = [i["name"] for i in pc.list_indexes()]  # Liste des index existants dans Pinecone

if PINECONE_INDEX_NAME not in existing_indexes:
    # Si l'index n'existe pas, nous le créons avec la dimension des vecteurs d'embedding et la métrique "cosine" pour la similarité.
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,   # Taille des vecteurs d'embedding (768, voir modules/embeddings.py)
        metric="cosine",  # Type de métrique pour la similarité (cosine : indépendante de la norme des vecteurs)
        spec=spec)

    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        # Attente que l'index soit prêt avant de continuer
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)  # Création d'une instance de l'index Pinecone pour interagir avec les vecteurs stockés

# Modèle d'embedding pour transformer le texte en vecteurs numériques (choix du modèle : voir modules/embeddings.py).
# Créé une seule fois au démarrage et partagé par l'indexation (ce module) et la recherche (modules/retriever.py) :
# les documents et les questions doivent impérativement être vectorisés avec le même modèle et la même dimension.
embed_model = get_embed_model()

EMBED_BATCH_SIZE = 100  # Nombre de textes envoyés par requête d'embedding
EMBED_MAX_RETRIES = 8  # Nouvelles tentatives en cas d'erreur temporaire (quota de l'offre gratuite Gemini, panne passagère)
# Erreurs temporaires qui justifient de réessayer : quota dépassé (429) et indisponibilité passagère du service (500, 503)
RETRYABLE_ERRORS = ("429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "503", "UNAVAILABLE")


def embed_with_retry(texts: list[str]) -> list[list[float]]:
    """
    Vectorise les textes par lots. En cas d'erreur temporaire (quota ou indisponibilité du service),
    attend de plus en plus longtemps avant de réessayer (backoff exponentiel : 15 s, 30 s, puis 60 s maximum,
    car les quotas de l'API Gemini sont comptés par minute).
    """
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                vectors.extend(embed_model.embed_documents(batch))
                break
            except Exception as e:
                is_retryable = any(code in str(e) for code in RETRYABLE_ERRORS)
                if not is_retryable or attempt == EMBED_MAX_RETRIES - 1:
                    raise
                wait = min(15 * 2 ** attempt, 60)
                logger.warning(
                    f"Erreur temporaire d'embedding ({str(e)[:120]}...), "
                    f"nouvelle tentative dans {wait} s ({attempt + 1}/{EMBED_MAX_RETRIES})"
                )
                time.sleep(wait)
        logger.info(f"Embeddings : {min(start + EMBED_BATCH_SIZE, len(texts))}/{len(texts)} chunks")
    return vectors


def document_title(file_path: str, pages) -> str:
    """Titre du document : celui des métadonnées du PDF s'il existe, sinon le nom du fichier (« guide_bpco » -> « guide bpco »)."""
    title = (pages[0].metadata.get("title") or "").strip() if pages else ""
    return title or Path(file_path).stem.replace("_", " ").replace("-", " ")


def index_pdf(file_path: str, title: str | None = None) -> int:
    """
    Indexe un fichier PDF déjà présent sur le disque : extraction du texte, découpage en chunks,
    vectorisation (embed) et indexation (upsert) dans Pinecone. Renvoie le nombre de chunks indexés.
    Utilisée par la route /upload_pdfs et par le script d'indexation du corpus (scripts/index_corpus.py).

    title : titre du document, ajouté en en-tête de chaque chunk (par défaut : métadonnées du PDF ou nom du fichier).
    """
    # Chargement du fichier PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    title = title or document_title(file_path, pages)
    # Nettoyage avant découpage : en-têtes répétés, sommaire, bibliographie (voir modules/pdf_cleaning.py)
    documents = clean_pages(pages)

    # Division du texte en morceaux (chunks) pour l'indexation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Taille maximale de chaque chunk (en caractères)
        chunk_overlap=100,  # Chevauchement entre les chunks pour conserver le contexte
    )
    chunks = text_splitter.split_documents(documents)

    # En-tête de contexte (« contextual chunk header ») : le titre du document est ajouté au texte vectorisé.
    # Le nettoyage retire les en-têtes répétés de chaque page, qui portaient aussi le sujet du document : sans ce titre,
    # un chunk sur « les seuils de mesure » ne dirait plus qu'il parle d'hypertension et ne remonterait plus.
    texts = [f"{title}\n\n{chunk.page_content}" for chunk in chunks]
    # Métadonnées stockées avec chaque vecteur : uniquement ce qui sert aux citations, plus le texte du chunk sous la clé "text"
    # (Pinecone ne stocke que les vecteurs : sans cette clé, la route /ask/ ne pourrait pas retrouver le contenu).
    # On écarte les métadonnées du PDF (auteur, logiciel, dates...) : inutiles, et elles alourdissent chaque vecteur.
    metadata = [
        {
            "source": Path(file_path).name,  # Nom du fichier seul : ne révèle pas l'arborescence du serveur
            "page": chunk.metadata.get("page", 0),  # Numéro de page (commence à 0)
            "page_label": str(chunk.metadata.get("page_label", "")),  # Numéro affiché dans le PDF
            "title": title,  # Titre du document : indiqué au LLM avec chaque passage (voir modules/llm.py)
            "text": chunk.page_content,
        }
        for chunk in chunks
    ]
    ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]  # Génération d'IDs uniques pour chaque chunk

    # Transformation des textes en vecteurs d'embedding
    logger.info(f"Embedding de {len(chunks)} chunks pour {Path(file_path).name}")
    embedding = embed_with_retry(texts)

    # Upsert des vecteurs dans Pinecone
    logger.info("Upsert des embeddings dans Pinecone...")
    vectors = list(zip(ids, embedding, metadata))  # Liste de tuples (id, vecteur, métadonnées) attendue par Pinecone
    batch_size = 100  # Envoi par lots : Pinecone limite la taille de chaque requête (2 Mo)
    with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress :
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)  # Upsert du lot de vecteurs dans Pinecone
            progress.update(len(batch))  # Mise à jour de la barre de progression

    logger.info(f"Indexation terminée pour {Path(file_path).name}")
    return len(chunks)


# Fonction appelée par la route /upload_pdfs : sauvegarde les fichiers reçus, puis les indexe un par un
def load_vectorstore(uploaded_files):
    # Sauvegarde des fichiers PDF téléchargés sur le disque (voir modules/pdf_handlers.py)
    file_paths = save_uploaded_files(uploaded_files)

    for file_path in tqdm(file_paths, desc="Processing PDF files"):
        index_pdf(file_path)

