# Module de gestion du chargement du vectorstore pour l'application MedicalAIAssistant
import os 
import time 
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1" # Région de Pinecone (AWS par défaut: version gratuite US East 1)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # Nom de l'index Pinecone pour stocker les vecteurs (défini dans .env)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Modèle d'Embedding Google Generative AI nécessite la clé d'API Google pour fonctionner correctement.

# Nous allons créer un répertoire pour stocker les fichiers PDF téléchargés par les utilisateurs. Ce répertoire sera utilisé pour sauvegarder temporairement les documents avant de les traiter et d'extraire le texte pour l'analyse.
UPLOAD_DIR = "./uploaded_docs"  # Répertoire pour stocker les fichiers PDF téléchargés
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Crée le répertoire s'il n'existe pas déjà

# Initialisation de Pinecone avec la clé d'API 
pc = Pinecone(api_key=PINECONE_API_KEY) 
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)  # Spécification du serveur Pinecone (AWS par défaut)
existing_indexes = [i["name"] for i in pc.list_indexes()]  # Liste des index existants dans Pinecone

if PINECONE_INDEX_NAME not in existing_indexes:
    # Si l'index n'existe pas, nous le créons avec une dimension de 768 (taille des vecteurs d'embedding) et un type de métrique "cosine" pour la similarité.
    pc.create_index(
        name=PINECONE_INDEX_NAME, 
        dimension=768,   # Taille des vecteurs d'embedding (768, imposée via output_dimensionality du modèle gemini-embedding-001)
        metric="cosine",  # Type de métrique pour la similarité (cosine : robuste car les vecteurs réduits à 768 ne sont pas normalisés)
        spec=spec)

    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        # Attente que l'index soit prêt avant de continuer
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)  # Création d'une instance de l'index Pinecone pour interagir avec les vecteurs stockés

# Fonction pour charger un fichier PDF, extraire le contenu, le diviser en morceaux, le vectoriser (embed) et l'indexer (upsert) dans Pinecone

def load_vectorstore(uploaded_files): 
    # 1. Initialisation du modèle d'embedding Google Generative AI pour transformer le texte en vecteurs numériques
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # Modèle d'embedding multilingue de Google (prend en charge le français)
        output_dimensionality=768,  # Doit correspondre à la dimension de l'index Pinecone
    )
    
    # Liste pour stocker les chemins des fichiers PDF téléchargés
    file_paths = [] 
    # 2. upload des fichiers PDF dans le répertoire UPLOAD_DIR
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # 3. Chargement et traitement des fichiers PDF
    for file_path in tqdm(file_paths, desc="Processing PDF files"):
        # Chargement du fichier PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Division du texte en morceaux (chunks) pour l'indexation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Taille maximale de chaque chunk (en caractères)
            chunk_overlap=100,  # Chevauchement entre les chunks pour conserver le contexte
        )
        chunks = text_splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]  # Extraction du contenu textuel des chunks
        # Extraction des métadonnées associées aux chunks, en y ajoutant le texte du chunk sous la clé "text"
        # (Pinecone ne stocke que les vecteurs : sans cette clé, la route /ask/ ne pourrait pas retrouver le contenu)
        metadata = [{**chunk.metadata, "text": chunk.page_content} for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]  # Génération d'IDs uniques pour chaque chunk

        # 4. Transformation des textes en vecteurs d'embedding 
        print(f"{chunks} Embedding chunks")
        embedding = embed_model.embed_documents(texts)  # Transformation des textes en vecteurs d'embedding


        # 5. Préparation des vecteurs pour l'upsert dans Pinecone
        print("Upserting embeddings...")
        vectors = list(zip(ids, embedding, metadata))  # Liste de tuples (id, vecteur, métadonnées) attendue par Pinecone
        batch_size = 100  # Envoi par lots : Pinecone limite la taille de chaque requête (2 Mo)
        with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress :
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)  # Upsert du lot de vecteurs dans Pinecone
                progress.update(len(batch))  # Mise à jour de la barre de progression

        print(f"Upload complete for {file_path}") 

