from dotenv import load_dotenv

# Chargement des variables d'environnement (.env) une seule fois, avant tout import de module qui les utilise
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_router 
from routes.ask_question import router as ask_router


app = FastAPI(title="MedicalAIAssistant API", description="API for MedicalAIAssistant Chatbot", version="1.0.0")


# CORS configuration to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials= True, # Allow credentials (cookies, authorization headers, etc.)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# middlewares exception handlers: En français, gestion des exceptions pour les middlewares
app.middleware("http")(catch_exception_middleware)

# routers : (routes) En français, définition des routes pour l'API

# 1. Une route sera dédiée au téléchargement des fichiers PDF. (upload pdf documents)
# Cette route permettra aux utilisateurs de soumettre des documents PDF à l'application pour traitement. Le fichier sera reçu via une requête POST, et le serveur pourra ensuite extraire le texte du PDF pour l'analyse.
app.include_router(upload_router)
# 2. Une autre route sera dédiée à la génération de réponses basées sur le texte extrait des fichiers PDF. (asking query)
# Cette route permettra aux utilisateurs de poser des questions ou de demander des informations spécifiques, et l'application utilisera le texte extrait pour fournir des réponses pertinentes.
app.include_router(ask_router)
