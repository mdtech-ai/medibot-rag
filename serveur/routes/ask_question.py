# Les bibliothèques nécessaires
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm  import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List
from logger import logger
import os

router = APIRouter()  # Initialisation du routeur FastAPI pour gérer les routes liées aux questions posées par les utilisateurs

@router.post("/ask/")
async def ask_question(question:str=Form(...)): 
    try: 
        logger.info(f"Requête de l'utilisateur:{question}")
        # Embed model + pinecone setup 
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])  # Nom de l'index défini dans .env
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768,  # Même dimension que l'index Pinecone (et que les vecteurs indexés)
            )
        embed_query = embed_model.embed_query(question)
        res = index.query(vector=embed_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content= match["metadata"].get("text", ""),\
                metadata = match["metadata"]
            ) for match in res["matches"]
        ]

        # 
        class SimpleRetriever(BaseRetriever):
            # BaseRetriever est un modèle Pydantic : les attributs doivent être déclarés comme champs
            docs: List[Document] = Field(default_factory=list)

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        retriever = SimpleRetriever(docs=docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain , question) 

        logger.info("Requête traitée avec succès")
        return result

    except Exception as e: 
        logger.exception("Erreur du traitement de la question") 
        return JSONResponse(status_code=500, content={"error": str(e)})
    