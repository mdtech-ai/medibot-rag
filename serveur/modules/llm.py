from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def format_docs(docs):
    """Concatène le contenu des documents récupérés pour former le contexte du prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm_chain(retriever):
    """
    Crée une chaîne RAG (LCEL) basée sur le modèle ChatGroq.

    Args:
        retriever: L'objet retriever utilisé pour récupérer les documents pertinents.

    Returns:
        Une chaîne LCEL qui prend la question (str) en entrée via `invoke` et renvoie
        un dictionnaire avec les clés "question", "source_documents" et "answer".
    """

    # Initialize the ChatGroq model with the API key
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name='openai/gpt-oss-120b',  # Llama 3.3 a été retiré de Groq ; liste à jour : https://console.groq.com/docs/models
        )

     # Define the prompt template for the LLM
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Tu es **MediBot**, un assistant basé sur l'intelligence artificielle, conçu pour aider les utilisateurs à comprendre des documents médicaux et à répondre à leurs questions liées à la santé.

        Ta mission est de fournir des réponses claires, précises et utiles en te basant **uniquement sur le contexte fourni**.

        📋 **Consignes** :
        - Réponds toujours en français, sur un ton calme, factuel et respectueux.
        - Utilise des explications simples lorsque c'est nécessaire.
        - Si le contexte ne contient pas la réponse, dis : « Je suis désolé, mais je n'ai pas trouvé d'informations pertinentes dans les documents fournis. »
        - N'invente JAMAIS de faits.
        - Ne donne PAS de conseils médicaux ni de diagnostics.

        ---

        🔍 **Contexte** :
        {context}

        🙋‍♂️ **Question de l'utilisateur** :
        {question}

        ---

        💬 **Réponse** :
        """
    )

    # Sous-chaîne de génération : documents -> contexte -> prompt -> LLM -> texte
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["source_documents"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chaîne complète : récupère les documents, puis ajoute la réponse générée en conservant les sources
    return RunnableParallel(
        source_documents=retriever,
        question=RunnablePassthrough(),
    ).assign(answer=answer_chain)