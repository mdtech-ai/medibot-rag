from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"  # Llama 3.3 a été retiré de Groq ; liste à jour : https://console.groq.com/docs/models
MAX_HISTORY_MESSAGES = 6  # Nombre de messages récents conservés (3 échanges) : assez pour le contexte, sans alourdir le prompt

def format_docs(docs):
    """Concatène le contenu des documents récupérés pour former le contexte du prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    """Met en forme les derniers messages de la conversation : une ligne par message, préfixée par l'auteur."""
    lines = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        author = "Utilisateur" if message["role"] == "user" else "MediBot"
        lines.append(f"{author} : {message['content']}")
    return "\n".join(lines) if lines else "(aucun échange précédent)"

def get_llm_chain(retriever):
    """
    Crée une chaîne RAG conversationnelle (LCEL) basée sur le modèle ChatGroq.

    Étapes :
    1. Reformulation : si un historique existe, la question de suivi est réécrite en question autonome
       (ex. « Et comment on le traite ? » -> « Comment traite-t-on le diabète de type 1 ? »)
    2. Recherche : la question autonome est envoyée au retriever (Pinecone)
    3. Génération : le LLM répond à partir des passages récupérés et de l'historique

    Args:
        retriever: L'objet retriever utilisé pour récupérer les documents pertinents.

    Returns:
        Une chaîne LCEL qui prend en entrée {"question": str, "history": list[{"role", "content"}]}
        et renvoie un dictionnaire avec les clés "question", "history", "standalone_question",
        "source_documents" et "answer".
    """

    # Modèle principal : génération des réponses.
    # temperature=0 : réponses déterministes. Pour un assistant médical factuel, la même question doit recevoir
    # la même réponse ; c'est aussi indispensable pour que deux évaluations successives soient comparables.
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0,
        )

    # Modèle de reformulation : même modèle, mais déterministe (temperature=0) et avec un raisonnement réduit,
    # car la tâche est simple : on gagne en latence sans perdre en qualité
    condense_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0,
        reasoning_effort="low",
        )

    # Prompt de reformulation de la question de suivi en question autonome
    condense_prompt = PromptTemplate(
        input_variables=["history", "question"],
        template="""
        À partir de l'historique de la conversation et de la question de suivi, reformule la question de suivi
        en une question autonome, compréhensible sans l'historique, pour effectuer une recherche dans des documents médicaux.

        Règles :
        - Remplace les pronoms et références implicites (« il », « ça », « ce type », « ses symptômes »...) par ce qu'ils désignent.
        - Si la question est déjà autonome, renvoie-la telle quelle.
        - Ne réponds PAS à la question : renvoie uniquement la question reformulée, sans aucun autre texte.
        - Conserve la langue de la question de suivi.

        Historique :
        {history}

        Question de suivi : {question}

        Question autonome :
        """
    )

    # Prompt de génération de la réponse
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""
        Tu es **MediBot**, un assistant basé sur l'intelligence artificielle, conçu pour aider les utilisateurs à comprendre des documents médicaux et à répondre à leurs questions liées à la santé.

        Ta mission est de fournir des réponses claires, précises et utiles en te basant **uniquement sur le contexte fourni**.

        📋 **Consignes** :
        - Réponds toujours en français, sur un ton calme, factuel et respectueux.
        - Utilise des explications simples lorsque c'est nécessaire.
        - L'historique de la conversation sert uniquement à comprendre la question ; les faits de ta réponse doivent provenir du contexte.
        - Si le contexte ne contient pas la réponse, dis : « Je suis désolé, mais je n'ai pas trouvé d'informations pertinentes dans les documents fournis. »
        - N'invente JAMAIS de faits.
        - Ne donne PAS de conseils médicaux ni de diagnostics.

        ---

        🗂️ **Historique de la conversation** :
        {history}

        🔍 **Contexte** :
        {context}

        🙋‍♂️ **Question de l'utilisateur** :
        {question}

        ---

        💬 **Réponse** :
        """
    )

    # 1. Reformulation : sans historique, la question est utilisée telle quelle (pas d'appel LLM supplémentaire)
    standalone_question_chain = RunnableBranch(
        (lambda x: not x.get("history"), itemgetter("question")),
        RunnablePassthrough.assign(history=lambda x: format_history(x["history"]))
        | condense_prompt
        | condense_llm
        | StrOutputParser()
        | (lambda text: text.strip()),
    ).with_config(run_name="reformulation")

    # 3. Sous-chaîne de génération : documents + historique -> prompt -> LLM -> texte
    answer_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source_documents"]),
            history=lambda x: format_history(x["history"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="generation")

    # Chaîne complète : chaque étape ajoute sa clé au dictionnaire, en conservant les précédentes
    return (
        RunnablePassthrough.assign(standalone_question=standalone_question_chain)
        # 2. Recherche avec la question autonome (et non la question brute de l'utilisateur)
        | RunnablePassthrough.assign(
            source_documents=(itemgetter("standalone_question") | retriever).with_config(run_name="recherche")
        )
        | RunnablePassthrough.assign(answer=answer_chain)
    )
