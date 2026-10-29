# Schémas Pydantic : format des données échangées avec l'API (validation des entrées et documentation automatique dans /docs)
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Auteur du message : l'utilisateur ou MediBot")
    content: str = Field(..., max_length=8000, description="Contenu du message")


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # Supprime les espaces en début/fin : "   " devient "" et est rejeté

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question posée à MediBot",
        examples=["Combien existe-t-il de types de diabète ?"],
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
        description="Messages précédents de la conversation, du plus ancien au plus récent (seuls les derniers sont utilisés)",
    )
    session_id: str | None = Field(
        None,
        max_length=100,
        description="Identifiant de la conversation, généré par le client : regroupe ses questions dans Langfuse",
        examples=["3f2b8c1e-conversation-demo"],
    )


class Source(BaseModel):
    file: str = Field(..., description="Nom du fichier PDF", examples=["DIABETES.pdf"])
    page: int | None = Field(None, description="Numéro de page (commence à 1)", examples=[2])
    excerpt: str = Field(..., description="Début du passage utilisé pour répondre")
    score: float | None = Field(None, description="Score de pertinence du passage (0 à 1) : score du reranker, ou similarité cosinus sans reranking")


class AskResponse(BaseModel):
    response: str = Field(..., description="Réponse générée par MediBot à partir des documents")
    standalone_question: str = Field(..., description="Question autonome réellement utilisée pour la recherche (reformulée si un historique est fourni)")
    sources: list[Source] = Field(default_factory=list, description="Passages utilisés, sans doublons (un par fichier et par page)")
