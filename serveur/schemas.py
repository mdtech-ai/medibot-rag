# Schémas Pydantic : format des données échangées avec l'API (validation des entrées et documentation automatique dans /docs)
from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # Supprime les espaces en début/fin : "   " devient "" et est rejeté

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question posée à MediBot",
        examples=["Combien existe-t-il de types de diabète ?"],
    )


class AskResponse(BaseModel):
    response: str = Field(..., description="Réponse générée par MediBot à partir des documents")
    sources: list[str] = Field(default_factory=list, description="Fichiers d'où proviennent les passages utilisés")
