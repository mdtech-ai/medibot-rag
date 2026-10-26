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


class Source(BaseModel):
    file: str = Field(..., description="Nom du fichier PDF", examples=["DIABETES.pdf"])
    page: int | None = Field(None, description="Numéro de page (commence à 1)", examples=[2])
    excerpt: str = Field(..., description="Début du passage utilisé pour répondre")
    score: float | None = Field(None, description="Score de similarité cosinus entre la question et le passage (0 à 1)")


class AskResponse(BaseModel):
    response: str = Field(..., description="Réponse générée par MediBot à partir des documents")
    sources: list[Source] = Field(default_factory=list, description="Passages utilisés, sans doublons (un par fichier et par page)")
