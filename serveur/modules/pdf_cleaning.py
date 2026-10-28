"""
Nettoyage du texte extrait des PDF avant le découpage en chunks.

Les guides HAS et INCa contiennent du texte qui n'apporte aucune information médicale mais pollue la recherche :
- en-têtes et pieds de page répétés sur chaque page (titre du guide, organisme, numéro de page) : avec des chunks
  de 500 caractères, ils occupent jusqu'à un quart du premier chunk de chaque page et rendent tous les chunks
  d'un même document artificiellement similaires ;
- sommaires : des listes de titres de sections qui « ressemblent » à toutes les questions ;
- bibliographie et pages de fin (participants, remerciements) : elles reprennent le vocabulaire du sujet
  et remontaient dans les résultats (ex. : la bibliographie du guide MRC citée comme source).

Les règles sont génériques (aucune liste propre à un document), pour s'appliquer aussi aux PDF ajoutés par l'API.
"""
import re
from collections import Counter

from langchain_core.documents import Document

from logger import logger

REPEATED_LINE_MIN_RATIO = 0.3  # Une ligne présente sur au moins 30 % des pages est un en-tête ou un pied de page
TOC_ZONE_RATIO = 0.15  # Les sommaires sont cherchés dans les 15 premiers % du document
TOC_LINE_MIN_RATIO = 0.5  # Page de sommaire : au moins la moitié des lignes se terminent par un numéro de page
END_MATTER_ZONE_RATIO = 0.6  # La bibliographie n'est cherchée qu'après 60 % du document
MIN_PAGE_LENGTH = 100  # Page quasi vide après nettoyage (couverture, page blanche) : ignorée

BIBLIOGRAPHY_TITLE = re.compile(r"^(r[ée]f[ée]rences(\s+bibliographiques)?|bibliographie)$", re.IGNORECASE)
TOC_TITLE = re.compile(r"^(sommaire|table des mati[èe]res)$", re.IGNORECASE)
# Ligne de sommaire : texte court terminé par un numéro de page, avec ou sans points de suite (« ....... 12 », « ____ 86 »)
TOC_LINE = re.compile(r"^.{3,100}?(\s|\.{2,}|_{2,}|…)\s*\d{1,3}$")


def normalize(line: str) -> str:
    """Espaces normalisés et chiffres remplacés par # : « Page 12 » et « Page 13 » deviennent la même ligne."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def is_toc_page(lines: list[str]) -> bool:
    if any(TOC_TITLE.match(line) for line in lines):
        return True
    return len(lines) >= 5 and sum(bool(TOC_LINE.match(line)) for line in lines) / len(lines) >= TOC_LINE_MIN_RATIO


def clean_pages(pages: list[Document]) -> list[Document]:
    """
    Nettoie les pages d'un même PDF (une Document par page, comme les renvoie PyPDFLoader) :
    retire les lignes répétées, les pages de sommaire, la bibliographie et tout ce qui la suit, les pages vides.
    Les métadonnées (fichier, page) sont conservées pour les citations.
    """
    if not pages:
        return pages
    page_lines = [[line.strip() for line in page.page_content.splitlines() if line.strip()] for page in pages]

    # 1. En-têtes et pieds de page : lignes (normalisées) présentes sur une grande partie des pages
    counts = Counter(norm for lines in page_lines for norm in {normalize(line) for line in lines})
    min_count = max(3, REPEATED_LINE_MIN_RATIO * len(pages))
    repeated = {norm for norm, count in counts.items() if count >= min_count}

    # 2. Bibliographie : à partir de son titre (dans la seconde partie du document), tout le reste est ignoré
    end = len(pages)
    for i in range(int(END_MATTER_ZONE_RATIO * len(pages)), len(pages)):
        if any(BIBLIOGRAPHY_TITLE.match(line) for line in page_lines[i]):
            end = i
            break

    cleaned, removed_toc, removed_empty = [], 0, 0
    for i in range(end):
        if i < max(1, TOC_ZONE_RATIO * len(pages)) and is_toc_page(page_lines[i]):
            removed_toc += 1
            continue
        text = "\n".join(line for line in page_lines[i] if normalize(line) not in repeated)
        if len(text) < MIN_PAGE_LENGTH:
            removed_empty += 1
            continue
        cleaned.append(Document(page_content=text, metadata=pages[i].metadata))

    logger.info(
        f"Nettoyage : {len(repeated)} ligne(s) répétée(s) retirée(s), {removed_toc} page(s) de sommaire, "
        f"{len(pages) - end} page(s) de bibliographie et de fin, {removed_empty} page(s) vide(s) "
        f"-> {len(cleaned)}/{len(pages)} pages conservées"
    )
    return cleaned
