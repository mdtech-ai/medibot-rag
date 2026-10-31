"""Tests du nettoyage des PDF (modules/pdf_cleaning.py) : en-têtes répétés, sommaire, bibliographie, pages vides."""
from langchain_core.documents import Document
from modules.pdf_cleaning import clean_pages

HEADER = "Guide du parcours de soins – BPCO"
BODY = "La BPCO est une maladie respiratoire chronique qui se manifeste par un essoufflement à l'effort. " * 2


def page(i: int, body: str | None = None) -> Document:
    """Page type d'un guide : en-tête répété, contenu propre à la page, numéro de page en pied de page."""
    # Contenu différent sur chaque page : une ligne identique sur toutes les pages serait (à raison) traitée comme un en-tête
    body = body or f"Section {chr(65 + i)} : {BODY}"
    return Document(page_content=f"{HEADER}\n{body}\n{i + 1}", metadata={"source": "guide.pdf", "page": i})


def test_retire_les_en_tetes_et_numeros_de_page_repetes():
    cleaned = clean_pages([page(i) for i in range(10)])
    assert len(cleaned) == 10
    for doc in cleaned:
        assert HEADER not in doc.page_content
        assert "essoufflement" in doc.page_content
        assert not doc.page_content.strip().splitlines()[-1].isdigit()  # Numéro de page retiré


def test_conserve_les_metadonnees_pour_les_citations():
    cleaned = clean_pages([page(i) for i in range(5)])
    assert [doc.metadata["page"] for doc in cleaned] == [0, 1, 2, 3, 4]


def test_retire_le_sommaire_en_debut_de_document():
    toc = "Sommaire\n1. Définition ....... 4\n2. Diagnostic ....... 7\n3. Traitement ....... 12\n4. Suivi ....... 20"
    pages = [page(0, toc)] + [page(i) for i in range(1, 20)]
    cleaned = clean_pages(pages)
    assert all("Sommaire" not in doc.page_content for doc in cleaned)
    assert len(cleaned) == 19


def test_ignore_la_bibliographie_et_les_pages_suivantes():
    pages = [page(i) for i in range(10)]
    pages[8] = page(8, "Références\n1. Haute Autorité de Santé. Guide BPCO. 2019.\n2. GOLD report. 2023.")
    cleaned = clean_pages(pages)
    assert [doc.metadata["page"] for doc in cleaned] == list(range(8))


def test_un_titre_references_en_debut_de_document_n_est_pas_une_bibliographie():
    # Seule la seconde partie du document est examinée : un titre « Références » au début ne supprime rien
    pages = [page(i) for i in range(10)]
    pages[2] = page(2, "Références\n" + BODY)
    assert len(clean_pages(pages)) == 10


def test_ignore_les_pages_quasi_vides():
    pages = [page(i) for i in range(6)] + [page(6, "Notes")]
    cleaned = clean_pages(pages)
    assert len(cleaned) == 6


def test_document_vide():
    assert clean_pages([]) == []
