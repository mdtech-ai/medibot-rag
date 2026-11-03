# MediBot — assistant médical RAG en français

MediBot répond à des questions de santé **à partir de guides officiels** de la Haute Autorité de Santé (HAS) et de
l'Institut national du cancer (INCa), en **citant ses sources** (document et page). Il refuse de répondre hors de
son corpus, ne pose pas de diagnostic et ne donne pas de posologie personnalisée.

Le projet couvre toute la chaîne d'un système RAG en production : indexation, recherche
vectorielle + reranking, génération en streaming, **évaluation mesurée** à chaque amélioration, observabilité,
tests, CI avec barrière d'évaluation, conteneurisation et déploiement.

> ⚠️ Projet de démonstration : MediBot fournit des informations générales et ne remplace pas l'avis d'un professionnel de santé.

![Architecture de MediBot : pipeline d'indexation, puis recherche et génération](docs/architecture.svg)

## Résultats

Mesurés sur un jeu de test de 43 questions écrit à la main (faits, listes, langage de patient, questions
multi-documents, questions de suivi, hors sujet, pièges : diagnostic, posologie, injection de prompt).

| | Version initiale | Version actuelle |
|---|---|---|
| **Recherche** — bonne page dans les 3 premiers passages (hit@3) | 69,7 % | **87,9 %** |
| **Recherche** — rang moyen de la bonne page (MRR@3) | 0,581 | **0,828** |
| **Réponses** — fidélité au contexte (pas d'hallucination) | 0,864 | **0,939** |
| **Réponses** — exactitude (faits de la référence présents) | 0,460 | **0,784** |
| **Réponses** — rappel du contexte | 0,494 | **0,852** |
| Faux refus (« je n'ai pas trouvé » alors que la réponse existe) | 15,2 % | **3,1 %** (1 cas) |
| Comportements attendus (refus hors sujet, diagnostic, injection) | 90 % | **100 %** |

Version actuelle : évaluation complète du 3 novembre 2026 ([`generation_20261103_165659.json`](data/eval/results/generation_20261103_165659.json)) ;
un cas sur 43 n'a pas pu être jugé (requête trop longue pour l'offre gratuite du juge, depuis l'élargissement du contexte).

Chaque amélioration a été mesurée séparément (résultats versionnés dans [`data/eval/results/`](data/eval/results/)) :

| Étape | Effet principal |
|---|---|
| Nettoyage des PDF (en-têtes répétés, sommaires, bibliographies) | hit@3 69,7 % → 75,8 % ; langage de patient 50 % → 83 % |
| Titre du document en en-tête de chaque passage | MRR 0,566 → 0,626 ; le LLM sait de quel guide vient chaque passage |
| **Reranking** par cross-encoder (20 candidats → 3) | hit@3 72,7 % → **87,9 %**, MRR 0,626 → **0,828** |
| Élargissement aux passages voisins | réponses coupées entre deux passages retrouvées ; rappel du contexte 0,60 → 0,86 |
| Consignes précisées (questions personnelles, sujet exact) | faux refus 15 % → 0 à 3 % selon les passages |
| Seuil de pertinence du reranker | question hors sujet bloquée avant le LLM (comportements 100 %) |

## Architecture

```
Navigateur ─► Chainlit (client/) ──HTTP/SSE──► API FastAPI (serveur/)
                                                 │
                    1. reformulation de la question de suivi (LLM, si historique)
                    2. recherche : embeddings multilingual-e5 ─► Pinecone (20 candidats)
                    3. reranking : cross-encoder bge-reranker-v2-m3 ─► 3 meilleurs + seuil de pertinence
                    4. élargissement aux passages voisins
                    5. génération en streaming : gpt-oss-120b (Groq), avec citations
                                                 │
                                         traces ─► Langfuse
```

| Composant | Choix | Pourquoi |
|---|---|---|
| API | FastAPI, Pydantic, streaming SSE | validation des entrées, documentation `/docs`, réponse affichée au fil de l'eau |
| Orchestration | LangChain (LCEL) | chaîne composable : reformulation → recherche → génération |
| Embeddings | `intfloat/multilingual-e5-base` (local) | multilingue, gratuit, sans quota |
| Base vectorielle | Pinecone serverless | index managé, offre gratuite |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) | cross-encoder multilingue : le plus gros gain mesuré |
| LLM | `openai/gpt-oss-120b` via Groq, température 0 | réponses déterministes, évaluations comparables |
| Évaluation | LLM juge `qwen3.8-27b` (autre famille que le modèle évalué) | métriques inspirées de RAGAS, sans biais d'auto-évaluation |
| Observabilité | Langfuse | trace de chaque question : reformulation, passages et scores, prompt, tokens, latence |
| Interface | Chainlit | client de l'API (aucune logique RAG côté client) |

## Démarrage rapide

**Avec Docker** (recommandé) :

```bash
cp serveur/.env.example serveur/.env    # puis renseigner les clés (Groq, Pinecone, Langfuse facultatif)
docker compose up --build               # interface : http://localhost:8001 — API : http://localhost:8000/docs
```

**Sans Docker** (Python 3.12, [uv](https://docs.astral.sh/uv/)) :

```bash
uv venv && uv pip install -r serveur/requirements.txt -r client/requirements.txt
python scripts/download_corpus.py       # télécharge les 10 guides (data/sources.csv)
python scripts/index_corpus.py          # nettoyage, découpage, embeddings, indexation dans Pinecone
cd serveur && uvicorn main:app          # API sur le port 8000
cd client && chainlit run app.py --port 8001
```

**Évaluer** :

```bash
python scripts/eval_retrieval.py        # recherche : hit@k, MRR@k (sans LLM, environ 1 min)
python scripts/eval_generation.py       # réponses : LLM juge (environ 30 min, quota Groq)
pytest                                  # 27 tests unitaires, sans clé d'API
```

**Docker et carte graphique** : l'image tourne sur processeur (aucune carte graphique dans le conteneur). Le reranker
y est nettement plus lent : environ 10 s par question avec 10 passages reclassés, contre environ 4,5 s (20 passages)
en lançant l'API directement sur une machine équipée d'une carte graphique NVIDIA.

## Qualité et production

- **Tests** : nettoyage des PDF, retriever (reranking, seuil, passages voisins), citations, validation des requêtes.
- **CI** (GitHub Actions) : lint et tests à chaque push ; **barrière d'évaluation** : la CI échoue si le hit@3
  passe sous 80 % sur le vrai index.
- **Sécurité** : clés hors du code ; ajout de documents protégé par jeton d'administration ; messages d'erreur
  génériques côté client ; noms de fichiers assainis ; entrées validées (longueur, rôles de l'historique).
- **Serveur sans état** : l'historique de conversation est envoyé par le client ; un `session_id` regroupe les
  questions d'une conversation dans Langfuse.

## Limites connues

- **Tableaux** : l'extraction de texte des PDF aplatit les tableaux (une question sur la fréquence des dosages
  selon le stade de la maladie rénale échoue encore).
- **Seuil de pertinence** choisi sur le jeu de test lui-même, avec une marge étroite : un jeu de validation séparé
  serait nécessaire pour le confirmer. Plus généralement, 43 questions ne suffisent pas à départager des variantes
  proches (une question = 2 à 3 points de pourcentage).
- **Latence** : environ 4,5 s par question sur une petite carte graphique (le reranker représente 2 s) ; sur CPU
  (conteneur Docker), 10 candidats sont reclassés au lieu de 20.
- **Variabilité** : malgré la température 0, une même question peut recevoir une réponse légèrement différente d'une
  exécution à l'autre (ex. une question de patient sur les siestes, tantôt répondue, tantôt refusée).
- **Corpus** limité à 10 guides ; pas de lecture des images des PDF.

## Corpus

10 documents publics (HAS 2014–2025, INCa 2018–2025) : diabète de type 2, hypertension artérielle, BPCO,
insuffisance cardiaque, maladie rénale chronique, chimiothérapie, fatigue et cancer, mélanomes.
Liste et liens dans [`data/sources.csv`](data/sources.csv) ; les PDF ne sont pas redistribués dans ce dépôt.
