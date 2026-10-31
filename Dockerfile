# Image unique de MediBot : API FastAPI (serveur/) + client Chainlit (client/)
# - en local : docker compose lance deux conteneurs à partir de cette image (voir docker-compose.yml)
# - sur Hugging Face Spaces : docker/start.sh lance les deux processus dans un seul conteneur (port 7860)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Les modèles Hugging Face sont stockés dans l'image (téléchargés une fois, à la construction)
    HF_HOME=/app/.cache/huggingface

# Utilisateur non root (bonne pratique de sécurité, et exigence de Hugging Face Spaces : identifiant 1000)
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# 1. Dépendances (couche mise en cache tant que les fichiers requirements ne changent pas)
#    PyTorch en version CPU : environ 5 fois plus léger que la version CUDA, inutile sans carte graphique
COPY serveur/requirements.txt serveur/requirements.txt
COPY client/requirements.txt client/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r serveur/requirements.txt -r client/requirements.txt

# 2. Modèles locaux téléchargés à la construction : le conteneur démarre sans télécharger plusieurs Go
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-base'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

# 3. Code de l'application (dernière couche : une modification du code ne réinstalle rien)
COPY serveur/ serveur/
COPY client/ client/
COPY docker/start.sh start.sh
RUN chown -R user:user /app

USER user
EXPOSE 7860
CMD ["bash", "start.sh"]
