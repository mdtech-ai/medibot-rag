#!/usr/bin/env bash
# Démarrage tout-en-un (hébergement à conteneur unique, ex. Cloud Run) : l'API écoute en interne sur le port 8000,
# le client Chainlit est exposé sur le port public (7860 par défaut) et appelle l'API en local.
set -euo pipefail

cd /app/serveur
# Sur CPU (sans carte graphique) : 10 passages reclassés au lieu de 20, pour une latence acceptable
export RERANKER_FETCH_K="${RERANKER_FETCH_K:-10}"
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Attente de l'API : le chargement des modèles prend plusieurs dizaines de secondes sur CPU
python - <<'EOF'
import time, urllib.request
for _ in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        print("API prête")
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("L'API n'a pas démarré")
EOF

cd /app/client
export MEDIBOT_API_URL=http://127.0.0.1:8000
exec chainlit run app.py --host 0.0.0.0 --port "${PORT:-7860}" --headless
