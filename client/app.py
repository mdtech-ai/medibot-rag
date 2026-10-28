"""
Client Chainlit de MediBot : interface de chat qui consomme l'API FastAPI (serveur/).

Le client ne contient aucune logique RAG : il envoie la question, l'historique de la conversation et un
identifiant de session à la route /ask/stream, puis affiche la réponse token par token et les sources citées.

Lancement (l'API doit tourner sur le port 8000 ; Chainlit lit sa configuration dans client/.chainlit/) :
    cd client
    chainlit run app.py --port 8001 -w   # -w : rechargement automatique quand le code change
"""
import json
import os

import chainlit as cl
import httpx

API_URL = os.getenv("MEDIBOT_API_URL", "http://127.0.0.1:8000")
MAX_HISTORY_MESSAGES = 10  # Le serveur n'en utilise que les 6 derniers ; inutile d'en envoyer davantage

# Pas de message de bienvenue : Chainlit n'affiche les questions d'exemple que tant que la conversation est vide.
# L'avertissement médical est dans la zone de saisie (.chainlit/translations/fr-FR.json) et dans chainlit.md.


@cl.set_starters
async def set_starters():
    """Exemples de questions proposés sur l'écran d'accueil."""
    return [
        cl.Starter(label="Diabète de type 2", message="Quels sont les objectifs de glycémie pour un diabétique de type 2 ?"),
        cl.Starter(label="Hypertension", message="Comment mesurer sa tension artérielle à domicile ?"),
        cl.Starter(label="BPCO", message="Quels sont les symptômes de la BPCO ?"),
        cl.Starter(label="Chimiothérapie", message="Comment se déroule une cure de chimiothérapie ?"),
    ]


@cl.on_chat_start
async def on_chat_start():
    # Historique propre à chaque conversation, conservé côté client (le serveur est sans état)
    cl.user_session.set("history", [])


async def read_sse(response: httpx.Response):
    """Lit un flux Server-Sent Events et renvoie chaque événement sous la forme (nom, données JSON)."""
    event = None
    async for line in response.aiter_lines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            yield event, json.loads(line[len("data: "):])


def format_sources(sources: list[dict]) -> tuple[str, list[cl.Text]]:
    """Construit la liste des sources affichée sous la réponse, et un élément cliquable par source (extrait)."""
    lines, elements = [], []
    for i, source in enumerate(sources, start=1):
        name = f"Source {i}"
        page = f", page {source['page']}" if source.get("page") else ""
        lines.append(f"- {name} : {source['file']}{page}")
        # Le nom de l'élément apparaît dans le texte : Chainlit le rend cliquable et affiche l'extrait dans le panneau latéral
        elements.append(cl.Text(name=name, content=f"{source['file']}{page}\n\n« {source['excerpt']} »", display="side"))
    return "\n\n**Sources**\n" + "\n".join(lines), elements


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    payload = {
        "question": message.content,
        "history": history[-MAX_HISTORY_MESSAGES:],
        # Identifiant de la conversation Chainlit : regroupe toutes ses questions dans une même session Langfuse
        "session_id": cl.context.session.id,
    }

    answer = cl.Message(content="")
    sources = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            async with client.stream("POST", f"{API_URL}/ask/stream", json=payload) as response:
                response.raise_for_status()
                async for event, data in read_sse(response):
                    if event == "sources":
                        sources = data
                    elif event == "token":
                        await answer.stream_token(data["text"])
                    elif event == "error":
                        await answer.stream_token(f"\n\n❌ {data['error']}")
    except httpx.ConnectError:
        await cl.ErrorMessage(content=f"L'API MediBot est injoignable ({API_URL}). Le serveur est-il lancé ?").send()
        return
    except httpx.HTTPError as e:
        await cl.ErrorMessage(content=f"Erreur lors de l'appel à l'API : {e}").send()
        return

    # Les sources ne sont affichées que si MediBot a réellement répondu (pas après un refus)
    if sources and "je n'ai pas trouvé" not in answer.content.lower():
        sources_text, answer.elements = format_sources(sources)
        await answer.stream_token(sources_text)
    await answer.send()

    # L'historique conserve la réponse sans la liste des sources : elle n'aide pas la reformulation des questions de suivi
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": answer.content.split("\n\n**Sources**")[0]})
