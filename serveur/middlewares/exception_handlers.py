from fastapi import Request
from fastapi.responses import JSONResponse
from logger import logger


async def catch_exception_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.exception(f"Exception non gérée : {e}")
        # Message générique pour le client : le détail (chemins, services, extraits de requête) reste dans les journaux
        return JSONResponse(
            status_code=500,
            content={"error": "Erreur interne du serveur."},
        )
