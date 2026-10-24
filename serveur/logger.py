import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Répertoire des journaux : serveur/logs (indépendant du dossier depuis lequel le serveur est lancé)
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "medicalai_assistant.log"

# Configuration du logger : Module de journalisation pour l'application MedicalAIAssistant
def setup_logger(name="MedicalAIAssistant"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s:%(module)s:%(lineno)d] --- [%(message)s]")

    # Handler console : affichage des journaux dans le terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # Handler fichier : enregistrement des journaux dans serveur/logs/
    # Rotation : quand le fichier atteint 5 Mo, il est archivé (.log.1, .log.2, ...) et on garde les 5 derniers
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


logger = setup_logger()
