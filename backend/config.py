"""
Configuration for the LLM Council.

Ce fichier centralise toute la configuration liée :
- aux endpoints Ollama (Council / Chairman)
- aux modèles utilisés
- à la gestion de la concurrence et de la stabilité
"""

import os
from dotenv import load_dotenv

# Charge les variables d'environnement depuis un fichier .env
load_dotenv()

# Le conseil reste sur mon PC
COUNCIL_BASE_URL = os.getenv("COUNCIL_BASE_URL", "http://localhost:11434")

# Chairman tourne sur le pc de l'autre membre du binôme. On utilise l'IP du second ordinateur
CHAIRMAN_BASE_URL = os.getenv("CHAIRMAN_BASE_URL", "http://172.20.10.4:11434")

# Liste des modèles composant le Council
# Chaque modèle peut être surchargé via les variables d'environnement

COUNCIL_MODELS = [
    os.getenv("COUNCIL_MODEL_1", "llama3"),
    os.getenv("COUNCIL_MODEL_2", "mistral"),
    os.getenv("COUNCIL_MODEL_3", "qwen2.5"),
]

# Modèle utilisé par le Chairman pour la synthèse / décision finale
CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "llama3")

#Répertoire où sont sauvegardées les conversations
DATA_DIR = "data/conversations"

# Nombre maximum de requêtes parallèles envoyées aux modèles
# 1 = mode safe
MAX_PARALLEL_REQUESTS = int(os.getenv("MAX_PARALLEL_REQUESTS", "1"))  # 1 = safe

# Nombre de tentatives en cas d'échec d'une requête
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "1"))  # 0/1/2

# Temps d'attente (en secondes) entre deux tentatives
RETRY_SLEEP_SECONDS = float(os.getenv("RETRY_SLEEP_SECONDS", "0.8"))
