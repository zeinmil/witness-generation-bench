"""
config.py

tout ce qui dépend du modèle utilisé est centralisé ici. si tu veux tester
un autre modèle que phi-2 (par exemple un modèle plus gros, sur une machine
plus puissante), c'est le SEUL fichier à modifier — pas la peine de toucher
au code de benchmark/ ou rag/.

MODEL_NAME : le modèle huggingface à charger (générateur)
MAX_CONTEXT_TOKENS : la fenêtre de contexte totale du modèle (entrée + sortie)
MAX_NEW_TOKENS : le nombre de tokens qu'on autorise le modèle à générer
DEVICE : "cuda" si un gpu est disponible, sinon "cpu" (détecté automatiquement)
"""
import torch

MODEL_NAME = "microsoft/phi-2"

# fenêtre de contexte du modèle choisi. phi-2 = 2048. si tu changes de modèle,
# mets ici sa vraie fenêtre de contexte (regarde la fiche du modèle sur
# huggingface, champ "max_position_embeddings" ou "context length")
MAX_CONTEXT_TOKENS = 2048

# nombre de tokens qu'on laisse le modèle générer pour produire une instance
MAX_NEW_TOKENS = 200

# marge de sécurité pour laisser de la place au delimiteur "Instance:" etc.
MARGE_SECURITE = 48

# la limite d'entrée est calculée automatiquement à partir des deux valeurs
# ci-dessus : pas la peine de changer ça à la main, il suffit d'ajuster
# MAX_CONTEXT_TOKENS et MAX_NEW_TOKENS pour un nouveau modèle
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS - MARGE_SECURITE

# détection automatique gpu/cpu, pas besoin d'y toucher
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
