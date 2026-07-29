"""
rag/prompts.py

les différentes formulations de prompt testées pour la génération RAG.
chaque variante isole une hypothèse différente sur ce qui pourrait améliorer
la génération de phi-2.

résultats obtenus (k=1, embedding CodeBERT, échantillon de 20 schémas) :
    - PROMPT_NEUTRE            : 11/20 (55%)  -- référence
    - PROMPT_INSISTANT_NOT     : 11/20 (55%)  -- aucun changement
    - PROMPT_COMPLET           :  8/20 (40%)  -- moins bon (prompt trop long)

conclusion : allonger le prompt avec des instructions supplémentaires ne
suffit pas à améliorer un petit modèle comme phi-2, et peut même nuire en
consommant de la place dans sa fenêtre de contexte (2048 tokens).
"""
from langchain_core.prompts import PromptTemplate


PROMPT_NEUTRE = """Tu es un générateur spécialisé en JSON. Ta tâche est de produire UNE SEULE instance JSON qui satisfait strictement le schéma JSON Schema fourni à la fin.

Voici des exemples de paires (schéma, instance valide) pour t'aider à comprendre le format attendu :

{exemples_formates}

Maintenant, génère une instance JSON valide pour CE schéma :

Schéma :
{schema_cible}

Réponds UNIQUEMENT avec l'instance JSON, sans aucune explication, sans balises de code, rien d'autre que le JSON.

Instance :
"""

PROMPT_INSISTANT_NOT = """Tu es un générateur spécialisé en JSON. Ta tâche est de produire UNE SEULE instance JSON qui satisfait strictement le schéma JSON Schema fourni à la fin.

ATTENTION : ce schéma contient une ou plusieurs contraintes "not", qui interdisent certaines valeurs. Fais bien attention à NE PAS produire les valeurs interdites par ces contraintes.

Voici des exemples de paires (schéma, instance valide) pour t'aider à comprendre le format attendu :

{exemples_formates}

Maintenant, génère une instance JSON valide pour CE schéma :

Schéma :
{schema_cible}

Réponds UNIQUEMENT avec l'instance JSON, sans aucune explication, sans balises de code, rien d'autre que le JSON.

Instance :
"""

PROMPT_COMPLET = """Tu es un générateur spécialisé en JSON. Ta tâche est de produire UNE SEULE instance JSON qui satisfait STRICTEMENT le schéma JSON Schema fourni à la fin.

Avant de répondre, vérifie attentivement chacun de ces points dans le schéma :
- "required" : toutes les propriétés listées doivent être présentes dans ton instance.
- "type" : chaque valeur doit avoir exactement le type attendu (object, array, string, integer, number, boolean).
- "enum" : si une propriété a un "enum", ta valeur doit être L'UNE des valeurs listées, jamais une autre.
- "oneOf" : ton instance doit satisfaire EXACTEMENT UNE des branches proposées, pas zéro, pas plusieurs.
- "not" : si une propriété a une contrainte "not", ta valeur ne doit JAMAIS correspondre à ce qui est interdit.

Voici des exemples de paires (schéma, instance valide) pour t'aider à comprendre le format attendu :

{exemples_formates}

Maintenant, génère une instance JSON valide pour CE schéma, en vérifiant chaque point ci-dessus :

Schéma :
{schema_cible}

Réponds UNIQUEMENT avec l'instance JSON, sans aucune explication, sans balises de code, rien d'autre que le JSON.

Instance :
"""


def get_prompt_template(nom="neutre"):
    """renvoie un PromptTemplate langchain prêt à l'emploi pour la variante demandée"""
    templates = {
        "neutre": PROMPT_NEUTRE,
        "insistant_not": PROMPT_INSISTANT_NOT,
        "complet": PROMPT_COMPLET,
    }
    return PromptTemplate(
        input_variables=["exemples_formates", "schema_cible"],
        template=templates[nom],
    )
