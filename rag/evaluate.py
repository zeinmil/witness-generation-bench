"""
rag/evaluate.py

harnais d'évaluation comparative : fait tourner le pipeline RAG sur un
échantillon fixe de schémas, pour comparer honnêtement différents réglages
d'hyperparamètres (embedding, k, prompt) entre eux et par rapport au
zero-shot pur (sans RAG).

résultats obtenus sur un échantillon de 20 schémas identiques :

    configuration                          | score
    ---------------------------------------|----------
    sans RAG (zero-shot pur)               | 1/20 (5%)
    RAG k=3, MiniLM                        | 6/20 (30%)
    RAG k=1, MiniLM                        | 11/20 (55%)
    RAG k=1, CodeBERT                      | 11/20 (55%)
    ensemble k=[1,2], MiniLM               | 11/20 (55%)
    ensemble k=[1,2], CodeBERT             | 12/20 (60%)  <- meilleure config
    prompt insistant sur le not            | 11/20 (55%)  <- aucun gain
    prompt complet (toutes contraintes)    |  8/20 (40%)  <- moins bon

enseignements principaux :
    - le RAG améliore massivement par rapport au zero-shot pur (x10)
    - k=1 est nettement meilleur que k=2/k=3 : phi-2 a une fenêtre de contexte
      limitée à 2048 tokens, donc plus d'exemples = moins de place pour que
      le modèle termine sa génération (d'où plus de troncatures)
    - CodeBERT (orienté code/structure) fait un peu mieux que MiniLM
      (généraliste) à k et prompt égaux
    - modifier le prompt n'aide pas, et peut nuire si le prompt devient trop
      long (moins de place dans la fenêtre de contexte)
    - sur l'ensemble des runs, seulement ~35% des échecs sont de vraies
      violations du "not" : les deux tiers restants sont d'autres erreurs
      (type, enum, oneOf), indépendantes du sujet d'étude initial
"""
from langchain_core.prompts import PromptTemplate

from pipeline import pipeline_rag, generer_instance, extraire_premier_json, valider_instance


def evaluer_configuration(model, tokenizer, base_vectorielle, noms, textes_schemas,
                           nom_config, k=1, prompt_nom="neutre", n_echantillon=20):
    """évalue une configuration (embedding + k + prompt) sur un échantillon de schémas"""
    resultats = []
    for nom, schema_texte in list(zip(noms, textes_schemas))[:n_echantillon]:
        r = pipeline_rag(model, tokenizer, nom, schema_texte, base_vectorielle, k=k, prompt_nom=prompt_nom)
        resultats.append(r)
        print(f"{nom:40s} -> {'VALIDE' if r['valide'] else r['erreur']}")
    n_valides = sum(r["valide"] for r in resultats)
    n_tronques = sum(1 for r in resultats if r["erreur"] == "tronqué/aucun json")
    print(f"\n=== {nom_config} : {n_valides}/{len(resultats)} valides | {n_tronques} tronqués ===")
    return resultats


def evaluer_sans_rag(model, tokenizer, noms, textes_schemas, n_echantillon=20, max_new_tokens=200):
    """référence de comparaison : génération directe, sans aucun exemple récupéré"""
    prompt_simple = PromptTemplate(
        input_variables=["schema_cible"],
        template=(
            "Tu es un générateur spécialisé en JSON. Génère UNE SEULE instance JSON "
            "qui satisfait strictement ce schéma JSON Schema.\n\nSchéma :\n{schema_cible}\n\n"
            "Réponds UNIQUEMENT avec l'instance JSON, rien d'autre.\n\nInstance :\n"
        ),
    )
    resultats = []
    for nom, schema_texte in list(zip(noms, textes_schemas))[:n_echantillon]:
        prompt_final = prompt_simple.format(schema_cible=schema_texte)
        brut = generer_instance(model, tokenizer, prompt_final, max_new_tokens=max_new_tokens)
        candidat = extraire_premier_json(brut)
        if candidat is None:
            resultats.append({"nom_schema": nom, "valide": False, "erreur": "tronqué/aucun json"})
        else:
            valide, erreur = valider_instance(candidat, schema_texte)
            resultats.append({"nom_schema": nom, "valide": valide, "erreur": erreur})
        print(f"{nom:40s} -> {'VALIDE' if resultats[-1]['valide'] else resultats[-1]['erreur']}")
    n_valides = sum(r["valide"] for r in resultats)
    print(f"\n=== sans RAG (zero-shot pur) : {n_valides}/{len(resultats)} valides ===")
    return resultats


def categoriser_erreur(resultat, detail):
    """
    classe un résultat d'échec en 3 catégories, pour distinguer les vraies
    violations du "not" des autres types d'erreurs (type, enum, oneOf...)
    """
    if resultat == "VALIDE":
        return "valide"
    if resultat in ("TRONQUE", "NON_TESTE"):
        return "tronque_ou_non_teste"
    if detail and "should not be valid under" in detail:
        return "violation_not"
    return "autre_erreur"
