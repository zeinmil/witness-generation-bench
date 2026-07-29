"""
rag/ensemble.py

stratégie "ensemble" : au lieu de fixer une seule valeur de k, on essaie
plusieurs valeurs successivement pour un même schéma, et on garde la première
instance valide trouvée.

attention méthodologique : cette stratégie mélange l'effet de plusieurs
valeurs de k dans une seule mesure. elle répond à la question "une stratégie
de secours améliore-t-elle le taux de réussite final ?", pas à la question
"quel k est le meilleur isolément ?" (pour ça, voir evaluate.py, qui teste
chaque k séparément).

résultat obtenu (CodeBERT, échantillon de 20 schémas) :
    - k=1 seul                : 11/20 (55%)
    - ensemble k=[1, 2]       : 12/20 (60%)  -- gain marginal (+1 schéma)
"""
from pipeline import pipeline_rag


def pipeline_ensemble(model, tokenizer, nom_schema, schema_texte, base_vectorielle, valeurs_k=(1, 2)):
    """essaie plusieurs valeurs de k dans l'ordre, garde la première instance valide trouvée"""
    r = None
    for k in valeurs_k:
        r = pipeline_rag(model, tokenizer, nom_schema, schema_texte, base_vectorielle, k=k)
        if r["valide"]:
            r["k_utilise"] = k
            return r
    r["k_utilise"] = valeurs_k[-1]
    return r


def evaluer_ensemble(model, tokenizer, base_vectorielle, noms, textes_schemas, valeurs_k=(1, 2), n_echantillon=20):
    resultats = []
    for nom, schema_texte in list(zip(noms, textes_schemas))[:n_echantillon]:
        r = pipeline_ensemble(model, tokenizer, nom, schema_texte, base_vectorielle, valeurs_k)
        resultats.append(r)
        etiquette = f"VALIDE (k={r['k_utilise']})" if r["valide"] else r["erreur"]
        print(f"{nom:40s} -> {etiquette}")
    n_valides = sum(r["valide"] for r in resultats)
    print(f"\n=== ensemble k={list(valeurs_k)} : {n_valides}/{len(resultats)} valides ===")
    return resultats
