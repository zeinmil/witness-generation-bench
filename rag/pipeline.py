"""
rag/pipeline.py

le pipeline RAG complet : retrieval -> prompt -> génération (phi-2,
déterministe) -> extraction du json -> validation.

    [schéma cible]
          |
      RETRIEVER   (cherche les k schémas les plus proches dans la base vectorielle)
          |
      PROMPT      (instruction + exemples récupérés + schéma cible)
          |
      Phi-2       (génération déterministe, do_sample=False)
          |
      EXTRACTION  (isole le premier objet json complet dans la sortie brute)
          |
      VALIDATION  (jsonschema.validate contre le schéma cible)
"""
import os
import sys
import json

import torch
import jsonschema

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_NEW_TOKENS
from prompts import get_prompt_template


def recuperer_exemples(schema_cible_texte, base_vectorielle, k=1, exclure_identique=True):
    """
    cherche les k schémas les plus proches du schéma cible dans la base vectorielle.

    on exclut le schéma cible lui-même s'il apparaît dans ses propres voisins,
    sinon le modèle se contente de recopier la réponse au lieu de généraliser.
    """
    resultats = base_vectorielle.similarity_search(schema_cible_texte, k=k + 1)
    exemples = []
    for doc in resultats:
        if exclure_identique and doc.page_content.strip() == schema_cible_texte.strip():
            continue
        exemples.append({
            "nom_schema": doc.metadata["nom_schema"],
            "schema": doc.page_content,
            "instance": doc.metadata["instance"],
        })
    return exemples[:k]


def formater_exemples(exemples):
    """transforme les exemples récupérés (objets python) en texte lisible pour le prompt"""
    blocs = [f"Schéma:\n{ex['schema']}\nInstance:\n{ex['instance']}\n" for ex in exemples]
    return "\n".join(blocs)


def extraire_premier_json(texte):
    """isole le premier objet json {...} complet dans un texte (voir benchmark/generation.py)"""
    d = texte.find("{")
    if d == -1:
        return None
    profondeur, en_chaine, echap = 0, False, False
    for i in range(d, len(texte)):
        c = texte[i]
        if en_chaine:
            if echap:
                echap = False
            elif c == "\\":
                echap = True
            elif c == '"':
                en_chaine = False
        else:
            if c == '"':
                en_chaine = True
            elif c == "{":
                profondeur += 1
            elif c == "}":
                profondeur -= 1
                if profondeur == 0:
                    return texte[d:i + 1]
    return None


def generer_instance(model, tokenizer, prompt_texte, max_new_tokens=None):
    """
    appelle phi-2 en génération déterministe (do_sample=False, num_beams=1) :
    à chaque étape, le modèle choisit toujours le token le plus probable.

    pas de créativité recherchée ici, cohérent avec l'usage visé (génération
    de données structurées, pas de texte créatif).
    """
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    inputs = tokenizer(prompt_texte, return_tensors="pt").to(model.device)
    n_tok = inputs["input_ids"].shape[1]
    with torch.no_grad():
        sortie = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(sortie[0][n_tok:], skip_special_tokens=True)


def valider_instance(instance_texte, schema_texte):
    try:
        instance = json.loads(instance_texte)
    except json.JSONDecodeError as e:
        return False, f"json mal formé : {e}"
    try:
        schema = json.loads(schema_texte)
        jsonschema.validate(instance, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"invalide : {e.message}"
    except Exception as e:
        return False, f"erreur : {type(e).__name__}"


def pipeline_rag(model, tokenizer, nom_schema_cible, schema_cible_texte, base_vectorielle,
                  k=1, prompt_nom="neutre", max_new_tokens=None):
    """exécute le pipeline complet pour un seul schéma et renvoie le résultat détaillé"""
    exemples = recuperer_exemples(schema_cible_texte, base_vectorielle, k=k)
    prompt_template = get_prompt_template(prompt_nom)
    prompt_final = prompt_template.format(
        exemples_formates=formater_exemples(exemples),
        schema_cible=schema_cible_texte,
    )
    brut = generer_instance(model, tokenizer, prompt_final, max_new_tokens=max_new_tokens)
    candidat = extraire_premier_json(brut)

    if candidat is None:
        return {"nom_schema": nom_schema_cible, "valide": False,
                "erreur": "tronqué/aucun json", "instance": None}

    valide, erreur = valider_instance(candidat, schema_cible_texte)
    return {"nom_schema": nom_schema_cible, "valide": valide, "erreur": erreur, "instance": candidat}
