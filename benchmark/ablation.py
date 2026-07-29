"""
benchmark/ablation.py

étude d'ablation : pour chaque schéma contenant "not", on compare la réussite
de la génération avec et sans ce mot-clé, pour isoler son effet propre
(indépendamment de la taille ou de la complexité générale du schéma).

un schéma est "bloqué par le not" si le modèle réussit sur la version SANS
"not", mais que l'instance produite violerait le "not" de la version originale.
"""
import os
import copy
import json

import torch
import jsonschema

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation import (
    contient_not, extraire_premier_json, construire_prompt,
    charger_modele, INSTRUCTION,
)


def retirer_not(obj):
    """
    renvoie une copie profonde du schéma dont toutes les clés "not" ont été
    retirées, à n'importe quel niveau d'imbrication. ne touche jamais à
    l'original (copie profonde).
    """
    def _rec(o):
        if isinstance(o, dict):
            return {k: _rec(v) for k, v in o.items() if k != "not"}
        if isinstance(o, list):
            return [_rec(x) for x in o]
        return o
    return _rec(copy.deepcopy(obj))


def generer_instance_brute(model, tokenizer, schema_text, max_input_tokens=1800, max_new_tokens=200):
    prompt = INSTRUCTION + f"\nSchema:\n{schema_text}\nInstance:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tok = inputs["input_ids"].shape[1]
    if n_tok > max_input_tokens:
        return None, "overflow"
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    genere = tokenizer.decode(out[0][n_tok:], skip_special_tokens=True)
    cand = extraire_premier_json(genere)
    if cand is None:
        return None, "tronqué"
    try:
        return json.loads(cand), None
    except json.JSONDecodeError:
        return None, "mal formé"


def valide_contre(instance, schema):
    if instance is None:
        return False
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


def ablation_collection(model, tokenizer, data_dir, collection):
    base = os.path.join(data_dir, collection)
    schemas = sorted(
        n for n in os.listdir(base)
        if n.endswith(".json") and contient_not(json.load(open(os.path.join(base, n))))
    )

    resultats = []
    for fichier in schemas:
        schema_avec = json.load(open(os.path.join(base, fichier)))
        schema_sans = retirer_not(schema_avec)

        inst_avec, _ = generer_instance_brute(model, tokenizer, json.dumps(schema_avec))
        inst_sans, _ = generer_instance_brute(model, tokenizer, json.dumps(schema_sans))

        valide_avec = valide_contre(inst_avec, schema_avec)
        valide_sans = valide_contre(inst_sans, schema_sans)
        sans_satisfait_avec = valide_contre(inst_sans, schema_avec) if inst_sans is not None else None

        resultats.append({
            "collection": collection,
            "fichier": fichier,
            "valide_avec_not": valide_avec,
            "valide_sans_not": valide_sans,
            "sans_satisfait_avec": sans_satisfait_avec,
            # "bloqué par le not" : réussit sans la contrainte, mais la violerait avec
            "bloque_par_not": bool(valide_sans and sans_satisfait_avec is False),
        })
    return resultats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/content/jsonschemabench/data/")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    model, tokenizer = charger_modele()
    collections = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))

    for collection in collections:
        sortie = os.path.join(args.out_dir, f"ablation_{collection}.json")
        if os.path.exists(sortie):
            print(f"[déjà fait] {collection}")
            continue
        resultats = ablation_collection(model, tokenizer, args.data_dir, collection)
        json.dump(resultats, open(sortie, "w"), indent=2, ensure_ascii=False)
        if resultats:
            n_avec = sum(r["valide_avec_not"] for r in resultats)
            n_sans = sum(r["valide_sans_not"] for r in resultats)
            n_bloques = sum(r["bloque_par_not"] for r in resultats)
            print(f"{collection:18s} avec:{n_avec}/{len(resultats)}  sans:{n_sans}/{len(resultats)}  bloqués:{n_bloques}")


if __name__ == "__main__":
    main()
