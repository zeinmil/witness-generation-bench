"""
benchmark/generation.py

génération zero-shot et few-shot d'instances json par le modèle configuré
dans config.py, sur un corpus de schémas json schema contenant le mot-clé "not".

usage :
    python generation.py --collection Github_medium --condition zero-shot

produit un fichier resultats_generation_<collection>.json avec, pour chaque
schéma : l'instance générée, sa validité, et la raison d'échec le cas échéant.
"""
import os
import sys
import json
import argparse

import torch
import jsonschema
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, DEVICE, DTYPE


INSTRUCTION = (
    "You generate JSON data. Given a JSON Schema, output ONE JSON instance "
    "that satisfies it.\nOutput only the JSON instance. Do not repeat the schema.\n"
)

# exemples few-shot fixes, choisis pour être neutres (pas d'amorçage des schémas testés)
EXEMPLES_FEWSHOT = [
    ('{"type":"object","not":{"required":["deprecated"]},"properties":{"title":{"type":"string"}}}',
     '{"title":"hello"}'),
    ('{"type":"array","not":{"contains":{"const":"zzz"}}}',
     '["alpha","beta"]'),
    ('{"type":"string","not":{"pattern":"^xx_"}}',
     '"kappa"'),
]


def contient_not(obj):
    """recherche récursive du mot-clé 'not' à n'importe quel niveau du schéma"""
    if isinstance(obj, dict):
        return "not" in obj or any(contient_not(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contient_not(x) for x in obj)
    return False


def extraire_premier_json(texte):
    """
    extrait le premier objet json {...} complet d'un texte généré par le modèle.

    on suit si on est dans une chaîne de caractères ou pas, sinon une accolade
    présente dans une valeur texte (genre "un ensemble de {clés}") fausserait
    le comptage. renvoie None si aucun objet complet n'est trouvé (génération
    tronquée).
    """
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


def construire_prompt(schema_text, condition):
    if condition == "zero-shot":
        return INSTRUCTION + f"\nSchema:\n{schema_text}\nInstance:\n"
    prompt = INSTRUCTION
    for sch, inst in EXEMPLES_FEWSHOT:
        prompt += f"\nSchema:\n{sch}\nInstance:\n{inst}\n"
    return prompt + f"\nSchema:\n{schema_text}\nInstance:\n"


def charger_modele():
    """
    charge le modèle défini dans config.py (MODEL_NAME), avec détection
    automatique gpu/cpu. pour changer de modèle : modifier config.py,
    pas ce fichier.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    print(f"{MODEL_NAME} chargé sur {model.device}")
    return model, tokenizer


def generer_et_valider(model, tokenizer, schema, condition, max_input_tokens=None, max_new_tokens=None):
    max_input_tokens = max_input_tokens or MAX_INPUT_TOKENS
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS

    schema_text = json.dumps(schema)
    prompt = construire_prompt(schema_text, condition)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tok = inputs["input_ids"].shape[1]
    if n_tok > max_input_tokens:
        # le schéma est déjà trop gros pour la fenêtre de contexte, pas la peine de générer
        return {"instance": None, "valide": False, "erreur": f"overflow ({n_tok} tokens)"}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    genere = tokenizer.decode(out[0][n_tok:], skip_special_tokens=True)

    candidat = extraire_premier_json(genere)
    if candidat is None:
        return {"instance": genere, "valide": False, "erreur": "tronqué / aucun objet json complet"}

    try:
        instance = json.loads(candidat)
        jsonschema.validate(instance, schema)
        return {"instance": instance, "valide": True, "erreur": None}
    except json.JSONDecodeError as e:
        return {"instance": candidat, "valide": False, "erreur": f"json mal formé : {e}"}
    except jsonschema.ValidationError as e:
        return {"instance": instance, "valide": False, "erreur": f"invalide : {e.message}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/content/jsonschemabench/data/")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--condition", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = os.path.join(args.data_dir, args.collection)
    schemas = sorted(
        n for n in os.listdir(base)
        if n.endswith(".json") and contient_not(json.load(open(os.path.join(base, n))))
    )
    print(f"{len(schemas)} schémas 'not' dans {args.collection}")

    model, tokenizer = charger_modele()

    resultats = []
    for fichier in schemas:
        schema = json.load(open(os.path.join(base, fichier)))
        r = generer_et_valider(model, tokenizer, schema, args.condition)
        r["fichier"] = fichier
        resultats.append(r)
        print(f"{fichier:30s} -> {'VALIDE' if r['valide'] else r['erreur']}")

    n_valides = sum(r["valide"] for r in resultats)
    print(f"\nscore {args.condition} : {n_valides}/{len(schemas)}")

    out = args.out or f"resultats_generation_{args.collection}_{args.condition}.json"
    json.dump(resultats, open(out, "w"), indent=2, ensure_ascii=False, default=str)
    print(f"résultats sauvegardés dans {out}")


if __name__ == "__main__":
    main()
