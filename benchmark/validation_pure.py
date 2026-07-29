"""
benchmark/validation_pure.py

test complémentaire à la génération : au lieu de demander à phi-2 de PRODUIRE
une instance, on lui donne un schéma + une valeur (valide ou invalide au
regard d'un "not"), et on lui demande de juger : "valid" ou "invalid".

ce test élimine toute question de taille de schéma ou de mise en forme json :
il mesure directement la capacité du modèle à reconnaître une violation
explicite de la négation.
"""
import json
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation import charger_modele

INSTRUCTION = (
    "You are a JSON Schema validator. Given a SCHEMA and a VALUE, decide if the "
    "value satisfies the schema.\nAnswer with exactly one word: 'valid' or 'invalid'.\n"
)

# 10 cas synthétiques couvrant des formes variées de "not", avec vérité-terrain
SUITE = [
    {"id": "not_enum_string", "schema": {"type": "string", "not": {"enum": ["rouge"]}},
     "valide": "bleu", "invalide": "rouge"},
    {"id": "not_type_number", "schema": {"not": {"type": "number"}},
     "valide": "texte", "invalide": 42},
    {"id": "not_const", "schema": {"not": {"const": 0}},
     "valide": 5, "invalide": 0},
    {"id": "not_pattern", "schema": {"type": "string", "not": {"pattern": "^tmp_"}},
     "valide": "fichier.txt", "invalide": "tmp_cache"},
    {"id": "not_required", "schema": {"type": "object", "not": {"required": ["password"]}},
     "valide": {"user": "alice"}, "invalide": {"user": "alice", "password": "1234"}},
    {"id": "not_enum_multi", "schema": {"type": "integer", "not": {"enum": [1, 2, 3]}},
     "valide": 10, "invalide": 2},
    {"id": "not_minimum", "schema": {"type": "number", "not": {"minimum": 100}},
     "valide": 50, "invalide": 150},
    {"id": "not_array_contains", "schema": {"type": "array", "not": {"contains": {"const": 0}}},
     "valide": [1, 2, 3], "invalide": [1, 0, 3]},
    {"id": "not_boolean_true", "schema": {"type": "boolean", "not": {"const": True}},
     "valide": False, "invalide": True},
    {"id": "not_nested_property", "schema": {"type": "object",
        "properties": {"status": {"not": {"const": "banned"}}}, "required": ["status"]},
     "valide": {"status": "active"}, "invalide": {"status": "banned"}},
]


def demander(model, tokenizer, schema_text, valeur_text):
    prompt = INSTRUCTION + f"\nSchema: {schema_text}\nValue: {valeur_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tok = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    rep = tokenizer.decode(out[0][n_tok:], skip_special_tokens=True).lower()
    # "invalid" contient "valid" comme sous-chaîne, on teste "invalid" en premier
    if "invalid" in rep:
        return "invalid"
    if "valid" in rep:
        return "valid"
    return f"?({rep.strip()[:15]})"


def main():
    model, tokenizer = charger_modele()

    bons, total = 0, 0
    n_violations_detectees = 0
    print(f"{'schéma':22s} | valeur valide -> attendu 'valid' | valeur invalide -> attendu 'invalid'")
    print("-" * 90)
    for cas in SUITE:
        sch = json.dumps(cas["schema"])
        rep_v = demander(model, tokenizer, sch, json.dumps(cas["valide"]))
        rep_i = demander(model, tokenizer, sch, json.dumps(cas["invalide"]))

        ok_v = rep_v == "valid"
        ok_i = rep_i == "invalid"
        bons += ok_v + ok_i
        total += 2
        n_violations_detectees += int(ok_i)

        print(f"{cas['id']:22s} | {'OK ' if ok_v else 'X  '}(dit {rep_v:7s}) | "
              f"{'OK ' if ok_i else 'X  '}(dit {rep_i})")

    print("-" * 90)
    print(f"score validation : {bons}/{total} réponses correctes")
    print(f"violations de 'not' correctement détectées : {n_violations_detectees}/{len(SUITE)}")


if __name__ == "__main__":
    main()
