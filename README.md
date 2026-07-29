# witness-generation-bench

projet de recherche mené pendant mon stage de data analyst (licence IM2D,
université paris-dauphine PSL), sur la capacité d'un modèle de langage à
générer des instances json valides à partir d'un schéma json schema, en
particulier quand celui-ci contient le mot-clé de négation `not`.

## question de recherche

un modèle de langage raisonne-t-il correctement sur une contrainte de
**négation** explicite (`not` en json schema), ou se contente-t-il de générer
des instances plausibles sans vraiment tenir compte de ce qu'il doit éviter ?

## structure du dépôt

```
config.py              modèle utilisé, fenêtre de contexte, gpu/cpu — le seul
                        fichier à modifier pour changer de modèle

benchmark/              benchmark initial : génération zero-shot/few-shot,
                        étude d'ablation, test de validation pure
  generation.py           génération + validation (zero-shot / few-shot)
  ablation.py             comparaison avec/sans "not" pour isoler son effet
  validation_pure.py      test de reconnaissance directe (valid/invalid)

rag/                    système RAG (retrieval-augmented generation)
  embeddings.py           construction des bases vectorielles FAISS
                        (MiniLM généraliste vs CodeBERT orienté code)
  pipeline.py             pipeline complet : retriever -> prompt -> génération
                        -> extraction -> validation
  prompts.py              les différentes formulations de prompt testées
  ensemble.py             stratégie d'ensemble (essayer plusieurs k)
  evaluate.py             harnais de comparaison des hyperparamètres

data/                   emplacement attendu pour la base de référence (202
                        paires schéma/instance) — voir data/README.md

results/                résultats bruts (csv, json) et graphiques
```

## changer de modèle

tout ce qui dépend du modèle est centralisé dans `config.py`. pour tester un
autre modèle (par exemple un modèle plus gros, sur une machine plus
puissante), il suffit de modifier ces deux lignes en haut du fichier :

```python
MODEL_NAME = "microsoft/phi-2"      # remplacer par le modèle voulu
MAX_CONTEXT_TOKENS = 2048            # remplacer par sa vraie fenêtre de contexte
```

le reste du code (limite de tokens en entrée, choix gpu/cpu) est calculé
automatiquement à partir de ces deux valeurs. aucun autre fichier n'a besoin
d'être modifié.

## reproduire les résultats — guide pas à pas (google colab)

ce projet a été développé et testé dans google colab. voici la séquence
exacte pour reproduire le benchmark, dans l'ordre.

### 1. cloner ce dépôt et installer les dépendances

```python
!git clone https://github.com/zeinmil/witness-generation-bench.git
%cd witness-generation-bench
!pip install -q -r requirements.txt
```

### 2. récupérer le corpus de schémas (jsonschemabench)

le benchmark (`benchmark/`) a besoin du jeu de données source, à télécharger
séparément (il n'est pas inclus dans ce dépôt, il est trop volumineux) :

```python
!git clone https://github.com/guidance-ai/jsonschemabench.git /content/jsonschemabench
```

vérifie que ça a bien fonctionné :
```python
import os
print(os.listdir("/content/jsonschemabench/data/"))
```
tu dois voir les 10 collections : `Github_easy`, `Github_trivial`,
`Github_medium`, `Github_hard`, `Github_ultra`, `JsonSchemaStore`,
`Kubernetes`, `WashingtonPost`, `Glaiveai2K`, `Snowplow`.

### 3. lancer le benchmark de génération

```python
%cd benchmark
!python generation.py --collection Github_easy --condition zero-shot
```

ça charge le modèle défini dans `config.py`, génère une instance pour chaque
schéma contenant `not` de la collection choisie, et affiche le score final.
change `Github_easy` pour n'importe quelle autre collection, et `zero-shot`
en `few-shot` pour la seconde condition testée.

### 4. lancer l'étude d'ablation

```python
!python ablation.py --data_dir /content/jsonschemabench/data/ --out_dir .
```

attention : ce script tourne sur **toutes** les collections d'un coup, ça
peut prendre du temps selon la machine. pour tester juste une collection,
importer directement `ablation_collection()` dans une cellule plutôt que de
lancer `main()`.

### 5. tester le RAG

pour la partie RAG, il faut la base de référence (202 paires schéma/instance,
voir `data/README.md` pour savoir où la récupérer), au format csv avec les
colonnes `nom_schema`, `schema`, `instance`.

```python
%cd ../rag
!python embeddings.py
```

ça construit les deux bases vectorielles (`faiss_generaliste` et
`faiss_code`) à partir de `base_rag.csv`. ensuite, dans une cellule python :

```python
import sys
sys.path.append(".")
from embeddings import charger_base, construire_documents, charger_base_vectorielle
from pipeline import pipeline_rag
from generation import charger_modele  # si pas déjà chargé
from langchain_community.embeddings import HuggingFaceEmbeddings

model, tokenizer = charger_modele()

df = charger_base()
noms = df["nom_schema"].tolist()
textes_schemas = df["schema"].tolist()

embedding_code = HuggingFaceEmbeddings(model_name="microsoft/codebert-base")
base_vectorielle_code = charger_base_vectorielle("faiss_code", embedding_code)

# test sur un seul schéma
resultat = pipeline_rag(model, tokenizer, noms[0], textes_schemas[0], base_vectorielle_code, k=1)
print(resultat)
```

pour comparer plusieurs configurations d'un coup (embedding, k, prompt), voir
`rag/evaluate.py`.

## méthodologie en résumé

1. **corpus** : 202 schémas contenant `not`, filtrés depuis *jsonschemabench*.
2. **benchmark** : le modèle testé en zero-shot et few-shot sur ce corpus.
3. **ablation** : comparaison de chaque schéma avec et sans son `not`, pour
   isoler l'effet propre de cette contrainte. résultat obtenu avec phi-2 : le
   taux de réussite double presque quand on retire le `not` (10/202 -> 18/202),
   et 7 schémas sont bloqués précisément par lui.
4. **validation pure** : test direct (valid/invalid) sur 10 violations
   explicites de `not` -> 0 détectées par phi-2.
5. **base de référence** : 202 paires (schéma, instance) construites pour le
   RAG, dont 96% ont une instance de qualité rédactionnelle humaine. 5
   schémas ont été identifiés et prouvés logiquement insatisfiables.
6. **RAG** : retrieval + prompt + génération déterministe + validation, avec
   comparaison systématique de plusieurs hyperparamètres.

## résultats obtenus avec phi-2

| configuration | score (n=20) |
|---|---|
| sans RAG (zero-shot pur) | 1/20 (5%) |
| RAG k=3, MiniLM | 6/20 (30%) |
| RAG k=1, MiniLM | 11/20 (55%) |
| RAG k=1, CodeBERT | 11/20 (55%) |
| ensemble k=[1,2], MiniLM | 11/20 (55%) |
| **ensemble k=[1,2], CodeBERT** | **12/20 (60%)** |
| prompt insistant sur le `not` | 11/20 (55%) |
| prompt complet (toutes contraintes) | 8/20 (40%) |

enseignements clés (obtenus avec phi-2, à revérifier avec un autre modèle) :
- le RAG améliore massivement le taux de réussite par rapport au zero-shot pur
- `k=1` est nettement meilleur que `k=2`/`k=3` : phi-2 a une fenêtre de
  contexte limitée à 2048 tokens, donc ajouter des exemples réduit la place
  disponible pour terminer la génération. **avec un modèle à plus grande
  fenêtre de contexte, cet effet pourrait disparaître ou s'inverser** — c'est
  justement un point intéressant à retester.
- l'embedding orienté code (codebert) fait systématiquement un peu mieux que
  le généraliste (minilm)
- modifier le prompt n'améliore pas les résultats avec phi-2, et peut les
  dégrader si le prompt devient trop long
- sur l'ensemble des échecs observés, seuls ~35% sont de vraies violations
  du `not` : les deux tiers restants sont d'autres erreurs (type, enum,
  `oneOf`)

## outils utilisés

- **embeddings** : `sentence-transformers/all-MiniLM-L6-v2`,
  `microsoft/codebert-base`
- **orchestration RAG** : [langchain](https://python.langchain.com/)
  (`langchain-community`, FAISS)
- **validation** : [`jsonschema`](https://pypi.org/project/jsonschema/)
- **corpus source** : [jsonschemabench](https://github.com/guidance-ai/jsonschemabench)

## auteur

zein mil — licence IM2D, université paris-dauphine (PSL)
