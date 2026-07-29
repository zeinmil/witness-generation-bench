# data/

Ce dossier est destiné à recevoir, localement :

- `corpus_not.json` : les 202 schémas JSON Schema contenant `not`, extraits de
  jsonschemabench.
- `base_rag.csv` (ou `base_rag_propre.json`) : la base de référence de 202
  paires (schéma, instance), utilisée pour construire les bases vectorielles
  du RAG. Colonnes attendues : `nom_schema`, `schema`, `instance`.


