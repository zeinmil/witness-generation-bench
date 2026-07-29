# data/

Ce dossier est destiné à recevoir, localement :

- `corpus_not.json` : les 202 schémas JSON Schema contenant `not`, extraits de
  jsonschemabench.
- `base_rag.csv` (ou `base_rag_propre.json`) : la base de référence de 202
  paires (schéma, instance), utilisée pour construire les bases vectorielles
  du RAG. Colonnes attendues : `nom_schema`, `schema`, `instance`.

Ces fichiers ne sont pas versionnés dans le dépôt Git (voir `.gitignore`) car
certains schémas dépassent plusieurs centaines de milliers de caractères. Ils
sont disponibles séparément (Google Drive du projet) ou reconstructibles via
les scripts de `base_construction/`.
