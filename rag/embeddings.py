"""
rag/embeddings.py

construction de deux bases vectorielles (faiss) à partir de la base de
référence de 202 paires (schéma, instance), pour comparer deux façons
différentes de mesurer la similarité entre schémas json schema :

- MiniLM (sentence-transformers/all-MiniLM-L6-v2) : embedding généraliste,
  capte le sens des mots (descriptions, noms de propriétés)
- CodeBERT (microsoft/codebert-base) : embedding orienté code, capte mieux
  la structure (imbrication, mots-clés comme oneOf/not/required)
"""
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def charger_base(chemin_csv="base_rag.csv"):
    """
    charge la base de référence depuis le csv (colonnes : nom_schema, schema, instance).

    keep_default_na=False évite que pandas convertisse certaines valeurs
    textuelles (un schéma vide, ou "null") en NaN, ce qui casserait la
    construction des Document juste après.
    """
    df = pd.read_csv(chemin_csv, keep_default_na=False)
    # les schémas marqués "insatisfiable" n'ont pas de vraie instance,
    # ils n'ont rien à faire dans une base d'exemples pour le retrieval
    df = df[df["instance"].str.strip() != "insatisfiable"].reset_index(drop=True)
    return df


def construire_documents(df):
    """transforme les lignes du csv en objets Document langchain (schéma = texte, instance = métadonnée)"""
    return [
        Document(
            page_content=row["schema"],
            metadata={"nom_schema": row["nom_schema"], "instance": row["instance"]},
        )
        for _, row in df.iterrows()
    ]


def construire_base_vectorielle(documents, nom_modele, chemin_sauvegarde):
    embedding = HuggingFaceEmbeddings(model_name=nom_modele)
    base_vectorielle = FAISS.from_documents(documents, embedding)
    base_vectorielle.save_local(chemin_sauvegarde)
    return base_vectorielle, embedding


def charger_base_vectorielle(chemin_sauvegarde, embedding):
    return FAISS.load_local(chemin_sauvegarde, embedding, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    df = charger_base()
    print(f"{len(df)} schémas chargés (hors insatisfiables)")
    documents = construire_documents(df)

    print("construction de la base MiniLM (généraliste)...")
    construire_base_vectorielle(documents, "sentence-transformers/all-MiniLM-L6-v2", "faiss_generaliste")

    print("construction de la base CodeBERT (code)...")
    construire_base_vectorielle(documents, "microsoft/codebert-base", "faiss_code")

    print("terminé.")
