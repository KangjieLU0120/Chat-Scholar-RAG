import json
import os

REGISTRY_PATH = "vector_db/documents.json"


def load_documents():
    if not os.path.exists(REGISTRY_PATH):
        return []

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def add_document(filename):

    docs = load_documents()

    if filename not in docs:
        docs.append(filename)

        os.makedirs("vector_db", exist_ok=True)

        with open(REGISTRY_PATH, "w") as f:
            json.dump(docs, f, indent=2)
