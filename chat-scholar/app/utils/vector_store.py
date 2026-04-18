import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []   # stores text + source

    # ---------------------------------
    # Add embeddings with metadata
    # ---------------------------------
    def add_embeddings(self, embeddings, chunks, source_name):

        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        for chunk in chunks:
            self.metadata.append({
                "text": chunk,
                "source": source_name
            })

    # ---------------------------------
    # Semantic search
    # ---------------------------------
    def search(self, query_embedding, top_k=3):

        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

    # ---------------------------------
    # Save vector DB
    # ---------------------------------
    def save(self, folder="vector_db"):
        os.makedirs(folder, exist_ok=True)

        faiss.write_index(self.index, f"{folder}/faiss.index")

        with open(f"{folder}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        print("✅ Vector database saved")

    # ---------------------------------
    # Load vector DB
    # ---------------------------------
    @classmethod
    def load(cls, folder="vector_db"):

        if not os.path.exists(f"{folder}/faiss.index"):
            return None

        index = faiss.read_index(f"{folder}/faiss.index")

        with open(f"{folder}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata

        print("✅ Vector database loaded")

        return store
