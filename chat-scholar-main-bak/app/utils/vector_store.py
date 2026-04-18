import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self, dimension):
        # self.index = faiss.IndexFlatL2(dimension)
        # If vectors are L2-normalized, Inner Product == Cosine Similarity
        # Higher score means more similar.
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []   # stores text + source

    # ---------------------------------
    # Add embeddings with metadata
    # ---------------------------------
    def add_embeddings(self, embeddings, chunks, source_name):

        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        # for chunk in chunks:
        #     self.metadata.append({
        #         "text": chunk,
        #         "source": source_name
        #     })

        for chunk in chunks:
            if isinstance(chunk, dict):
                item = dict(chunk)
            else:
                item = {"text": str(chunk), "source": source_name}

            if source_name and not item.get("source"):
                item["source"] = source_name
            # if document_id and not item.get("document_id"):
            #     item["document_id"] = document_id

            self.metadata.append(item)

            print(f"✅ Added chunk to vector store with metadata: {item}")

    # ---------------------------------
    # Semantic search
    # ---------------------------------
    def search(self, query_embedding, top_k=3, threshold=0.4):
        """
        Search for similar vectors but only return those above the threshold.
        """
        query_vector = np.array([query_embedding]).astype("float32")

        # Search a wider pool first so that post-filtering by document/source still returns enough hits.
        candidate_k = min(max(top_k * 8, 20), len(self.metadata))
        distances, indices = self.index.search(query_vector, candidate_k)

        # distances, indices = self.index.search(query_vector, top_k)

        results = []

        # for idx in indices[0]:
        #     if idx < len(self.metadata):
        #         results.append(self.metadata[idx])

        # Zip distances and indices together so we can return the score
        for score, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata) and idx != -1 and score >= threshold: # -1 means not enough vectors found
                # results.append({
                #     "text": self.metadata[idx]["text"],
                #     "source": self.metadata[idx]["source"],
                #     "score": float(score) # Include score for threshold filtering later
                # })

                item = self.metadata[idx]
                result = dict(item)
                result["score"] = float(score)
                results.append(result)

            if len(results) >= top_k:
                break

        print(f"🔍 Search results (score >= {threshold}):", results)

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
