import os
import pickle

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_embeddings(self, embeddings, chunks, source_name=None, document_id=None):
        if not embeddings:
            return

        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)

        for chunk in chunks:
            if isinstance(chunk, dict):
                item = dict(chunk)
            else:
                item = {"text": str(chunk), "source": source_name}

            if source_name and not item.get("source"):
                item["source"] = source_name
            if document_id and not item.get("document_id"):
                item["document_id"] = document_id

            self.metadata.append(item)

    def _matches_filters(self, item, source_name=None, document_id=None):
        if source_name and item.get("source") != source_name:
            return False
        if document_id and item.get("document_id") != document_id:
            return False
        return True

    def search(self, query_embedding, top_k=3, source_name=None, document_id=None):
        if self.index.ntotal == 0 or not self.metadata:
            return []

        query_vector = np.array([query_embedding], dtype="float32")

        # Search a wider pool first so that post-filtering by document/source still returns enough hits.
        candidate_k = min(max(top_k * 8, 20), len(self.metadata))
        distances, indices = self.index.search(query_vector, candidate_k)

        results = []
        seen_chunk_ids = set()

        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]
            if not self._matches_filters(item, source_name=source_name, document_id=document_id):
                continue

            result = dict(item)
            result["distance"] = float(distance)

            chunk_id = result.get("chunk_id")
            if chunk_id is not None:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

            results.append(result)
            if len(results) >= top_k:
                break

        return results

    def save(self, folder="vector_db"):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print("✅ Vector database saved")

    @classmethod
    def load(cls, folder="vector_db"):
        index_path = os.path.join(folder, "faiss.index")
        metadata_path = os.path.join(folder, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return None

        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata if isinstance(metadata, list) else []
        print("✅ Vector database loaded")
        return store
