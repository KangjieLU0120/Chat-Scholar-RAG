import re
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
import os

class BM25Store:
    def __init__(self):
        # Store metadata
        self.metadata = []
        # Store tokenized corpus for building the BM25 index
        self.tokenized_corpus = []
        # The BM25 index object
        self.bm25_index = None

    def _tokenize(self, text):
        """
        Tokenize English academic text using Regular Expressions.
        
        The regex pattern r'[a-z0-9\-]+' does the following:
        1. Converts all text to lowercase to ensure case-insensitive matching.
        2. Extracts sequences of letters (a-z), numbers (0-9), and hyphens (-).
        3. Crucially, it preserves academic terms like "CRISPR-Cas9", "ResNet-50", 
           or "GPT-4" as single, complete tokens instead of splitting them.
        4. It automatically strips out commas, periods, parentheses, etc.
        """
        # Convert text to lowercase first
        text = text.lower()
        # Find all matching tokens based on the regex pattern
        tokens = re.findall(r'[a-z0-9\-]+', text)
        return tokens

    def add_documents(self, chunks, source_name):
        """
        Process new text chunks and rebuild the BM25 index.
        """
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

            self.metadata.append(item)

            # Tokenize the chunk and add it to the corpus
            text_to_tokenize = item.get("text", "")
            self.tokenized_corpus.append(self._tokenize(text_to_tokenize))
        
        # Initialize or rebuild the BM25 index with the updated corpus
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=3):
        """
        Search the BM25 index for the most relevant chunks.
        """
        if not self.bm25_index:
            return []

        # Tokenize the user's query using the exact same logic
        tokenized_query = self._tokenize(query)
        
        # Retrieve raw BM25 scores
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Sort indices by score in descending order
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = doc_scores[idx]
            # Only include results with a score greater than zero
            if score > 0:
                item = self.metadata[idx]
                result = dict(item)
                result["score"] = float(score)
                results.append(result)

                # results.append({
                #     "text": self.metadata[idx]["text"],
                #     "source": self.metadata[idx]["source"],
                #     "score": float(score)
                # })
            print(f"🔍 BM25 score for chunk {idx}: {score}")

        return results
    
    def save(self, folder="vector_db"):
        """
        Save the BM25 tokenized corpus and metadata to disk.
        """
        os.makedirs(folder, exist_ok=True)
        # We save the corpus and metadata. BM25Okapi can be quickly re-instantiated from this.
        with open(f"{folder}/bm25_data.pkl", "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        print("✅ BM25 database saved to disk")

    @classmethod
    def load(cls, folder="vector_db"):
        """
        Load BM25 data from disk and reconstruct the index.
        """
        if not os.path.exists(f"{folder}/bm25_data.pkl"):
            return cls() # Return an empty instance if no file exists

        with open(f"{folder}/bm25_data.pkl", "rb") as f:
            data = pickle.load(f)

        store = cls()
        store.metadata = data["metadata"]
        store.tokenized_corpus = data["tokenized_corpus"]
        
        # Reconstruct the index object using the loaded corpus
        from rank_bm25 import BM25Okapi
        if store.tokenized_corpus:
            store.bm25_index = BM25Okapi(store.tokenized_corpus)
            
        print("✅ BM25 database loaded from disk")
        return store
    
def reciprocal_rank_fusion(vector_results, bm25_results, top_k=3, k=60):
    """
    Fuse the results from Vector Search and BM25 Search using RRF.
    :param vector_results: List of dicts from VectorStore.search()
    :param bm25_results: List of dicts from BM25Store.search()
    :param k: Smoothing constant, 60 is the industry standard default.
    """
    
    # Dictionary to accumulate RRF scores for each unique text chunk
    fused_scores = {}

    def add_to_fusion(results):
        for rank, item in enumerate(results):
            # 1. Safely extract the raw text data
            raw_text_data = item.get("text", "")
            
            # 2. Smart unwrapping: If it's a dictionary, extract the actual string inside
            if isinstance(raw_text_data, dict):
                # Fallback to string representation if 'text' key is somehow missing
                text_key = raw_text_data.get("text", str(raw_text_data))
            else:
                # If it's already a string, just use it safely
                text_key = str(raw_text_data)
            
            # 3. Use the pure string as the hash key for fusion
            if text_key not in fused_scores:
                fused_entry = dict(item)
                fused_entry["rrf_score"] = 0.0
                fused_scores[text_key] = fused_entry

                # fused_scores[text_key] = {
                #     "text": raw_text_data, # Keep original structure for downstream use
                #     "source": item.get("source", "unknown"),
                #     "rrf_score": 0.0
                # }
            
            # 4. Calculate RRF score: 1 / (k + rank), rank is 0-indexed so we add 1
            current_rank = rank + 1
            fused_scores[text_key]["rrf_score"] += 1.0 / (k + current_rank)

    # Apply the helper function to both result lists
    add_to_fusion(vector_results)
    add_to_fusion(bm25_results)

    # Convert the dictionary back to a list and sort by the accumulated rrf_score
    sorted_fused_results = sorted(
        fused_scores.values(), 
        key=lambda x: x["rrf_score"], 
        reverse=True
    )

    # Return the ultimate top_k chunks
    return sorted_fused_results[:top_k]
