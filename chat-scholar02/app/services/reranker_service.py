from sentence_transformers import CrossEncoder
import numpy as np

class RerankerService:
    def __init__(self):
        # Using a highly efficient, lightweight Cross-Encoder model
        # ms-marco-MiniLM is optimized for passage re-ranking tasks
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query, candidates, top_k=3):
        """
        Refine the order of retrieved candidates using semantic relevance scoring.
        :param query: The user's question string.
        :param candidates: List of dictionaries containing 'text' and 'source'.
        :param top_k: Number of best-matching chunks to return.
        """
        if not candidates:
            return []

        # # 1. Prepare (Query, Text) pairs for the model to evaluate
        # # Cross-Encoders process the query and document together to capture deep semantic links
        # pairs = []
        # for c in candidates:
        #     # Safely extract string to prevent tokenizer TypeError
        #     text_content = c.get("text", "")
        #     safe_text = text_content["text"] if isinstance(text_content, dict) else str(text_content)
            
        #     pairs.append([query, safe_text])
            
        #     # Normalize the candidate dictionary to ensure downstream compatibility
        #     # This guarantees that c["text"] is always a pure string, preventing AttributeError later
        #     c["text"] = safe_text

        # 1. Prepare (Query, Text) pairs for the model
        pairs = []
        for c in candidates:
            raw_data = c.get("text", "")
            
            if isinstance(raw_data, dict):
                for key, value in raw_data.items():
                    if key != "text" and key not in c:
                        c[key] = value
                
                safe_text = raw_data.get("text", str(raw_data))
            else:
                safe_text = str(raw_data)
            
            pairs.append([query, safe_text])
            
            c["text"] = safe_text
        
        # 2. Compute relevance scores (usually ranging from 0 to 1 or logits)
        scores = self.model.predict(pairs)

        # 3. Attach scores to candidates and sort them
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)

        # Sort by rerank_score in descending order
        sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        # 4. Return the most relevant chunks
        return sorted_candidates[:top_k]