# import requests


# class EmbeddingService:
#     def __init__(self):
#         self.url = "http://localhost:11434/api/embeddings"
#         self.model = "nomic-embed-text"

#     def get_embedding(self, text):

#         try:
#             payload = {
#                 "model": self.model,
#                 "prompt": text
#             }

#             response = requests.post(self.url, json=payload)

#             if response.status_code == 200:
#                 return response.json()["embedding"]

#             print("Embedding error:", response.text)
#             return None

#         except Exception as e:
#             print("Embedding Service Error:", e)
#             return None
        
import requests
import numpy as np

class EmbeddingService:
    def __init__(self, truncate_dim=256):
        self.url = "http://localhost:11434/api/embeddings"
        self.model = "nomic-embed-text:v1.5" # Locked to v1.5
        # Set the default dimension to retain, leveraging Matryoshka Representation Learning
        self.truncate_dim = truncate_dim 

    def get_embedding(self, text, task_type="document"):
        """
        Get the text embedding vector.
        :param text: The original text to be converted.
        :param task_type: Core distinction -> "document" (papers stored in the DB) or "query" (user's question).
        """
        
        # 1. Automatically append the prefix strongly required by Nomic
        if task_type == "document":
            prompt = f"search_document: {text}"
        elif task_type == "query":
            prompt = f"search_query: {text}"
        else:
            prompt = text

        try:
            payload = {
                "model": self.model,
                "prompt": prompt
            }

            response = requests.post(self.url, json=payload)

            if response.status_code == 200:
                # Ollama returns a 768-dimensional Python list
                raw_embedding = response.json()["embedding"]

                # 2. Core lightweighting: Truncate dimensions (768 -> 256)
                vector = np.array(raw_embedding[:self.truncate_dim])

                # 3. L2 Normalization (Strict requirement for FAISS when using IndexFlatIP to simulate cosine similarity)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                return vector

            print(f"Embedding error (Status {response.status_code}):", response.text)
            return None

        except requests.exceptions.RequestException as e:
            print("Network error connecting to Ollama:", e)
            return None
        except Exception as e:
            print("Embedding Service Error:", e)
            return None
