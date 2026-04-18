import requests


class EmbeddingService:
    def __init__(self):
        self.url = "http://localhost:11434/api/embeddings"
        self.model = "nomic-embed-text"

    def get_embedding(self, text):

        try:
            payload = {
                "model": self.model,
                "prompt": text
            }

            response = requests.post(self.url, json=payload)

            if response.status_code == 200:
                return response.json()["embedding"]

            print("Embedding error:", response.text)
            return None

        except Exception as e:
            print("Embedding Service Error:", e)
            return None
