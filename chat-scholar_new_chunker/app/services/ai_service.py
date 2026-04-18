import requests
import json


class AIService:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "tinyllama"

    # ---------------------------------------------------
    # NORMAL RESPONSE (PDF CHAT)
    # ---------------------------------------------------
    def generate_response(self, chat_history, document_chunks=None):

        latest_question = ""

        for msg in reversed(chat_history):
            if msg["role"] == "user":
                latest_question = msg["content"]
                break

        source_info = ""

        if document_chunks:
            context_text = "\n\n".join(
                [chunk["text"] for chunk in document_chunks]
            )

            sources = list(set(
                [chunk["source"] for chunk in document_chunks]
            ))

            source_info = ", ".join(sources)

            prompt = f"""
You are an academic AI assistant.

Use ONLY the provided context to answer.
If answer not present, say you cannot find it.

Context:
{context_text}

Question: {latest_question}

Answer:
"""
        else:
            prompt = f"Question: {latest_question}\nAnswer:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.url, json=payload)
            reply = response.json()["response"].strip()

            if source_info:
                reply += f"\n\n📄 Source: {source_info}"

            return reply

        except Exception as e:
            print("AI response error:", e)
            return "⚠️ AI service error."

    # ---------------------------------------------------
    # STREAMING RESPONSE (LIVE CHAT)
    # ---------------------------------------------------
    def stream_response(self, chat_history, document_chunks=None):

        latest_question = ""

        for msg in reversed(chat_history):
            if msg["role"] == "user":
                latest_question = msg["content"]
                break

        source_info = ""

        if document_chunks:
            context_text = "\n\n".join(
                [chunk["text"] for chunk in document_chunks]
            )

            sources = list(set(
                [chunk["source"] for chunk in document_chunks]
            ))

            source_info = ", ".join(sources)

            prompt = f"""
You are an academic AI assistant.

Use ONLY the provided context.

Context:
{context_text}

Question: {latest_question}

Answer:
"""
        else:
            prompt = f"Question: {latest_question}\nAnswer:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }

        response = requests.post(
            self.url,
            json=payload,
            stream=True
        )

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")
                except:
                    continue

        if source_info:
            yield f"\n\n📄 Source: {source_info}"

    # ---------------------------------------------------
    # ESSAY GRADING (NEW FEATURE)
    # ---------------------------------------------------
    def grade_essay(self, essay_text):

        prompt = f"""
You are an academic essay evaluator.

Evaluate the following student essay and provide:

1. Overall Score (out of 10)
2. Strengths
3. Weaknesses
4. Grammar Feedback
5. Suggestions for Improvement

Essay:
{essay_text}

Evaluation:
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 300
            }
        }

        try:
            response = requests.post(self.url, json=payload)

            if response.status_code == 200:
                return response.json()["response"].strip()

            return "⚠️ Essay grading failed."

        except Exception as e:
            print("Essay grading error:", e)
            return "⚠️ AI grading service unavailable."
