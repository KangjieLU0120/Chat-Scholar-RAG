import requests
import json

class AIService:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "tinyllama"

    def _get_latest_question(self, chat_history):
        for msg in reversed(chat_history):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _build_context_and_sources(self, document_chunks, max_chunks=5):
        if not document_chunks:
            return "", ""

        selected_chunks = document_chunks[:max_chunks]
        context_parts = []
        source_map = {}

        for chunk in selected_chunks:
            text = chunk.get("text", "").strip()
            source = chunk.get("source", "Unknown Source")
            page = chunk.get("page") or chunk.get("page_start")
            section = chunk.get("section")

            if text:
                header_parts = [f"Source: {source}"]
                if section:
                    header_parts.append(f"Section: {section}")
                if page is not None:
                    header_parts.append(f"Page: {page}")
                context_parts.append(f"[{' | '.join(header_parts)}]\n{text}")

            source_map.setdefault(source, {"pages": set(), "sections": set()})
            if page is not None:
                source_map[source]["pages"].add(page)
            if section:
                source_map[source]["sections"].add(section)

        source_lines = []
        for source, info in source_map.items():
            parts = [source]
            if info["sections"]:
                parts.append("sections: " + "; ".join(sorted(info["sections"])))
            if info["pages"]:
                parts.append("p. " + ", ".join(str(p) for p in sorted(info["pages"])))
            source_lines.append("- " + " | ".join(parts))

        return "\n\n".join(context_parts), "\n".join(source_lines)

    def _build_chat_prompt(self, latest_question, context_text=""):
        if context_text:
            return f"""
                    You are an academic AI assistant.Answer the question using ONLY the provided context.

                    [CRITICAL RULES]
                    1. Do not summarize or repeat these instructions. Just answer directly.
                    2. If the answer is not clearly supported by the context, you MUST say: "I cannot find a reliable answer in the provided documents."
                    3. Do NOT use any general knowledge or information from your training data.
                    4. Every claim must be directly supported by the context.
                    5. Keep the answer clear and concise.
                    6. Use the metadata in the context when it helps distinguish sections or pages.
                    7. Focus only on information relevant to the user's question.

                    Context:
                    {context_text}

                    Question:
                    {latest_question}

                    Answer:
                    """
        return f"""
                [Task]
                Answer directly: I cannot find information in the provided documents to answer.

                [CRITICAL RULES]
                Do NOT use any general knowledge or information from your training data.
                """
    # ---------------------------------------------------
    # STREAMING RESPONSE (LIVE CHAT)
    # ---------------------------------------------------
    def stream_response(self, chat_history, document_chunks=None):
        latest_question = self._get_latest_question(chat_history)
        context_text, source_info = self._build_context_and_sources(document_chunks)
        prompt = self._build_chat_prompt(latest_question, context_text)

        print("🚀 Sending prompt to AI service:", prompt)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {      
                "temperature": 0.1,  # Lower = more deterministic, less creative
                "top_p": 0.3  # More conservative sampling
            }
        }

        try:
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
                yield f"\n\n📄 Full Sources: {source_info}"

        except Exception as e:
            print("AI streaming error:", e)
            yield "⚠️ AI service error."

# import requests
# import json


# class AIService:
#     def __init__(self):
#         self.url = "http://localhost:11434/api/generate"
#         self.model = "tinyllama"

#     # ---------------------------------------------------
#     # NORMAL RESPONSE (PDF CHAT)
#     # ---------------------------------------------------
#     def generate_response(self, chat_history, document_chunks=None):

#         latest_question = ""

#         for msg in reversed(chat_history):
#             if msg["role"] == "user":
#                 latest_question = msg["content"]
#                 break

#         source_info = ""

#         if document_chunks:
#             context_text = "\n\n".join(
#                 [chunk["text"] for chunk in document_chunks]
#             )

#             sources = list(set(
#                 [chunk["source"] for chunk in document_chunks]
#             ))

#             source_info = ", ".join(sources)

#             prompt = f"""
# You are an academic AI assistant.

# Use ONLY the provided context to answer.
# If answer not present, say you cannot find it.

# Context:
# {context_text}

# Question: {latest_question}

# Answer:
# """
#         else:
#             prompt = f"Question: {latest_question}\nAnswer:"

#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "stream": False
#         }

#         try:
#             response = requests.post(self.url, json=payload)
#             reply = response.json()["response"].strip()

#             if source_info:
#                 reply += f"\n\n📄 Source: {source_info}"

#             return reply

#         except Exception as e:
#             print("AI response error:", e)
#             return "⚠️ AI service error."

#     # ---------------------------------------------------
#     # STREAMING RESPONSE (LIVE CHAT)
#     # ---------------------------------------------------
#     def stream_response(self, chat_history, document_chunks=None):

#         latest_question = ""

#         for msg in reversed(chat_history):
#             if msg["role"] == "user":
#                 latest_question = msg["content"]
#                 break

#         source_info = ""

#         if document_chunks:
#             context_text = "\n\n".join(
#                 [chunk["text"] for chunk in document_chunks]
#             )

#             sources = list(set(
#                 [chunk["source"] for chunk in document_chunks]
#             ))

#             source_info = ", ".join(sources)

#             prompt = f"""
# You are an academic AI assistant.

# Use ONLY the provided context.

# Context:
# {context_text}

# Question: {latest_question}

# Answer:
# """
#         else:
#             prompt = f"Question: {latest_question}\nAnswer:"

#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "stream": True
#         }

#         response = requests.post(
#             self.url,
#             json=payload,
#             stream=True
#         )

#         for line in response.iter_lines():
#             if line:
#                 try:
#                     data = json.loads(line.decode("utf-8"))
#                     yield data.get("response", "")
#                 except:
#                     continue

#         if source_info:
#             yield f"\n\n📄 Source: {source_info}"