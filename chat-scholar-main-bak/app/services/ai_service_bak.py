import json
import requests


class AIService:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "tinyllama"



    def _build_context_and_sources(self, document_chunks, max_chunks=3):
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
You are an academic AI assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do not invent or guess information.
2. If the answer is not clearly supported by the context, say:
   \"I cannot find a reliable answer in the provided document.\"
3. Keep the answer clear and concise.
4. Use the metadata in the context when it helps distinguish sections or pages.
5. Focus only on information relevant to the user's question.

Context:
{context_text}

Question:
{latest_question}

Answer:
"""
        return f"""
You are an academic AI assistant.

Answer the following question clearly and concisely.

Question:
{latest_question}

Answer:
"""


    def stream_response(self, chat_history, document_chunks=None):
        latest_question = self._get_latest_question(chat_history)
        context_text, source_info = self._build_context_and_sources(document_chunks)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2, "num_predict": 250},
        }

        try:
            response = requests.post(self.url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")
                except Exception:
                    continue

            if source_info:
                yield f"\n\n📄 Sources:\n{source_info}"

        except Exception as e:
            print("AI streaming error:", e)
            yield "⚠️ AI service error."
