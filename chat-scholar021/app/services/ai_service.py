import requests
import json

class AIService:
    def __init__(self, retrieval_service):
        self.url = "http://localhost:11434/api/generate"
        self.model = "tinyllama"
        self.retriever = retrieval_service

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
            # 强制证据抽取的两阶段格式
            return f"""
You are an academic AI assistant. Answer the question using ONLY the provided context.

Follow this exact two-step format:

Step 1 - Quote the exact sentence(s) from the context that contain the answer. If multiple sentences are relevant, list them.
Step 2 - Based solely on the quoted evidence, provide a concise answer.

If no relevant evidence exists in the context, output exactly:

EVIDENCE: None
ANSWER: I cannot find a reliable answer in the provided documents.

Context:
{context_text}

Question:
{latest_question}

Now, produce your response in the format:
EVIDENCE: ...
ANSWER: ...
"""
        else:
            return f"""
[Task]
Answer directly: I cannot find information in the provided documents to answer.

[CRITICAL RULES]
Do NOT use any general knowledge or information from your training data.
"""

    def _extract_answer_from_response(self, response_text):
        """
        从模型输出中提取 ANSWER 部分。
        如果找不到，则返回原文本（但会提示警告）。
        """
        answer_marker = "ANSWER:"
        if answer_marker in response_text:
            # 分割并取最后一个 ANSWER: 之后的部分
            parts = response_text.split(answer_marker)
            answer_part = parts[-1].strip()
            return answer_part
        else:
            print("Warning: Model response did not contain ANSWER: marker. Using full response.")
            return response_text

    # ---------------------------------------------------
    # FAITHFULNESS SELF-CHECK
    # ---------------------------------------------------
    def _check_faithfulness(self, answer, context_text):
        """
        判断答案是否完全基于上下文。返回 True/False
        """
        if not context_text or not answer:
            return False

        # 如果答案已经是拒绝语句，直接视为忠实
        if "cannot find" in answer.lower() or "not find" in answer.lower():
            return True

        prompt = f"""
You are a faithfulness checker. Determine if the following answer can be directly inferred from the provided context.
Reply ONLY with "YES" or "NO". Do not explain.

Context:
{context_text}

Answer:
{answer}

Faithful:
"""
        decision = self._sync_generate(prompt, temperature=0.0).upper()
        return "YES" in decision

    # ---------------------------------------------------
    # GENERATE RESPONSE (NON-STREAMING) FOR EVALUATION
    # ---------------------------------------------------
    def generate_response(self, chat_history, document_chunks=None):
        """
        Generate a non-streaming response for evaluation scripts.
        """
        latest_question = self._get_latest_question(chat_history)
        context_text, source_info = self._build_context_and_sources(document_chunks)
        prompt = self._build_chat_prompt(latest_question, context_text)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0
            }
        }

        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            reply = data.get("response", "").strip()

            # 解析 ANSWER 部分
            answer = self._extract_answer_from_response(reply)

            # 自检
            if context_text and not self._check_faithfulness(answer, context_text):
                print("Faithfulness check failed, rejecting answer.")
                answer = "I cannot find a reliable answer in the provided documents."

            if source_info:
                answer += f"\n\n📄 Sources:\n{source_info}"
            return answer
        except Exception as e:
            print("AI generate error:", e)
            return "⚠️ AI service error."

    # ---------------------------------------------------
    # STREAMING RESPONSE (LIVE CHAT)
    # ---------------------------------------------------
    def stream_response(self, latest_question, document_chunks=None):
        context_text, source_info = self._build_context_and_sources(document_chunks)
        prompt = self._build_chat_prompt(latest_question, context_text)

        print("🚀 Sending prompt to AI service:", prompt)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0
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

    # ---------------------------------------------------
    # Agentic Stream Response
    # ---------------------------------------------------

    def _sync_generate(self, prompt, temperature=0.1):
        """Helper method for non-streaming LLM calls used in evaluation and rewriting."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.5
            }
        }
        try:
            response = requests.post(self.url, json=payload).json()
            return response.get("response", "").strip()
        except Exception as e:
            print("Sync LLM error:", e)
            return ""

    def _evaluate_context(self, question, context_text):
        """Evaluates if the retrieved context is sufficient to answer the question."""
        if not context_text:
            return False

        prompt = f"""
        [Task]
        Does the provided context contain enough information to answer the question?
        Reply ONLY with "YES" or "NO". Do not explain.

        Context:
        {context_text}

        Question:
        {question}

        Decision:
        """
        decision = self._sync_generate(prompt).upper()
        return "YES" in decision

    def _rewrite_query(self, original_question):
        """Rewrites the original question into a better search query."""
        prompt = f"""
        [Task]
        The previous database search failed to find an answer to this question: "{original_question}"
        Rewrite the question into a concise keyword search query to find better documents.
        Reply ONLY with the new search query. Do not add quotes or explanations.

        New Query:
        """
        new_query = self._sync_generate(prompt, temperature=0.4)
        return new_query.strip(' "\'\n')

    def agentic_stream_response(self, chat_history, max_attempts=2):
        """
        Execute the Agentic RAG loop using the injected retrieval service.
        """
        original_question = self._get_latest_question(chat_history)
        current_query = original_question

        context_text = ""
        source_info = ""

        # Step 1 to 3: The Agentic Loop (Retrieve -> Evaluate -> Rewrite)
        for attempt in range(1, max_attempts + 1):
            print(f"🤖 Agentic Loop - Attempt {attempt}/{max_attempts} | Query: '{current_query}'")

            document_chunks = self.retriever.search(current_query)
            context_text, source_info = self._build_context_and_sources(document_chunks)

            if attempt < max_attempts:
                is_sufficient = self._evaluate_context(original_question, context_text)
                if is_sufficient:
                    break
                else:
                    new_query = self._rewrite_query(original_question)
                    current_query = new_query if new_query else original_question

        # Step 4: Final Generation (Stream the final answer)
        print("🚀 Sending final prompt to AI service...")
        prompt = self._build_chat_prompt(original_question, context_text)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0
            }
        }

        try:
            response = requests.post(self.url, json=payload, stream=True)

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data.get("response", "")
                    except Exception as parse_error:
                        print(f"JSON parsing error during stream: {parse_error}")
                        continue

            if source_info:
                yield f"\n\n📄 Full Sources:\n{source_info}"

        except Exception as e:
            print(f"AI streaming request error: {e}")
            yield "⚠️ AI service error while generating the final response."

    # ---------------------------------------------------
    # AGENTIC GENERATE (NON-STREAMING) FOR EVALUATION
    # ---------------------------------------------------
    def agentic_generate(self, chat_history, max_attempts=2):
        """
        Non-streaming version of agentic response.
        Returns (answer, final_document_chunks) tuple.
        """
        original_question = self._get_latest_question(chat_history)
        current_query = original_question

        context_text = ""
        source_info = ""
        final_chunks = []  # will store the final document chunks used for generation

        # Agentic loop
        for attempt in range(1, max_attempts + 1):
            print(f"🤖 Agentic Loop - Attempt {attempt}/{max_attempts} | Query: '{current_query}'")

            document_chunks = self.retriever.search(current_query)
            context_text, source_info = self._build_context_and_sources(document_chunks)

            if attempt < max_attempts:
                is_sufficient = self._evaluate_context(original_question, context_text)
                if is_sufficient:
                    final_chunks = document_chunks
                    break
                else:
                    new_query = self._rewrite_query(original_question)
                    current_query = new_query if new_query else original_question
            else:
                final_chunks = document_chunks  # last attempt

        # Generate final answer
        prompt = self._build_chat_prompt(original_question, context_text)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0
            }
        }

        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            reply = data.get("response", "").strip()
            answer = self._extract_answer_from_response(reply)

            # 自检
            if context_text and not self._check_faithfulness(answer, context_text):
                print("Faithfulness check failed, rejecting answer.")
                answer = "I cannot find a reliable answer in the provided documents."

            if source_info:
                answer += f"\n\n📄 Sources:\n{source_info}"
            return answer, final_chunks
        except Exception as e:
            print("AI agentic generate error:", e)
            return "⚠️ AI service error.", []