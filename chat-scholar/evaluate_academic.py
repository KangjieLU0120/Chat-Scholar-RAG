# evaluate_academic.py
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
from app.utils.vector_store import VectorStore

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI

# ------------------------------
# 问题与参考答案（ResNet）
# ------------------------------
questions_resnet = [
    "What is the main problem addressed by the residual learning framework in deep neural networks?",
    "How is the residual mapping formally defined in the paper?",
    "What do the authors explicitly let the stacked layers fit instead of the unreferenced mapping H(x)?",
    "What is the deepest residual network evaluated on the ImageNet dataset?",
    "What top-5 error rate was achieved by an ensemble of residual nets on the ImageNet test set?",
    "In which competition did the residual nets win 1st place on the ILSVRC 2015 classification task?",
    "What are the two types of shortcut connections used in the residual networks?",
    "How many layers does the 34-layer plain network and its residual counterpart have?",
    "On which dataset did the authors present analysis of models with 100 and 1000 layers?",
    "What operation is performed by the shortcut connections in the residual building block?"
]

reference_resnet = [
    "The degradation problem, where accuracy gets saturated and then degrades rapidly as the network depth increases.",
    "F(x) := H(x) - x, where H(x) is the desired underlying mapping.",
    "A residual mapping F(x) := H(x) - x, so the original mapping is recast into F(x) + x.",
    "A 152-layer residual net.",
    "3.57%.",
    "ILSVRC 2015 classification task.",
    "Identity mapping (when dimensions are the same) and projection shortcut (1×1 convolutions when dimensions increase).",
    "34 layers (both the plain network and the residual network).",
    "CIFAR-10.",
    "Element-wise addition of the input x to the output of the stacked layers (y = F(x) + x)."
]

# ------------------------------
# 问题与参考答案（Transformer）
# ------------------------------
questions_transformer = [
    "What is the name of the new simple network architecture proposed in the paper?",
    "What two components does the Transformer model completely dispense with?",
    "What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?",
    "What single-model state-of-the-art BLEU score was established on the WMT 2014 English-to-French translation task?",
    "How many identical layers are stacked in both the encoder and the decoder of the Transformer?",
    "What mechanism is used exclusively for positional encoding in the Transformer?",
    "What is the exact formula for Scaled Dot-Product Attention?",
    "How many heads are used in the multi-head attention mechanism?",
    "What is the maximum path length for self-attention layers according to the complexity comparison table?",
    "What are the three different ways multi-head attention is applied in the Transformer model?"
]

reference_transformer = [
    "The Transformer.",
    "Recurrence and convolutions.",
    "28.4 BLEU, improving over the existing best results (including ensembles) by over 2 BLEU.",
    "41.8.",
    "N = 6 identical layers.",
    "Sine and cosine functions of different frequencies: PE(pos,2i) = sin(pos/10000^(2i/d_model)) and PE(pos,2i+1) = cos(pos/10000^(2i/d_model)).",
    "Attention(Q, K, V) = softmax(QK^T / √d_k) V.",
    "h = 8 parallel attention layers (heads).",
    "O(1).",
    "1) Encoder-decoder attention layers; 2) Self-attention layers in the encoder; 3) Self-attention layers in the decoder (with masking)."
]

# 合并所有问题与参考
all_questions = questions_resnet + questions_transformer
all_references = reference_resnet + reference_transformer

# ------------------------------
# 自定义 LLM（使用 uiuiapi 代理）
# ------------------------------
def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api1.uiuiapi.com/v1"),
        temperature=0.0,
    )

# ------------------------------
# 主评估函数
# ------------------------------
def run_evaluation():
    print("Loading vector store...")
    vector_store = VectorStore.load()
    if vector_store is None:
        print("Error: No vector store found. Please upload ResNet.pdf and transformer.pdf before running evaluation.")
        return

    embedding_service = EmbeddingService()
    ai_service = AIService()

    answers = []
    contexts = []

    for i, question in enumerate(all_questions):
        print(f"Processing question {i+1}/{len(all_questions)}: {question}")

        query_emb = embedding_service.get_embedding(question)
        if query_emb is None:
            print(f"  Failed to get embedding for question {i+1}")
            answers.append("")
            contexts.append([])
            continue

        retrieved = vector_store.search(query_emb, top_k=3)
        context_texts = [chunk["text"] for chunk in retrieved]
        contexts.append(context_texts)

        chat_history = [{"role": "user", "content": question}]
        answer = ai_service.generate_response(chat_history, document_chunks=retrieved)
        answers.append(answer)

    data = {
        "question": all_questions,
        "answer": answers,
        "contexts": contexts,
        "reference": all_references,
    }
    dataset = Dataset.from_dict(data)

    print("\nStarting RAGAs evaluation...")
    llm = get_llm()
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
    )

    print("\n=== Evaluation Results ===")
    df = result.to_pandas()
    avg_scores = df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
    print(f"faithfulness: {avg_scores['faithfulness']:.4f}")
    print(f"answer_relevancy: {avg_scores['answer_relevancy']:.4f}")
    print(f"context_precision: {avg_scores['context_precision']:.4f}")
    print(f"context_recall: {avg_scores['context_recall']:.4f}")

    df.to_csv("ragas_results_academic.csv", index=False)
    print("\nDetailed results saved to ragas_results_academic.csv")

if __name__ == "__main__":
    run_evaluation()