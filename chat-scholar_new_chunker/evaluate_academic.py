# evaluate_academic.py (初始版本：纯向量检索，无混合检索/重排/Agentic)
import os
import sys
import pandas as pd
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
# 从 CSV 加载测试问题与参考答案
# ------------------------------
def load_testset(csv_path="ragas_golden_testset.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ 未找到测试集文件: {csv_path}")
        print("   请先运行 generate_testset.py 生成测试集。")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if "user_input" not in df.columns or "reference" not in df.columns:
        print("❌ CSV 文件缺少 user_input 或 reference 列")
        sys.exit(1)
    questions = df["user_input"].tolist()
    references = df["reference"].tolist()
    print(f"✅ 从 {csv_path} 加载了 {len(questions)} 个测试问题")
    return questions, references

# ------------------------------
# 自定义 LLM（用于 RAGAs 指标计算）
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
    # 1. 加载测试问题与参考答案
    questions, references = load_testset()

    # 2. 加载向量库和基础服务
    print("Loading vector store...")
    vector_store = VectorStore.load()
    if vector_store is None:
        print("Error: No vector store found. Please upload ResNet.pdf and transformer.pdf before running evaluation.")
        return

    embedding_service = EmbeddingService()
    ai_service = AIService()

    answers = []
    contexts = []

    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question[:80]}...")

        # 初始版本：纯向量检索，top_k=3（与原始代码一致）
        query_emb = embedding_service.get_embedding(question)
        if query_emb is None:
            print(f"  Failed to get embedding for question {i+1}")
            answers.append("")
            contexts.append([])
            continue

        retrieved = vector_store.search(query_emb, top_k=3)
        context_texts = [chunk.get("text", "") for chunk in retrieved]
        contexts.append(context_texts)

        chat_history = [{"role": "user", "content": question}]
        answer = ai_service.generate_response(chat_history, document_chunks=retrieved)
        answers.append(answer)

    # 3. 构建 RAGAs 数据集
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references,
    }
    dataset = Dataset.from_dict(data)

    # 4. 运行 RAGAs 评估
    print("\nStarting RAGAs evaluation...")
    llm = get_llm()
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm,
    )

    # 5. 输出结果
    print("\n=== Evaluation Results ===")
    df = result.to_pandas()
    avg_scores = df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
    print(f"faithfulness: {avg_scores['faithfulness']:.4f}")
    print(f"answer_relevancy: {avg_scores['answer_relevancy']:.4f}")
    print(f"context_precision: {avg_scores['context_precision']:.4f}")
    print(f"context_recall: {avg_scores['context_recall']:.4f}")

    # 保存详细结果
    output_csv = "ragas_results_initial.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")

if __name__ == "__main__":
    run_evaluation()