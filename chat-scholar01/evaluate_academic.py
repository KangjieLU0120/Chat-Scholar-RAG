# evaluate_academic.py (第一次优化：单次检索，无 Agentic 循环)
import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from app.services.embedding_service import EmbeddingService
from app.services.ai_service import AIService
from app.services.reranker_service import RerankerService
from app.utils.vector_store import VectorStore
from app.utils.bm25_store import BM25Store
# 导入 reciprocal_rank_fusion 函数（独立函数，不是类方法）
from app.utils.bm25_store import reciprocal_rank_fusion

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
def get_ragas_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api1.uiuiapi.com/v1"),
        temperature=0.0,
    )

# ------------------------------
# 混合检索 + 重排函数（与第一次优化实际流程一致）
# ------------------------------
def hybrid_retrieve(query, vector_store, bm25_store, embedding_service, reranker_service,
                    k_initial=20, top_k=5):
    """
    执行混合检索 + RRF 融合 + 重排，返回 top_k 个最终结果。
    """
    # 1. 向量检索
    query_emb = embedding_service.get_embedding(query, task_type="query")
    if query_emb is None:
        return []
    vector_results = vector_store.search(query_emb, top_k=k_initial) if vector_store else []

    # 2. BM25 检索
    bm25_results = bm25_store.search(query, top_k=k_initial) if bm25_store else []

    # 3. RRF 融合（使用独立函数）
    fused_pool = []
    if vector_results or bm25_results:
        fused_pool = reciprocal_rank_fusion(vector_results, bm25_results, top_k=15, k=60)

    # 4. 重排
    if fused_pool:
        reranked = reranker_service.rerank(query, fused_pool, top_k=top_k)
        # 可选阈值过滤（与第一次优化一致）
        if reranked and reranked[0].get('rerank_score', -float('inf')) > -10.0:
            return reranked
    return []

# ------------------------------
# 主评估函数
# ------------------------------
def run_evaluation():
    # 1. 加载测试问题与参考答案
    questions, references = load_testset()

    # 2. 加载组件
    print("Loading vector store...")
    vector_store = VectorStore.load()
    if vector_store is None:
        print("Error: No vector store found. Please upload ResNet.pdf and transformer.pdf before running evaluation.")
        return

    print("Loading BM25 store...")
    bm25_store = BM25Store.load()
    if bm25_store is None:
        print("Warning: BM25 store not loaded, will only use vector search.")

    print("Initializing embedding service...")
    embedding_service = EmbeddingService()

    print("Initializing reranker...")
    reranker_service = RerankerService()

    print("Creating AI service...")
    ai_service = AIService()   # 第一次优化版本 AIService 无参数

    # 3. 对每个问题执行检索并生成答案
    answers = []
    contexts = []

    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question[:80]}...")

        # 混合检索 + 重排
        retrieved = hybrid_retrieve(
            query=question,
            vector_store=vector_store,
            bm25_store=bm25_store,
            embedding_service=embedding_service,
            reranker_service=reranker_service,
            k_initial=20,
            top_k=5
        )

        # 提取文本列表用于 RAGAs 上下文
        context_texts = [chunk.get("text", "") for chunk in retrieved]
        contexts.append(context_texts)

        # 生成答案（非流式）
        chat_history = [{"role": "user", "content": question}]
        answer = ai_service.generate_response(chat_history, document_chunks=retrieved)
        answers.append(answer)

    # 4. 构建 RAGAs 数据集
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references,
    }
    dataset = Dataset.from_dict(data)

    # 5. 运行 RAGAs 评估
    print("\nStarting RAGAs evaluation...")
    llm = get_ragas_llm()
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

    # 6. 输出结果
    print("\n=== Evaluation Results ===")
    df = result.to_pandas()
    avg_scores = df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
    print(f"faithfulness: {avg_scores['faithfulness']:.4f}")
    print(f"answer_relevancy: {avg_scores['answer_relevancy']:.4f}")
    print(f"context_precision: {avg_scores['context_precision']:.4f}")
    print(f"context_recall: {avg_scores['context_recall']:.4f}")

    # 保存详细结果
    output_csv = "ragas_results_academic_v1.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")

if __name__ == "__main__":
    run_evaluation()