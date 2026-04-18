# services/retrieval.py

class RetrievalService:
    def __init__(
        self, 
        vector_store, 
        bm25_store, 
        embedding_service, 
        reranker_service
    ):
        """
        Initialize the retrieval service with necessary database and model dependencies.
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service

    def search(self, query: str, k_initial: int = 20) -> list:
        """
        Execute the hybrid retrieval pipeline for a given query.
        """
        final_retrieved_chunks = []
        fused_pool = []

        print(f"🔍 Starting retrieval for query: '{query}'")
    
        # ---------------------------------------------------
        # Path A: Semantic Search (Dense)
        # ---------------------------------------------------
        vector_results = []

        query_embedding = self.embedding_service.get_embedding(query, task_type="query")

        if query_embedding is not None:
            vector_results = self.vector_store.search(query_embedding, top_k=k_initial)
            print(f"🔍 Vector search: found {len(vector_results)} results")
            # print(f"🔍 Vector results: {vector_results}")
        else:
            print("❌ Query embedding failed!")

        # ---------------------------------------------------
        # Path B: Keyword Search (Sparse)
        # ---------------------------------------------------
        bm25_results = self.bm25_store.search(query, top_k=k_initial)
        print(f"🔍 BM25 search: found {len(bm25_results)} results")
        # print(f"🔍 BM25 results: {bm25_results}")

        # ---------------------------------------------------
        # Path C: Reciprocal Rank Fusion (RRF)
        # ---------------------------------------------------
        if vector_results or bm25_results:
            print("✅ Entering RRF fusion")
            fused_pool = self.bm25_store.reciprocal_rank_fusion(
                vector_results, 
                bm25_results, 
                top_k=15, 
                k=60
            )
            print(f"🔗 Fused pool size: {len(fused_pool)}")
            # print(f"🔗 Fused pool: {fused_pool}")

        # ---------------------------------------------------
        # Path D: Precision Reranking
        # ---------------------------------------------------
        RERANK_THRESHOLD = -10.0

        if fused_pool:
            print("✅ Entering reranking")
            ranked_chunks = self.reranker_service.rerank(
                query=query, 
                candidates=fused_pool, 
                top_k=5
            )
            # Threshold check: Only return chunks if the best score meets the threshold
            if ranked_chunks and ranked_chunks[0]['rerank_score'] > RERANK_THRESHOLD:
                final_retrieved_chunks = ranked_chunks
            else:
                final_retrieved_chunks = []
                
            print(f"🎯 Reranker selected top chunks with best scores: {[c['rerank_score'] for c in final_retrieved_chunks]}")
        else:
            print("❌ No fused pool, skipping reranking")

        return final_retrieved_chunks
    