# rag/pipeline.py

from rag.retreval import retrieve_relevant_resources
from rag.reranker import rerank_results
from rag.generation import generate_answer


def ask(
    query,
    embedding_model,
    index,
    pages_and_chunks,
    reranker,
    tokenizer,
    llm_model,
    top_k_retrieval,
    top_k_rerank,
    max_new_tokens,
    temperature
):
    scores, indices, context_items = retrieve_relevant_resources(
        query=query,
        embedding_model=embedding_model,
        index=index,
        pages_and_chunks=pages_and_chunks,
        top_k=top_k_retrieval
    )

    reranked_contexts = rerank_results(
        query=query,
        context_items=context_items,
        reranker=reranker,
        top_k=top_k_rerank
    )

    # Step 3: Generation
    answer = generate_answer(
        query=query,
        context_items=reranked_contexts,
        tokenizer=tokenizer,
        llm_model=llm_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    return answer, reranked_contexts
