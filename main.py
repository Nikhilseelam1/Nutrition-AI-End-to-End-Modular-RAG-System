from rag.document_Loader import load_pdf
from rag.chunking import create_sentence_chunks
from rag.embedding import load_embedding_model, generate_embeddings
from rag.index import build_faiss_index
from rag.reranker import load_reranker
from rag.model_loader import load_llm_model
from rag.sentence_splitter import add_sentences_to_pages


def initialize_pipeline():
    pdf_path = "data/human-nutrition-text.pdf"
    num_sentence_chunk_size = 10

    pages_and_texts = load_pdf(pdf_path)

    pages_and_texts = add_sentences_to_pages(pages_and_texts)

    pages_and_texts = create_sentence_chunks(
        pages_and_texts,
        num_sentence_chunk_size
    )

    pages_and_chunks = []
    for item in pages_and_texts:
        for chunk in item["sentence_chunks"]:
            pages_and_chunks.append({
                "sentence_chunk": " ".join(chunk),
                "page_number": item["page_number"]
            })

    embedding_model = load_embedding_model(
        model_name="all-mpnet-base-v2",
        device="cpu"
    )

    sentence_chunks = [item["sentence_chunk"] for item in pages_and_chunks]

    embeddings = generate_embeddings(
        embedding_model,
        sentence_chunks
    )

    index = build_faiss_index(embeddings)

    reranker = load_reranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    tokenizer, llm_model = load_llm_model()

    return {
        "embedding_model": embedding_model,
        "index": index,
        "pages_and_chunks": pages_and_chunks,
        "reranker": reranker,
        "tokenizer": tokenizer,
        "llm_model": llm_model
    }