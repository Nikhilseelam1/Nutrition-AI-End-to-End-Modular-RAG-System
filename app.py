import streamlit as st
from rag.pipeline import ask
from main import initialize_pipeline

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Nutrition RAG Chatbot")


@st.cache_resource
def load_resources():
    return initialize_pipeline()


resources = load_resources()

embedding_model = resources["embedding_model"]
index = resources["index"]
pages_and_chunks = resources["pages_and_chunks"]
reranker = resources["reranker"]
tokenizer = resources["tokenizer"]
llm_model = resources["llm_model"]


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_query = st.chat_input("Ask a question...")

if user_query:
    with st.spinner("Thinking..."):
        answer, contexts = ask(
            query=user_query,
            embedding_model=embedding_model,
            index=index,
            pages_and_chunks=pages_and_chunks,
            reranker=reranker,
            tokenizer=tokenizer,
            llm_model=llm_model,
            top_k_retrieval=5,
            top_k_rerank=3,
            max_new_tokens=256,   
            temperature=0.7
        )

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("bot", answer))


for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)