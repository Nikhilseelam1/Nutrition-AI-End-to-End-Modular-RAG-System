# Nutrition AI – End-to-End Modular RAG System

A fully modular, production-style Retrieval-Augmented Generation (RAG) system built from scratch that transforms a large academic nutrition textbook into a context-aware AI assistant.

This project demonstrates practical LLM engineering, semantic retrieval optimization, and GPU-accelerated local model inference without relying on external RAG frameworks.

---

## Project Overview

Nutrition AI converts a 1000+ page academic PDF into an intelligent chatbot by:

- Parsing and processing the full textbook
- Performing sentence-level semantic chunking
- Generating transformer-based embeddings
- Building a FAISS vector index for efficient similarity search
- Applying cross-encoder re-ranking to improve retrieval precision
- Generating grounded responses using a locally deployed LLM
- Serving everything through an interactive Streamlit interface

The system ensures responses remain grounded in document context and minimizes hallucinations.

---

## System Architecture

PDF → Sentence Splitting → Chunking → Embeddings → FAISS Index  
→ Semantic Retrieval → Cross-Encoder Re-Ranking → LLM Generation → Streamlit UI

Key design principles:
- Modular pipeline architecture
- Dependency injection
- GPU acceleration support
- Retrieval-first optimization
- Evaluation-driven development

---

## Tech Stack

- Python
- PyTorch (CUDA supported)
- HuggingFace Transformers
- SentenceTransformers (all-mpnet-base-v2)
- FAISS (Vector Similarity Search)
- Cross-Encoder (ms-marco-MiniLM)
- TinyLlama 1.1B (Local LLM)
- spaCy (Sentence Segmentation)
- PyMuPDF (PDF Parsing)
- Streamlit (Frontend UI)

---

## Installation

Create a Conda environment:

conda create -n rag python=3.10 -y  
conda activate rag  

Install dependencies:

pip install -r requirements.txt  

Install spaCy model:

python -m spacy download en_core_web_sm  

---

## Run the Application

streamlit run app.py  

Then open:

http://localhost:8501  

---

## Features

- Semantic search over 800+ document chunks
- Cross-encoder re-ranking for improved retrieval quality
- GPU acceleration support
- Local LLM inference (no external API dependency)
- Streaming chatbot responses
- Source citation display
- Clean modular architecture
- Evaluation metrics including Exact Match and Semantic Similarity

---

## Evaluation

Implemented evaluation strategies including:

- Exact Match scoring
- Semantic similarity analysis
- Retrieval hit rate measurement
- Context precision validation

The focus was on improving grounding and reducing hallucinated outputs.

---

## Why This Project Matters

This project reflects:

- Real-world LLM system engineering
- Deep understanding of retrieval importance in RAG pipelines
- GPU-based inference optimization
- Clean modular design for scalability
- End-to-end AI system implementation

Built entirely from scratch without external RAG frameworks.

---

## Future Improvements

- Persist FAISS index to disk
- Add conversational memory
- Multi-document support
- Query latency dashboard
- Containerized deployment
- REST API backend version

---

## Author

Nikhil Seelam  
AI / GenAI Engineer in Progress
