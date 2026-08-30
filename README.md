# RAG-Based Business Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers business-related questions using information retrieved from a provided knowledge base.

The system combines document retrieval, embeddings, vector search, prompt engineering, and a Generative AI model to produce context-grounded responses.

## 🎯 Objective

Large Language Models can generate incorrect or unsupported information when answering questions outside their reliable knowledge.

This project uses a RAG architecture to:

* Retrieve relevant information from a knowledge base
* Provide the retrieved information as context to the LLM
* Generate answers grounded in the retrieved evidence
* Reduce unsupported or hallucinated responses

## 🏗️ Architecture

```text
                 ┌─────────────────┐
                 │   User Query    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Query Embedding │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Vector Store    │
                 │ / Retriever     │
                 └────────┬────────┘
                          │
                   Relevant Chunks
                          │
                          ▼
                 ┌─────────────────┐
                 │ Prompt + Context│
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │      LLM        │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Final Answer    │
                 └─────────────────┘
```

## 🔄 RAG Pipeline

1. Load source documents
2. Split documents into smaller chunks
3. Generate embeddings for the chunks
4. Store embeddings in a vector store
5. Convert the user's question into an embedding
6. Retrieve semantically relevant chunks
7. Pass retrieved context to the LLM
8. Generate a context-grounded response

## 🛠️ Technologies

* Python
* LangChain
* Google Gemini
* Generative AI
* RAG
* Embeddings
* Vector Search
* Prompt Engineering

## 📁 Project Structure

```text
src/
├── agent.py          # LLM interaction and response generation
├── rag.py            # Retrieval pipeline
├── embeddings.py     # Embedding configuration
└── config.py         # Application configuration

data/
└── sample_documents/ # Knowledge-base documents

tests/
└── test_rag.py       # Basic tests
```

## ⚙️ Setup

Clone the repository:

```bash
git clone <repository-url>
cd rag-business-qa
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add the required API credentials.

## 🔐 Environment Variables

```text
GOOGLE_API_KEY=your_api_key_here
```

Never commit API keys or `.env` files to GitHub.

## ▶️ Running the Project

```bash
python src/agent.py
```

Enter a business-related question when prompted.

## 💡 Example

**Question:**

```text
What are the key services provided by the company?
```

**Process:**

```text
Question
   ↓
Embedding
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
Prompt
   ↓
Gemini
   ↓
Grounded Answer
```

## 🚀 Future Improvements

Potential improvements include:

* LangGraph-based agentic workflow
* Qdrant vector database for scalable retrieval
* Hybrid search
* Retrieval reranking
* RAG evaluation metrics
* API-based deployment
* Response monitoring and observability
* Cloud deployment using AWS/GCP

## 📌 Learning Outcomes

Through this project, I gained practical experience in:

* Building a RAG pipeline
* Working with LLMs and embeddings
* Integrating LangChain with Generative AI models
* Designing retrieval-based AI applications
* Prompt engineering
* Understanding hallucination reduction through grounding

