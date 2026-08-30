import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_rag_retriever():
    """Loads documents, splits them, and stores them in a Vector Database."""
    
    # 1. Load Documents
    print("Loading documents...")
    documents = []
    data_dir = os.path.join("data", "documents")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    
    # 2. Split into Chunks
    print("Splitting into chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings & Vector Database
    print("Creating vector database with local HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # FAISS is our local vector database
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 4. Create a Retriever (Fetch the top 3 most relevant chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    return retriever

if __name__ == "__main__":
    # Test our RAG system
    retriever = setup_rag_retriever()
    
    question = "What is the payment score and ownership of Acme?"
    print(f"\nQuestion: {question}")
    
    # 5. Retrieve relevant chunks
    results = retriever.invoke(question)
    
    print("\n--- RETRIEVED EVIDENCE ---")
    for i, res in enumerate(results):
        print(f"\nChunk {i+1}:\n{res.page_content}")
