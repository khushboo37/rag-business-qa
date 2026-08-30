from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import setup_rag_retriever

def generate_business_answer(question: str, evidence: str) -> str:
    """Uses an LLM to answer a business question based ONLY on the provided evidence."""
    
    # 1. Define the LLM (Large Language Model)
    # We use a temperature of 0.0 because we want factual, deterministic answers, not creative ones.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0 
    )
    
    # 2. Define the Prompt (Instructions for the AI)
    template = """
You are a business intelligence AI agent.
Answer the user's question using ONLY the retrieved evidence below.
If the evidence does not contain the answer, say "I don't know based on the evidence."
Keep your answer concise and factual.
    
EVIDENCE:
{evidence}
    
QUESTION: 
{question}
    
ANSWER:
"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 3. Create the Chain (Connect Prompt -> LLM)
    # This pipe operator (|) connects the prompt template directly into the LLM.
    chain = prompt | llm
    
    # 4. Generate the output
    response = chain.invoke({
        "evidence": evidence, 
        "question": question
    })
    
    return response.content

if __name__ == "__main__":
    # Test the Agent
    print("Setting up RAG (this might take a second)...")
    retriever = setup_rag_retriever()
    
    question = "Is Acme Manufacturing a low-risk supplier?"
    print(f"\nUSER QUESTION: {question}")
    
    # Get evidence from our RAG module
    results = retriever.invoke(question)
    
    # Combine the retrieved chunks into one big string so the LLM can read it
    evidence_text = "\n\n".join([doc.page_content for doc in results])
    
    print("\n[Thinking...] Generating answer based on evidence...")
    
    # Generate the answer
    answer = generate_business_answer(question, evidence_text)
    
    print("\n--- AI ANSWER ---")
    print(answer)
