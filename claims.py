from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define the Pydantic Schema
class ExtractedClaims(BaseModel):
    """A list of factual claims extracted from an AI answer."""
    claims: list[str] = Field(
        description="List of individual factual claims made in the text."
    )

def extract_claims_from_answer(answer_text: str) -> list[str]:
    """Takes a paragraph of text and breaks it down into individual factual claims."""
    
    # 2. Define the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )
    
    # This is the magic part: We force the LLM to output our Pydantic schema
    structured_llm = llm.with_structured_output(ExtractedClaims)
    
    # 3. Define the Prompt
    template = """
You are a factual analysis AI.
Extract all individual factual claims from the following text.
Break compound sentences into separate, simple claims.

TEXT:
{text}
"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 4. Create Chain and Run
    chain = prompt | structured_llm
    
    result = chain.invoke({"text": answer_text})
    
    return result.claims

if __name__ == "__main__":
    # Test our Extractor
    sample_answer = "Acme is relatively low risk because its payment score is 82 and debt ratio is 32%."
    print(f"ORIGINAL ANSWER:\n{sample_answer}\n")
    
    print("[Thinking...] Extracting claims...")
    claims = extract_claims_from_answer(sample_answer)
    
    print("\n--- EXTRACTED CLAIMS ---")
    for i, claim in enumerate(claims):
        print(f"{i+1}. {claim}")
