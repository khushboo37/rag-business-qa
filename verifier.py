from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define the possible outcomes strictly
class VerificationResult(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"

# 2. Define the Pydantic Schema for structured output
class ClaimVerification(BaseModel):
    """The result of verifying a single claim against evidence."""
    result: VerificationResult = Field(
        description="Whether the claim is SUPPORTED, CONTRADICTED, or UNSUPPORTED by the evidence."
    )
    explanation: str = Field(
        description="A short 1-sentence explanation of why this result was chosen."
    )

def verify_claim(claim: str, evidence: str) -> ClaimVerification:
    """Uses 'LLM-as-a-judge' to verify a single claim against the evidence."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )
    
    # Force the LLM to output our exact ClaimVerification object
    structured_llm = llm.with_structured_output(ClaimVerification)
    
    # 3. Define the Judge Prompt
    template = """
You are a strict fact-checking AI. 
Compare the CLAIM against the provided EVIDENCE.

Rules:
- If the EVIDENCE directly proves the claim is true, output SUPPORTED.
- If the EVIDENCE proves the claim false, OR contains conflicting/disputed information about the claim, output CONTRADICTED.
- If the EVIDENCE does not mention the information needed to verify the claim, output UNSUPPORTED.
- Do NOT use outside knowledge. You are strictly grading the claim based on the text provided.

EVIDENCE:
{evidence}

CLAIM:
{claim}
"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | structured_llm
    
    return chain.invoke({"evidence": evidence, "claim": claim})

# --- NEW: CONFLICT DETECTION ---

import re

class ConflictDetection(BaseModel):
    """The result of checking the evidence for internal conflicts."""
    conflict_detected: bool = Field(
        description="True if the evidence contains conflicting or contradictory information about the same business fact from different sources."
    )
    explanation: str = Field(
        description="If a conflict is detected, briefly explain what it is. If not, return 'No conflicts detected.'"
    )

def detect_evidence_conflicts(evidence: str, claims: list[str]) -> ConflictDetection:
    """Uses deterministic Python logic to detect if the retrieved evidence contains conflicting facts relevant to the claims."""
    
    conflict_detected = False
    explanation = "No conflicts detected."
    
    # Combine all claims into a single lowercase string for easy keyword checking
    claims_text = " ".join(claims).lower()
    
    # Because our synthetic data has predictable business fields (e.g., "OWNERSHIP: ..."),
    # we can detect conflicts by parsing these structured values.
    
    # Check for ownership conflicts ONLY if the AI's claims actually mention ownership
    if "own" in claims_text:
        ownership_match = re.search(r"OWNERSHIP:\s*(.*)", evidence, re.IGNORECASE)
        if ownership_match:
            ownership_text = ownership_match.group(1)
            
            # Extract all percentage values from the ownership text
            percentages = re.findall(r"(\d+)%", ownership_text)
            
            # If we find more than one UNIQUE percentage, there is a conflict!
            unique_percentages = set(percentages)
            if len(unique_percentages) > 1:
                conflict_detected = True
                explanation = f"Two sources report different ownership values for the same company ({'%, '.join(percentages)}%). ClaimTrace cannot determine which value should be treated as authoritative."
            
    return ConflictDetection(
        conflict_detected=conflict_detected, 
        explanation=explanation
    )

if __name__ == "__main__":
    # Test our Verifier with the deliberate conflict we created in Module 1
    test_evidence = "Acme ownership is disputed. State Registry says 45%. Annual Report says 60%. Payment score is 82."
    
    print(f"--- RAG EVIDENCE ---\n{test_evidence}\n")
    
    test_claims = [
        "Acme has a payment score of 82.",
        "Acme ownership is exactly 60%.",
        "Acme is the safest supplier in the industry."
    ]
    
    print("--- VERIFICATION ---")
    for claim in test_claims:
        print(f"Claim:  '{claim}'")
        verification = verify_claim(claim, test_evidence)
        print(f"Result: {verification.result.value}")
        print(f"Reason: {verification.explanation}\n")
