from verifier import VerificationResult, ClaimVerification

def calculate_confidence(verifications: list[ClaimVerification], evidence_conflict: bool = False) -> tuple[float, str]:
    """
    Calculates a deterministic confidence score based on verification results.
    Returns the score (0.0 to 1.0) and the final routing decision.
    """
    
    if not verifications:
        return 0.0, "human_review"
        
    # 1. Map results to numeric scores
    # We use hardcoded math, NOT an LLM, for this step to ensure absolute predictability.
    score_mapping = {
        VerificationResult.SUPPORTED: 1.0,
        VerificationResult.CONTRADICTED: 0.4,
        VerificationResult.UNSUPPORTED: 0.0
    }
    
    # 2. Calculate the average
    total_score = 0.0
    for v in verifications:
        total_score += score_mapping[v.result]
        
    average_score = total_score / len(verifications)
    
    # --- NEW: CONFLICT PENALTY ---
    if evidence_conflict:
        # Cap the confidence at 65% if a conflict exists in the evidence
        average_score = min(average_score, 0.65)
    
    # 3. Determine the routing decision based on strict thresholds
    if average_score >= 0.80 and not evidence_conflict:
        decision = "answer"
    elif average_score >= 0.60 or evidence_conflict:
        # If there's a conflict but the score isn't terribly low, it's a warning
        decision = "warning"
    else:
        decision = "human_review"
        
    # As a failsafe, make absolutely sure conflicts never result in 'answer'
    if evidence_conflict and decision == "answer":
        decision = "warning"
        
    return average_score, decision

if __name__ == "__main__":
    # Test our Scoring system
    
    # Scenario: 2 Supported, 1 Contradicted (like our Acme example)
    test_verifications = [
        ClaimVerification(result=VerificationResult.SUPPORTED, explanation="OK"),
        ClaimVerification(result=VerificationResult.SUPPORTED, explanation="OK"),
        ClaimVerification(result=VerificationResult.CONTRADICTED, explanation="Conflict found")
    ]
    
    score, decision = calculate_confidence(test_verifications)
    
    print("--- CONFIDENCE SCORING ---")
    print(f"Total Claims: {len(test_verifications)}")
    print(f"Calculated Score: {score * 100:.1f}%")
    print(f"Routing Decision: {decision.upper()}")
