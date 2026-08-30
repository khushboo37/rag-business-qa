from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Import the functions we built in previous modules
from rag import setup_rag_retriever
from agent import generate_business_answer
from claims import extract_claims_from_answer
from verifier import verify_claim, ClaimVerification, detect_evidence_conflicts
from scoring import calculate_confidence

# Initialize RAG once when the script starts
retriever = setup_rag_retriever()

# 1. Define our Shared State
class WorkflowState(TypedDict):
    """This dictionary holds the data as it flows through the graph."""
    question: str
    evidence: str
    ai_answer: str
    claims: list[str]
    verifications: list[ClaimVerification]
    evidence_conflict: bool
    conflict_explanation: str
    confidence_score: float
    decision: str

# 2. Define the Nodes (The Steps in the Workflow)

def retrieve_node(state: WorkflowState):
    """Fetches evidence from the Vector DB."""
    results = retriever.invoke(state["question"])
    evidence = "\n\n".join([doc.page_content for doc in results])
    return {"evidence": evidence}

def generate_node(state: WorkflowState):
    """Generates the initial AI answer."""
    answer = generate_business_answer(state["question"], state["evidence"])
    return {"ai_answer": answer}

def extract_node(state: WorkflowState):
    """Extracts single claims from the AI's answer."""
    claims = extract_claims_from_answer(state["ai_answer"])
    return {"claims": claims}

def verify_node(state: WorkflowState):
    """Checks each claim against the original evidence."""
    verifications = []
    for claim in state["claims"]:
        result = verify_claim(claim, state["evidence"])
        verifications.append(result)
    return {"verifications": verifications}

def detect_conflict_node(state: WorkflowState):
    """Checks if the evidence itself contains contradictory facts."""
    conflict = detect_evidence_conflicts(state["evidence"], state.get("claims", []))
    return {
        "evidence_conflict": conflict.conflict_detected,
        "conflict_explanation": conflict.explanation
    }

def score_node(state: WorkflowState):
    """Calculates the final score and routing decision."""
    conflict = state.get("evidence_conflict", False)
    score, decision = calculate_confidence(state["verifications"], conflict)
    return {"confidence_score": score, "decision": decision}

# Dummy nodes to represent our final endpoints
def answer_node(state: WorkflowState): return {}
def warning_node(state: WorkflowState): return {}
def human_review_node(state: WorkflowState): return {}

# 3. Define the Routing Logic
def route_decision(state: WorkflowState):
    """Reads the decision and routes to the correct end point."""
    return state["decision"]

# 4. Build the Graph
workflow = StateGraph(WorkflowState)

# Add our nodes to the graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_answer", generate_node)
workflow.add_node("extract_claims", extract_node)
workflow.add_node("verify_claims", verify_node)
workflow.add_node("detect_conflict", detect_conflict_node)
workflow.add_node("calculate_confidence", score_node)
workflow.add_node("answer", answer_node)
workflow.add_node("warning", warning_node)
workflow.add_node("human_review", human_review_node)

# Connect the nodes in a straight sequence
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", "extract_claims")
workflow.add_edge("extract_claims", "verify_claims")
workflow.add_edge("verify_claims", "detect_conflict")
workflow.add_edge("detect_conflict", "calculate_confidence")

# Conditional routing at the very end
workflow.add_conditional_edges(
    "calculate_confidence",
    route_decision,
    {
        "answer": "answer",
        "warning": "warning",
        "human_review": "human_review"
    }
)

# Connect the final endpoints to END
workflow.add_edge("answer", END)
workflow.add_edge("warning", END)
workflow.add_edge("human_review", END)

# Compile the graph into a runnable application
claimtrace_app = workflow.compile()

if __name__ == "__main__":
    # Test the whole graph!
    test_question = "What is the ownership structure of Acme Manufacturing?"
    print(f"Running graph for question: '{test_question}'\n")
    
    # We pass the initial state (just the question)
    # The graph will run automatically from START to END
    final_state = claimtrace_app.invoke({"question": test_question})
    
    print("\n--- FINAL GRAPH OUTPUT ---")
    print(f"AI Answer: {final_state['ai_answer']}")
    print(f"Confidence: {final_state['confidence_score'] * 100:.1f}%")
    print(f"Decision:   {final_state['decision'].upper()}")
