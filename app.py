import streamlit as st

# Set up the simple page MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="ClaimTrace", layout="centered")

from dotenv import load_dotenv
load_dotenv() # Load variables from .env file

from graph import claimtrace_app
st.title("ClaimTrace")
st.subheader("Business AI Reasoning Verification")

st.markdown("""
*RAG gives the AI evidence. ClaimTrace checks whether the AI actually used that evidence correctly.*
""")

# User Inputs
st.markdown("Ask a question about a company to see ClaimTrace verify the AI's reasoning.")
question = st.text_input("Question:", "What is the ownership structure of Acme Manufacturing?")

if st.button("Analyze"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving evidence and running ClaimTrace..."):
            
            try:
                # Run the LangGraph pipeline
                final_state = claimtrace_app.invoke({"question": question})
                
                st.divider()
                
                # 1. AI Answer
                st.markdown("### 🤖 AI ANSWER")
                st.info(final_state["ai_answer"])
                
                st.divider()
                
                # 2. Claim Verification
                st.markdown("### 🔍 CLAIM VERIFICATION")
                
                # Zip pairs up each claim with its corresponding verification result
                for claim, verification in zip(final_state["claims"], final_state["verifications"]):
                    
                    # Choose icon based on result
                    if verification.result == "SUPPORTED":
                        icon = "✅"
                    elif verification.result == "CONTRADICTED":
                        icon = "❌"
                    else:
                        icon = "⚠️"
                    
                    st.markdown(f"**{icon} {claim}**")
                    st.caption(f"**Result:** {verification.result.value}  \n**Reason:** {verification.explanation}")
                    st.write("") # small spacing
                    
                st.divider()
                
                # --- NEW: EVIDENCE CONFLICT ---
                if final_state.get("evidence_conflict"):
                    st.markdown("### ⚠️ EVIDENCE CONFLICT")
                    st.warning(final_state["conflict_explanation"])
                    st.write("")
                    st.divider()
                
                # 3. Confidence & Decision
                st.markdown("### 📊 CONFIDENCE & ROUTING")
                score_pct = final_state["confidence_score"] * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Score", f"{score_pct:.1f}%")
                
                with col2:
                    decision = final_state["decision"].upper()
                    if decision == "ANSWER":
                        st.success(f"Decision: {decision}")
                    elif decision == "WARNING":
                        st.warning(f"Decision: {decision}")
                    else:
                        st.error(f"Decision: {decision} RECOMMENDED")
                
                st.divider()
                
                # --- NEW: AUDIT TRAIL ---
                with st.expander("View ClaimTrace Audit Trail"):
                    st.markdown("### 📜 CLAIMTRACE AUDIT TRAIL")
                    st.markdown("1. **Question** received from user")
                    st.markdown("   ↓")
                    st.markdown("2. **RAG** retrieved evidence from vector database")
                    st.markdown("   ↓")
                    st.markdown("3. **AI** generated answer using evidence")
                    st.markdown("   ↓")
                    st.markdown("4. **Claims** extracted into structured list")
                    st.markdown("   ↓")
                    st.markdown("5. **Claims verified** individually by LLM judge")
                    
                    if final_state.get("evidence_conflict"):
                        st.markdown("   ↓")
                        st.markdown("6. ⚠️ **Evidence conflict detected**")
                        st.markdown(f"   *Explanation: {final_state['conflict_explanation']}*")
                        st.markdown("   ↓")
                        st.markdown("7. 📉 **Confidence capped at 65%** because conflicting evidence was detected.")
                    else:
                        st.markdown("   ↓")
                        st.markdown("6. ✅ No evidence conflicts detected")
                        
                    st.markdown("   ↓")
                    if final_state["decision"] == "answer":
                        st.markdown("8. 🟢 **Final Decision: Clean answer provided to user**")
                    elif final_state["decision"] == "warning":
                        st.markdown("8. 🟡 **Final Decision: Answer provided with warning**")
                    else:
                        st.markdown("8. 🔴 **Final Decision: Human Review**")
                
                # 4. Retrieved Evidence (For transparency)
                with st.expander("View Retrieved Evidence (Raw Data)"):
                    st.text(final_state["evidence"])
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Check your terminal for API key errors or missing packages.")
