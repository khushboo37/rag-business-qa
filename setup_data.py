import os
import json

def setup_mock_data():
    """Creates synthetic business documents for our RAG system."""
    
    # Define our base folder for data
    data_dir = os.path.join("data", "documents")
    
    # Create the directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Our synthetic company data
    companies = [
        {
            "name": "Acme Manufacturing",
            "duns": "123-456-789",
            "profile": "Acme Manufacturing is a global supplier of industrial equipment.",
            "financials": "Payment score is 82. Debt ratio is 32%. Overall risk is relatively low.",
            # DELIBERATE CONFLICT: Source A says 60%, Source B says 45%
            "ownership": "State Registry lists Acme's primary ownership at 45%. However, the 2025 Annual Report claims ownership is 60%.",
            "compliance": "No active sanctions found."
        },
        {
            "name": "Globex Corporation",
            "duns": "987-654-321",
            "profile": "Globex Corporation is an import/export logistics firm.",
            "financials": "Payment score is 45. Debt ratio is 78%. Overall risk is high.",
            "ownership": "100% owned by Scorpio Holdings.",
            "compliance": "WARNING: Entity is currently on a trade watch list."
        },
        {
            "name": "Initech",
            "duns": "555-444-333",
            "profile": "Initech produces banking and financial software solutions.",
            "financials": "Payment score is 95. Debt ratio is 10%.",
            "ownership": "Publicly traded. Major shareholder owns 80%.",
            "compliance": "Clear. No sanctions."
        },
        {
            "name": "Umbrella Corp",
            "duns": "666-777-888",
            "profile": "Umbrella Corp is a multinational pharmaceutical company.",
            "financials": "Payment score is 60. Debt ratio is 50%.",
            "ownership": "51% owned by private equity.",
            "compliance": "Pending FDA review, but no financial sanctions."
        },
        {
            "name": "Stark Industries",
            "duns": "111-222-333",
            "profile": "Stark Industries is a defense and energy contractor.",
            "financials": "Payment score is 88. Debt ratio is 20%.",
            "ownership": "90% owned by Tony Stark.",
            "compliance": "Subject to strict government oversight, no violations."
        }
    ]
    
    # 1. Write the raw JSON file for reference
    json_path = os.path.join("data", "companies.json")
    with open(json_path, "w") as f:
        json.dump(companies, f, indent=4)
        
    # 2. Generate text documents for our RAG to read
    for company in companies:
        filename = company["name"].replace(" ", "_").lower() + ".txt"
        filepath = os.path.join(data_dir, filename)
        
        # We format the text nicely so the RAG can easily read it
        doc_content = f"""COMPANY: {company['name']}
DUNS NUMBER: {company['duns']}
PROFILE: {company['profile']}
FINANCIALS: {company['financials']}
OWNERSHIP: {company['ownership']}
COMPLIANCE: {company['compliance']}
"""
        with open(filepath, "w") as f:
            f.write(doc_content)
            
    print(f"Successfully generated {len(companies)} company documents in '{data_dir}'!")

if __name__ == "__main__":
    setup_mock_data()
