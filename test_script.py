import os
os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6K48BLsGRwkJEa5G1B8BwTWY7CYxNHr84vwK9O2HC-0dA"

from graph import claimtrace_app

questions = [
    'What is Acme Manufacturing payment score?',
    'What is Acme Manufacturing ownership?',
    'Is Acme Manufacturing the largest industrial equipment supplier in India?'
]

for q in questions:
    print(f'\n--- Testing: {q} ---')
    result = claimtrace_app.invoke({'question': q})
    print('AI Answer:', result['ai_answer'])
    for c, v in zip(result['claims'], result['verifications']):
        print(f'  Claim: {c} -> {v.result.value}')
    print('Conflict:', result.get('evidence_conflict'))
    print(f'Score: {result["confidence_score"]*100:.1f}%')
    print('Decision:', result['decision'].upper())
