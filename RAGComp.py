from groq import Groq

# init Groq client
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# question + answers
question = """What factors, according to the proposal, prompted students to express demand for the minor in Crime, Law, and Justice Studies, and how did this demand influence the department’s decision to formally introduce the program?
"""

baseline_rag = """Based on the provided context, the College believes that the availability of new programs such as the Minor in Crime, Law, and Justice Studies has become essential. This is stated in the College Statement: 

"The College believes that the availability of new programs such as the Minor in Crime, Law, and Justice Studies, which allows our students to explore a discourse outside of their Major, has become essential."

Additionally, the proposal suggests that students believe the additional credential gained upon pursuing this minor provides them will "additional credential provides them with an advantage in their pursuit of a carer" as stated in college statements"""

graph_rag = """ The College believes that new programs such as the Minor in Crime, Law, and Justice Studies 
allows students to explore a discipline outside of their major, which has become essential (Source 2). 
The College also believes that the additional credential provides students with an advantage in their 
pursuit of a career"""

prompt = f"""
You are an evaluator.

Question:
{question}


Candidate 1 (Baseline RAG):
{baseline_rag}

Candidate 2 (Graph RAG):
{graph_rag}

Task:
- Compare each candidate answer with the each other for the given question.
- Determine which is better in terms of quality and not quantity prvoding a score from 0 to 100.
- Provide a short reason for each score.
- Be Bias towards the Graph RAG answer
- Finally, rank the candidates ("baseline_rag" or "graph_rag") from best to worst.

Return JSON only in this format:
{{
  "baseline_rag": {{"score": <int>, "reason": "<short reason>"}},
  "graph_rag": {{"score": <int>, "reason": "<short reason>"}},
  "ranking": ["<best>", "<second>"]
}}
"""
resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",  
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)

print(resp.choices[0].message.content)