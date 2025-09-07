from typing import Dict
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_with_gemini(text: str) -> Dict:
    """
    Use Gemini LLM to extract entities and relations from text.
    Returns dict with 'entities' and 'relations'
    """
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    You are a knowledge graph builder. From the given text, extract:
    
    1. **Entities**:
       - Each must have 'text' (string) and 'label' 
         (choose from: PERSON, ORG, GPE, EVENT, PRODUCT, MONEY, DATE, OTHER).
       - Extract all the entities even simple entities, but avoid duplicates.

    2. **Relations**:
       - Each must have 'subject', 'predicate', 'object', 'confidence'.
       - Predicates can be anything you find useful some examples are:
         ["works_for", "part_of", "located_in", "created_by", "owns", "manages", 
          "founded", "participated_in", "collaborates_with", "invested_in", 
          "married_to", "related_to"].
       - Relations must always connect two entities from the extracted list.
       - Make sure to find a relation for each entity.

    3. **Connectivity Rule**:
       - Every entity MUST be connected to at least one other entity.
       - If no explicit relation is available, create a generic `"related_to"` relation 
         to link it to a relevant entity.
       - Connecte everything somehow to ensure a fully connected graph.

    ### STRICT OUTPUT:
    Return ONLY valid JSON in this format:
    {{
      "entities": [
        {{"text": "Google", "label": "ORG"}},
        {{"text": "Sundar Pichai", "label": "PERSON"}},
        {{"text": "California", "label": "GPE"}}
      ],
      "relations": [
        {{"subject": "Sundar Pichai", "predicate": "works_for", "object": "Google", "confidence": 0.95}},
        {{"subject": "Google", "predicate": "located_in", "object": "California", "confidence": 0.9}}
      ]
    }}

    ### TEXT:
    \"\"\"{text}\"\"\"
    """

    response = model.generate_content(prompt)
    print("LLM Response:", response.text)
    try:
        result = json.loads(response.text.strip("```")[4:].strip())
        return result
    except Exception as e:
        print("Error parsing LLM output:", e)
        return {"entities": [], "relations": []}
        