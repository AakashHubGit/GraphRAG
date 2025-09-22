import spacy
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Tuple, Set
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import json

load_dotenv()

class DocumentProcessor:
    """Handles text processing, chunking, and embedding generation"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def extract_entities_relations_llm(self, text: str) -> Dict:
        """Extract entities and relations using Gemini LLM"""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        Extract entities and relations from this text for a knowledge graph.

        Return JSON with:
        1. "entities": [{{"text": "entity_name", "label": "PERSON|ORG|GPE|MONEY|DATE|OTHER"}}]
        2. "relations": [{{"subject": "entity1", "predicate": "relation_type", "object": "entity2", "confidence": 0.0-1.0}}]

        Rules:
        - Extract all meaningful entities
        - Connect every entity to at least one other
        - Use descriptive predicates like "leads", "requests", "costs", "located_in"
        - Ensure high connectivity

        Text: "{text}"
        """

        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip("```json").strip("```").strip())
            return result
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return {"entities": [], "relations": []}