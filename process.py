import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Set
import json
import re

class DocumentProcessor:
    def __init__(self):
        # Load spaCy model for Named Entity Recognition (NER)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentence transformer for converting text to embeddings
        # 'all-MiniLM-L6-v2' is a good balance of speed and quality
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    @staticmethod 
    def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split document into overlapping chunks
        
        Why chunking?
        - LLMs have context limits
        - Smaller chunks = more precise retrieval
        - Overlap ensures we don't lose context at chunk boundaries
        """
        print("Testing the Text: ",text)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
  

    def extract_entities_and_relations(self, text: str) -> Dict:
        """
        Extract entities and relationships from text using NLP
        
        This is the core function that transforms unstructured text into
        structured knowledge (entities and relationships)
        """
        # Process text with spaCy pipeline
        doc = self.nlp(text)
        
        entities = []
        relations = []
        
        # ENTITY EXTRACTION
        # spaCy identifies different types of entities:
        # PERSON: People, including fictional
        # ORG: Companies, agencies, institutions
        # GPE: Countries, cities, states
        # PRODUCT: Objects, vehicles, foods, etc.
        # EVENT: Named hurricanes, battles, wars, sports events
        # MONEY: Monetary values, including unit
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY']:
                entities.append({
                    'text': ent.text,           # The actual entity text
                    'label': ent.label_,        # Entity type (PERSON, ORG, etc.)
                    'start': ent.start_char,    # Character position in text
                    'end': ent.end_char         # End character position
                })
        
        # RELATIONSHIP EXTRACTION
        # Use dependency parsing to find relationships
        # We look for Subject-Verb-Object patterns
        
        for token in doc:
            # Look for tokens that are subjects or objects of verbs
            if token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.head.pos_ == 'VERB':
                subject = token.text
                predicate = token.head.text  # The verb
                
                # Find the object of this verb
                for child in token.head.children:
                    if child.dep_ in ['dobj', 'pobj'] and child != token:
                        obj = child.text
                        relations.append({
                            'subject': subject,
                            'predicate': predicate,
                            'object': obj,
                            'confidence': 0.8  # Confidence score
                        })
        
        return {
            'entities': entities,
            'relations': relations,
            'text': text
        }
    
  
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with NLP
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\$\%]', '', text)
        return text.strip()