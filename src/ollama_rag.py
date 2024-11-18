import requests
import chromadb
from chromadb.utils import embedding_functions
import json
from typing import List, Dict, Any
import textwrap
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import re

SENTENCE_TRANSF_MODELS = ["all-MiniLM-L6-v2", "all-MiniLM-L12"]

# model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_embedding(text):
#     return model.encode(text)


BASE_URL = "http://127.0.0.1:11434"


class OllamaRAG:
    def __init__(self, base_url=BASE_URL, collection_name="documents", model = "all-MiniLM-L6-v2"):
        self.base_url = base_url
        self.is_sentence_transformer = model in SENTENCE_TRANSF_MODELS
        self.model_name = model
        if self.is_sentence_transformer:
            self.model = SentenceTransformer(model)
        else:
            self.model = model

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def get_embedding_ollama(self, text: str) -> List[float]:
        """Get embeddings using Ollama model."""
        url = f"{self.base_url}/api/embeddings"
        response = requests.post(url, json={
            "model": self.model,
            "prompt": text
        })
        if "embedding" not in response.json():
            print(response.keys())
            return []
        return response.json()["embedding"]

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using SentenceTransformer model."""
        if self.is_sentence_transformer:
            return self.model.encode(text).tolist()
        else:
            return self.get_embedding_ollama(text)
        
    
    def add_documents(self, texts: List[str], ids: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Get embeddings for all texts
        embeddings = [self.get_embedding(text) for text in tqdm(texts)]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
    

    def query(self, question: str, n_results: int = 3, verbose = True) -> str:
        """
        Perform RAG query:
        1. Create embedding for question
        2. Find similar documents
        3. Generate answer using context
        """
        # Get question embedding and find similar documents
        question_embedding = self.get_embedding(question)
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        
        # Construct prompt with context
        context = "\n".join(results['documents'][0])
        prompt = f"""Use the following context to answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer using Ollama
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": "phi", "prompt": prompt, "stream": True},
            stream=True
        )
        
        print(f"Quotes:\n{context}")
        text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get('response', '')
                if verbose:
                    print(f"{chunk}", end='', flush=True)
                text += chunk
                # yield json.loads(line.decode()).get('response', '')
        return text
        # return response.json()["response"]
