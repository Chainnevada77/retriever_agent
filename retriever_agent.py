# retriever_agent/retriever_agent.py
import faiss
import pickle
import numpy as np
import os
from typing import List, Dict, Union, Callable

# Clear console for better visibility
os.system('cls' if os.name == 'nt' else 'clear')

# Configuration (Update these paths to match your environment)
FAISS_INDEX_PATH = "/path/to/your/hospital_index/index.faiss"
METADATA_PATH = "/path/to/your/hospital_index/index.pkl"

def load_resources() -> tuple:
    """Load FAISS index and metadata with error handling"""
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"✅ FAISS index loaded | Dimension: {index.d} | Vectors: {index.ntotal}")
        
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"✅ Metadata loaded | Items: {len(metadata)}")
        
        return index, metadata
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e.filename}")
        raise
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        raise

try:
    index, metadata = load_resources()
except Exception:
    print("Failed to initialize retriever. Exiting.")
    exit(1)

def retrieve_top_k(query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top_k most similar items from FAISS index
    
    Args:
        query_embedding: Embedding vector (1D or 2D)
        top_k: Number of results to return
        
    Returns:
        List of metadata dictionaries
    """
    # Validate input
    if query_embedding.size == 0:
        raise ValueError("Empty query embedding provided")
    
    # Reshape and type conversion
    original_shape = query_embedding.shape
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    query_embedding = query_embedding.astype('float32')
    
    # Dimension check
    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Dimension mismatch: Query {query_embedding.shape[1]}D "
            f"vs Index {index.d}D"
        )
    
    # Adjust top_k if needed
    top_k = min(top_k, index.ntotal) if index.ntotal > 0 else 0
    
    # Perform search
    distances, indices = index.search(query_embedding, top_k)
    
    # Collect valid results
    results = []
    for i in indices[0]:
        if i >= 0 and i < len(metadata):
            results.append(metadata[i])
        else:
            print(f"⚠️ Skipped invalid index: {i}")
    
    print(f"🔍 Retrieved {len(results)}/{top_k} results")
    return results

def retrieve_context(
    user_query: str, 
    embedding_fn: Callable[[str], np.ndarray], 
    top_k: int = 5
) -> List[Dict]:
    """
    User-facing retrieval function
    
    Args:
        user_query: Natural language query
        embedding_fn: Function that converts text to embedding
        top_k: Number of results to return
        
    Returns:
        List of relevant context dictionaries
    """
    print(f"🔎 Query: '{user_query}'")
    query_embedding = embedding_fn(user_query)
    return retrieve_top_k(query_embedding, top_k)

if __name__ == "__main__":
    print("\n🏥 Retriever Agent Self-Test")
    
    # Test configuration
    TEST_DIM = 384
    test_embedding = np.random.rand(TEST_DIM).astype('float32')
    
    print(f"\n🧪 Test embedding shape: {test_embedding.shape}")
    print(f"🧪 Index dimension: {index.d}")
    print(f"🧪 Total vectors: {index.ntotal}")
    
    try:
        results = retrieve_top_k(test_embedding, top_k=3)
        print("\n📊 Results:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res.get('text', str(res))[:80]}...")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    
    print("\n✅ Retriever agent setup complete")