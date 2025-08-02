# Hospital RAG Chatbot - Retriever Agent

This agent handles document retrieval using FAISS for a hospital information chatbot.

## Features
- FAISS index management
- Metadata handling
- Top-k similarity search
- Robust error handling
- Self-test capability

## File Structure
retriever_agent/
├── retriever_agent.py
├── requirements.txt
└── README.md

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt

2. Configure paths in retriever_agent.py:
FAISS_INDEX_PATH = "/your/actual/index.faiss"
METADATA_PATH = "/your/actual/index.pkl"

3. Usage:
from agents.retriever_agent import retrieve_context

# Define your embedding function
def my_embedding_fn(text: str) -> np.ndarray:
    # Your embedding logic here
    return embedding_vector

# Retrieve context
results = retrieve_context(
    "What are visiting hours?",
    embedding_fn=my_embedding_fn,
    top_k=3
)

4. Run:
python3 retriever_agent/retriever_agent.py

5. Expected output:
🏥 Retriever Agent Self-Test

✅ FAISS index loaded | Dimension: 384 | Vectors: 5000
✅ Metadata loaded | Items: 5000

🧪 Test embedding shape: (384,)
🧪 Index dimension: 384
🧪 Total vectors: 5000
🔍 Retrieved 3/3 results

📊 Results:
1. Visiting hours are 9AM-8PM Monday through Saturday...
2. Emergency department operates 24/7 with...
3. Pediatric ward visiting hours are...

✅ Retriever agent setup complete

6. Error Handling:
File not found errors
Dimension mismatches
Invalid indices
Empty queries
Invalid top_k values