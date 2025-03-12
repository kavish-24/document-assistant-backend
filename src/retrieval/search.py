import os
import logging
import numpy as np
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory (root level)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_FOLDER = os.path.join(BASE_DIR, "data/index")

class DocumentRetriever:
    def __init__(self, index_folder: str = INDEX_FOLDER):
        """Initialize retriever with indexed documents."""
        self.index_folder = index_folder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.chunks: List[str] = []
        self.chunk_to_doc_map: List[str] = []
        self.load_index()

    def load_index(self) -> None:
        """Load stored embeddings & documents."""
        logger.info(f"Loading index from: {self.index_folder}")
        try:
            with open(os.path.join(self.index_folder, "documents.pkl"), "rb") as f:
                self.documents = pickle.load(f)
            with open(os.path.join(self.index_folder, "embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
            with open(os.path.join(self.index_folder, "chunks.pkl"), "rb") as f:
                self.chunks = pickle.load(f)
            with open(os.path.join(self.index_folder, "chunk_to_doc_map.pkl"), "rb") as f:
                self.chunk_to_doc_map = pickle.load(f)
            logger.info(f"Loaded {len(self.documents)} documents, {len(self.embeddings)} embeddings, {len(self.chunks)} chunks")
            if not self.documents or len(self.embeddings) == 0:
                raise ValueError("Index is empty! Run `index_documents.py` again.")
        except FileNotFoundError:
            logger.error("Index files missing! Run `index_documents.py` first.")
            raise
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents matching the query."""
        logger.info(f"Documents: {len(self.documents)}, Embeddings: {len(self.embeddings)}")
        # Fix: Use len() to check emptiness, avoiding NumPy truth value ambiguity
        if len(self.documents) == 0 or len(self.embeddings) == 0:
            logger.error("Index is empty!")
            return {"error": "Index is empty. Please re-run indexing."}

        logger.info(f"Searching for: {query}")
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode(query).reshape(1, -1)
            logger.info(f"Query embedding shape: {query_embedding.shape}")

            # Convert document embeddings to numpy array
            document_embeddings = np.array(self.embeddings)
            logger.info(f"Document embeddings shape: {document_embeddings.shape}")

            # Compute cosine similarity
            dot_product = np.dot(document_embeddings, query_embedding.T).flatten()
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)

            logger.info(f"Query norm: {query_norm}, Doc norms: {doc_norms[:5]}...")
            if query_norm == 0 or np.any(doc_norms == 0):
                logger.error("Zero norm detected!")
                return {"error": "Normalization failed due to zero norm."}

            similarities = dot_product / (query_norm * doc_norms)
            logger.info(f"Similarity scores: {similarities[:5]}...")

            k = min(top_k, len(similarities))
            logger.info(f"Computed k: {k}")
            sorted_indices = np.argsort(similarities)
            top_indices = sorted_indices[-k:][::-1]
            logger.info(f"Top indices: {top_indices}")

            results = []
            seen_docs = set()
            for idx in top_indices:
                doc_id = self.chunk_to_doc_map[idx]
                if doc_id not in seen_docs:
                    doc = next((d for d in self.documents if d['id'] == doc_id), None)
                    if doc:
                        results.append({
                            "document": doc,
                            "relevance": float(similarities[idx]),
                            "preview": self.chunks[idx] if idx < len(self.chunks) else ""
                        })
                        seen_docs.add(doc_id)
                if len(results) >= top_k:
                    break

            logger.info(f"Search results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

if __name__ == "__main__":
    retriever = DocumentRetriever()
    results = retriever.search("What is the main topic?", top_k=3)
    print(results)