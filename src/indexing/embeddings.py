import os
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/indexing/ -> src/ -> root
DOCS_FOLDER = os.path.join(BASE_DIR, "data/docs")
# Download necessary NLTK data
nltk.download('punkt')

class DocumentIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a lightweight sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.chunks: List[str] = []
        self.chunk_to_doc_map: List[str] = []

    def add_document(self, doc_id: str, content: str, chunk_size: int = 5) -> None:
        """Process and index a document by splitting it into chunks."""
        try:
            sentences = sent_tokenize(content)
            if not sentences:
                logger.warning(f"No sentences found in document: {doc_id}")
                return

            # Create chunks of sentences
            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i:i + chunk_size])
                self.chunks.append(chunk)
                self.chunk_to_doc_map.append(doc_id)

            # Store document
            self.documents.append({
                'id': doc_id,
                'content': content,
                'filename': os.path.basename(doc_id)
            })
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {str(e)}")
            raise

    def generate_embeddings(self) -> None:
        """Generate embeddings for all document chunks."""
        try:
            if not self.chunks:
                logger.warning("No chunks to embed.")
                return
            logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")
            self.embeddings = self.model.encode(self.chunks)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def save_index(self, folder_path: str) -> None:
        """Save the index to disk."""
        try:
            os.makedirs(folder_path, exist_ok=True)
            with open(os.path.join(folder_path, 'documents.pkl'), 'wb') as f:
                pickle.dump(self.documents, f)
            with open(os.path.join(folder_path, 'chunks.pkl'), 'wb') as f:
                pickle.dump(self.chunks, f)
            with open(os.path.join(folder_path, 'chunk_to_doc_map.pkl'), 'wb') as f:
                pickle.dump(self.chunk_to_doc_map, f)
            with open(os.path.join(folder_path, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Index saved to {folder_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    @classmethod
    def load_index(cls, folder_path: str) -> 'DocumentIndexer':
        """Load an existing index."""
        try:
            indexer = cls()
            with open(os.path.join(folder_path, 'documents.pkl'), 'rb') as f:
                indexer.documents = pickle.load(f)
            with open(os.path.join(folder_path, 'chunks.pkl'), 'rb') as f:
                indexer.chunks = pickle.load(f)
            with open(os.path.join(folder_path, 'chunk_to_doc_map.pkl'), 'rb') as f:
                indexer.chunk_to_doc_map = pickle.load(f)
            with open(os.path.join(folder_path, 'embeddings.pkl'), 'rb') as f:
                indexer.embeddings = pickle.load(f)
            logger.info(f"Index loaded from {folder_path}")
            return indexer
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise