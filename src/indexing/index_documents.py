import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import logging
from parser import extract_text
from embeddings import DocumentIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/indexing/ -> src/ -> root
DOCS_FOLDER = os.path.join(BASE_DIR, "data/docs")
INDEX_FOLDER = os.path.join(BASE_DIR, "data/index")

# Ensure directories exist
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

def index_documents():
    """Index all documents in the docs folder."""
    indexer = DocumentIndexer()
    for file in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file)
        if os.path.isfile(file_path):
            logger.info(f"Processing: {file_path}")
            try:
                result = extract_text(file_path)
                text = result["text"]
                indexer.add_document(file_path, text)
            except Exception as e:
                logger.error(f"Skipped {file_path} due to error: {str(e)}")

    logger.info("Generating embeddings...")
    indexer.generate_embeddings()
    logger.info(f"Saving index to {INDEX_FOLDER}")
    indexer.save_index(INDEX_FOLDER)

if __name__ == "__main__":
    index_documents()