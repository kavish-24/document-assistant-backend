import os
import sys
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import numpy as np
import logging
from typing import Optional
import google.generativeai as genai
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
import networkx as nx
from src.indexing.parser import extract_text

# Configure logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory (root level)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/summarization/ -> src/ -> root
DOCS_FOLDER = os.path.join(BASE_DIR, "data/docs")

# Load API Key from Environment
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set. Falling back to extractive summarization.")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

class DocumentSummarizer:
    def __init__(self, use_ai: bool = True):
        """Initialize summarizer with option to use AI or extractive method."""
        self.use_ai = use_ai and API_KEY is not None
        if not self.use_ai:
            logger.info("Using extractive summarization as fallback.")
            self.stop_words = set(stopwords.words('english'))

    def summarize(self, text: str, num_sentences: int = 5) -> str:
        """Summarizes given text using AI (Gemini) or extractive method."""
        if self.use_ai:
            return self._summarize_with_ai(text, num_sentences)
        else:
            return self._summarize_with_pagerank(text, num_sentences)

    def _summarize_with_ai(self, text: str, num_sentences: int) -> str:
        """Summarize text using Gemini API."""
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = f"Summarize the following text in {num_sentences} sentences:\n\n{text}"
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
            else:
                logger.error("Gemini API returned no summary.")
                return "Error: No summary returned by Gemini API."
        except Exception as e:
            logger.error(f"Error in Gemini summarization: {str(e)}")
            return f"Error: {str(e)}"

    def _summarize_with_pagerank(self, text: str, num_sentences: int) -> str:
        """Summarize text using extractive method (TextRank)."""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return text

            # Create sentence similarity matrix
            similarity_matrix = self._build_similarity_matrix(sentences)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)

            # Sort sentences by score
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]
            return " ".join(summary_sentences)
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return f"Error: {str(e)}"

    def _build_similarity_matrix(self, sentences: list) -> np.ndarray:
        """Build sentence similarity matrix for TextRank."""
        num_sentences = len(sentences)
        similarity_matrix = np.zeros((num_sentences, num_sentences))
        for i in range(num_sentences):
            for j in range(num_sentences):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
        return similarity_matrix

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Compute similarity between two sentences."""
        words1 = [w.lower() for w in sent1.split() if w.lower() not in self.stop_words]
        words2 = [w.lower() for w in sent2.split() if w.lower() not in self.stop_words]
        all_words = list(set(words1 + words2))
        vector1 = [words1.count(w) for w in all_words]
        vector2 = [words2.count(w) for w in all_words]
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        norm1 = sum(v ** 2 for v in vector1) ** 0.5
        norm2 = sum(v ** 2 for v in vector2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def summarize_from_file(self, file_path: str, num_sentences: int = 5) -> str:
        """Summarize a document from a file."""
        try:
            result = extract_text(file_path)
            text = result["text"]
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return "Could not extract text from document."
            return self.summarize(text, num_sentences)
        except Exception as e:
            logger.error(f"Error in summarize_from_file: {str(e)}")
            return f"Error: {str(e)}"

if __name__ == "__main__":
    summarizer = DocumentSummarizer(use_ai=True)
    summary = summarizer.summarize_from_file("data/docs/p2.pdf")
    print(summary)