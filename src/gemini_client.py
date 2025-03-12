import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# ✅ Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Check if API key exists
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.error("❌ GEMINI_API_KEY not set in environment variables.")
    exit(1)

# ✅ Configure Gemini API
genai.configure(api_key=API_KEY)

def summarize_with_gemini(text: str, num_sentences: int = 5) -> str:
    """Summarize text using Gemini API."""
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        prompt = f"Summarize the following text in {num_sentences} sentences:\n\n{text}"
        response = model.generate_content(prompt)

        return response.text.strip() if hasattr(response, "text") else "Error: No summary returned."
    
    except Exception as e:
        logger.error(f"❌ Gemini summarization error: {str(e)}")
        return f"Error: {str(e)}"

# ✅ Test the script when run directly
if __name__ == "__main__":
    sample_text = """Artificial Intelligence (AI) is revolutionizing multiple industries. 
    From healthcare and finance to autonomous vehicles and customer service, AI is transforming how we live and work. 
    Machine learning and deep learning models allow AI systems to learn from data and improve over time. 
    This technological advancement has raised both excitement and ethical concerns regarding job displacement, data privacy, and AI bias."""
    
    print("🔍 **Test Summary:**")
    print(summarize_with_gemini(sample_text, num_sentences=3))
