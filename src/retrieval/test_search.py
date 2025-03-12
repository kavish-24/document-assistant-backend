import sys
import os

# Ensure Python can find all project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

# Now import the required modules
from src.indexing.embeddings import DocumentIndexer
from src.retrieval.search import DocumentRetriever
from src.summarization.summarizer import DocumentSummarizer

# Load the pre-built index
index_folder = os.path.abspath(os.path.join(current_dir, "../../data/index"))
indexer = DocumentIndexer.load_index(index_folder)
retriever = DocumentRetriever(indexer)
summarizer = DocumentSummarizer()

# Test query
query = input("Enter your search query: ")
results = retriever.search(query, top_k=5)

# Display results
print("\nSearch Results:")
for result in results:
    print(f"\n📄 Document: {result['document']['filename']}")
    print(f"🔍 Relevance: {result['relevance']:.2f}")
    print(f"📝 Preview: {result['preview']}")

    # Generate and display a summary
    summary = summarizer.summarize(result['preview'], num_sentences=2)  # Summarize only preview
    print(f"📌 Summary: {summary}")

