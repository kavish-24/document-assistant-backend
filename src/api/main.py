import os
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.summarization.summarizer import DocumentSummarizer
from src.retrieval.search import DocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data/docs")
INDEX_FOLDER = os.path.join(BASE_DIR, "data/index")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

app = FastAPI()

# Fix CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
summarizer = DocumentSummarizer(use_ai=True)
retriever = DocumentRetriever(INDEX_FOLDER)  # Move outside endpoint

class SummarizationRequest(BaseModel):
    filename: str

class SearchQuery(BaseModel):
    query: str

@app.get("/reindex/")
async def reindex():
    """Reindex all documents."""
    try:
        os.system(f"python src/indexing/index_documents.py")
        global retriever  # Update the retriever after reindexing
        retriever = DocumentRetriever(INDEX_FOLDER)
        return {"message": "Reindexing complete!"}
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")
# ... (previous imports and code remain unchanged)

# Existing code up to @app.get("/list_files/") remains the same...

# New endpoint to delete a file
@app.delete("/delete_file/{filename}")
async def delete_file(filename: str):
    """Deletes a file from the upload folder."""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found for deletion: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        os.remove(file_path)
        logger.info(f"File deleted successfully: {filename}")
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# New endpoint to rename a file
class RenameRequest(BaseModel):
    old_filename: str
    new_filename: str

@app.post("/rename_file/")
async def rename_file(request: RenameRequest):
    """Renames a file in the upload folder."""
    try:
        old_file_path = os.path.join(UPLOAD_FOLDER, request.old_filename)
        new_file_path = os.path.join(UPLOAD_FOLDER, request.new_filename)
        
        if not os.path.exists(old_file_path):
            logger.error(f"File not found for renaming: {old_file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        if os.path.exists(new_file_path):
            logger.error(f"File with new name already exists: {new_file_path}")
            raise HTTPException(status_code=400, detail="A file with the new name already exists")
        
        os.rename(old_file_path, new_file_path)
        logger.info(f"File renamed from {request.old_filename} to {request.new_filename}")
        return {"message": f"File renamed from {request.old_filename} to {request.new_filename}"}
    except Exception as e:
        logger.error(f"Error renaming file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error renaming file: {str(e)}")

# ... (remove the if __name__ == "__main__": block if it exists, as Render doesn't need it)
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles file uploads."""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"filename": file.filename, "message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/summarize/")
async def summarize_document(request: SummarizationRequest):
    """Summarizes a document from 'data/docs/' folder."""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, request.filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        summary = summarizer.summarize_from_file(file_path)
        return {"filename": request.filename, "summary": summary}
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/search/")
async def search_documents(query: SearchQuery):
    """Handles document search queries."""
    try:
        results = retriever.search(query.query)
        return {"results": results}
    except Exception as e:
        logger.exception(f"Error during search: {str(e)}")  # Full traceback
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}

@app.get("/list_files/")
async def list_files():
    """Returns a list of uploaded document files."""
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")