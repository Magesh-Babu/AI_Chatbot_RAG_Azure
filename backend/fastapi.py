import os
import tempfile
import shutil
from typing import List
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from modules.query_type import handle_general_query, handle_document_query
from modules.chat import initialize_llm, connect_chromadb_create_index, clear_chromadb_db

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import chromadb

load_dotenv()

# --- FastAPI Setup ---
app = FastAPI()

# Global variable to store the index (initialized when a document is uploaded)
global_index = None
global_document_name = None
# Initialize the Azure LLM globally
llm = initialize_llm()

# --- Document Processing ---

def create_index_from_document(file_path: str) -> None:
    """Creates a vector index from the uploaded document."""
    global global_index, global_document_name
    try:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        global_index = connect_chromadb_create_index(documents)
        global_document_name = os.path.basename(file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


def get_document_answer(question: str) -> str:
    """Gets an answer to a question about the uploaded document.

    Args:
        question: The question to ask.

    Returns:
        The answer to the question.
        Raises HTTPException on error.
    """
    global global_index
    if global_index is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet.")
    try:
      
        answer = handle_document_query(global_index, question, llm)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")


def get_general_answer(question: str) -> str:
    """Gets an answer to a general question.

    Args:
        question: The question to ask.

    Returns:
        The answer to the question.
        Raises HTTPException on error.
    """
    try:
        answer = handle_general_query(question, llm)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with general query: {str(e)}")


# --- FastAPI Endpoints ---

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document and creates an index from it."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

        create_index_from_document(tmp_file_path)
        return JSONResponse(content={"message": f"Document '{file.filename}' uploaded and processed successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document upload or processing: {str(e)}")
    finally:
        if 'tmp_file_path' in locals():
            os.remove(tmp_file_path)

@app.post("/document-query/")
async def document_query(question: str):
    """Asks a question about the uploaded document."""
    answer = get_document_answer(question)
    return JSONResponse(content={"answer": answer})

@app.post("/general-query/")
async def general_query(question: str):
    """Asks a general question (no document needed)."""
    answer = get_general_answer(question)
    return JSONResponse(content={"answer": answer})

@app.get("/clear-index/")
async def clear_index():
    """Clears the document index (resets the document-specific chat)."""
    global global_index
    global global_document_name
    try:
        global_index = None
        global_document_name = None
        clear_chromadb_db()
        return JSONResponse(content={"message": "Document index cleared."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during clearing index: {str(e)}")

@app.get("/status/")
async def status():
    """Check if document has been uploaded."""
    if global_document_name is not None:
        return JSONResponse(content={"message": f"Document '{global_document_name}' is uploaded.", "status": True})
    return JSONResponse(content={"message": "No Document uploaded.", "status": False})

