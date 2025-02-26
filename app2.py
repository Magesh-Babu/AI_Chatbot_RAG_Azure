import os
import tempfile
import shutil
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from llama_index.core import SimpleDirectoryReader
from query_type import handle_general_query, handle_document_query
from chat import initialize_llm, connect_chromadb_create_index

load_dotenv()

# --- FastAPI Setup ---
app = FastAPI()

# Global variable to store the index (initialized when a document is uploaded)
global_index = None
global_document_name = None
# Initialize the Azure LLM globally
llm = initialize_llm()

# --- Streamlit-related functions (modified for FastAPI context) ---
def process_document(file_path: str) -> str:
    """Processes a document and creates an index.

    Args:
        file_path: The path to the document file.

    Returns:
        A success message.
        Raises HTTPException on error.
    """
    global global_index
    global global_document_name

    try:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        global_index = connect_chromadb_create_index(documents)
        global_document_name = os.path.basename(file_path)
        return f"Document '{global_document_name}' uploaded and ingested successfully!"

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
    """Uploads a document (PDF or text) and processes it."""
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and text files are allowed.")

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

        # Process the document and create an index
        result = process_document(tmp_file_path)
        return JSONResponse(content={"message": result, "document_name": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove the temporary file after processing
        if os.path.exists(tmp_file_path):
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
    global_index = None
    global_document_name = None
    return JSONResponse(content={"message": "Document index cleared."})

@app.get("/status/")
async def status():
    """Check if document has been uploaded."""
    if global_document_name is not None:
        return JSONResponse(content={"message": f"Document '{global_document_name}' is uploaded.", "status": True})
    return JSONResponse(content={"message": "No Document uploaded.", "status": False})
