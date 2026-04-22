import os
import tempfile
import shutil
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from query_type import handle_general_query, handle_document_query
from chat import initialize_llm, connect_chromadb_create_index, clear_chromadb_db

load_dotenv()

# --- FastAPI Setup ---
app = FastAPI()

# Allow frontend to access backend
origins = [
    "https://ai-frontend-hrcwf4gfdhdadhgh.swedencentral-01.azurewebsites.net/",  # Frontend URL
    "http://localhost:8501",  # For local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the index (initialized when a document is uploaded)
global_index = None
global_document_name = None
# Initialize the Azure LLM globally
llm = initialize_llm()

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


# --- Document Processing ---

def create_index_from_document(file_path: str) -> None:
    """Creates a vector index from the uploaded document using embedding model.

    Args:
        file_path: uploaded document file path.

    Raises:
        HTTPException on error.
    """
    global global_index, global_document_name
    try:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        global_index = connect_chromadb_create_index(documents)
        global_document_name = os.path.basename(file_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing document: {str(e)}") from e


# --- Querying ---

def get_document_answer(question: str) -> str:
    """Gets an answer to a question about the uploaded document."""
    global global_index
    if global_index is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet.")
    try:
        return handle_document_query(global_index, question, llm)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error querying document: {str(e)}") from e


def get_general_answer(question: str) -> str:
    """Gets an answer to a general question."""
    try:
        return handle_general_query(question, llm)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error with general query: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error with general query: {str(e)}") from e


# --- FastAPI Endpoints ---

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document and creates an index from it."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only PDF and TXT files are allowed."
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the 5MB limit. Your file is {len(contents) / (1024*1024):.1f}MB."
        )

    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        create_index_from_document(tmp_file_path)
        return JSONResponse(content={"message": f"Document '{file.filename}' uploaded and processed successfully."})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document upload or processing: {str(e)}") from e
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/document-query/")
async def document_query(question: str = Query(..., min_length=1, description="Question about the uploaded document")):
    """Asks a question about the uploaded document."""
    if not question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    answer = get_document_answer(question)
    return JSONResponse(content={"answer": answer})


@app.post("/general-query/")
async def general_query(question: str = Query(..., min_length=1, description="General question for the LLM")):
    """Asks a general question without document context."""
    if not question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    answer = get_general_answer(question)
    return JSONResponse(content={"answer": answer})


@app.get("/clear-index/")
async def clear_index():
    """Clears the document index (resets the document-specific chat)."""
    global global_index, global_document_name
    try:
        global_index = None
        global_document_name = None
        clear_chromadb_db()
        return JSONResponse(content={"message": "Document index cleared."})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error clearing index: {str(e)}") from e


@app.get("/status/")
async def status():
    """Check if document has been uploaded."""
    if global_document_name is not None:
        return JSONResponse(content={"message": f"Document '{global_document_name}' is uploaded.", "status": True})
    return JSONResponse(content={"message": "No Document uploaded.", "status": False})
