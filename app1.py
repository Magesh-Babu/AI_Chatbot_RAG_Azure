import streamlit as st
import tempfile
import os
from llama_index.core import SimpleDirectoryReader
from query_type import handle_general_query, handle_document_query
from chat import display_chat, clear_chat_history, initialize_llm, connect_chromadb_create_index
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import shutil

PORT = os.getenv("PORT")

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

# --- Streamlit App (modified to interact with FastAPI) ---
# Streamlit now only shows a basic UI to interact with the API
def main():
    st.title("AI Chatbot (FastAPI Backend)")
    st.markdown("This chatbot uses a FastAPI backend to handle document processing and questions.")

    #status of document
    response_status = requests.get("http://localhost:8000/status/")
    if response_status.status_code == 200:
        data = response_status.json()
        if data["status"]:
            st.write(data["message"])
        else:
             st.write(data["message"])

    # File uploader to call /upload-document/
    uploaded_file = st.file_uploader("Upload a document (PDF or text)", type=["pdf", "txt"])
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post("http://localhost:8000/upload-document/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(data["message"])
        else:
            st.error(f"Error uploading document: {response.text}")

    # Chat input for document-specific questions
    if st.session_state.get("document_uploaded"):
        document_question = st.chat_input("Ask a question about the document")
        if document_question:
            payload = {"question": document_question}
            response = requests.post("http://localhost:8000/document-query/", params=payload)
            if response.status_code == 200:
                data = response.json()
                st.write(f"**Answer:** {data['answer']}")
            else:
                st.error(f"Error querying document: {response.text}")
    
    # Chat input for general questions
    general_question = st.chat_input("Ask a general question")
    if general_question:
        payload = {"question": general_question}
        response = requests.post("http://localhost:8000/general-query/", params=payload)

        if response.status_code == 200:
            data = response.json()
            st.write(f"**Answer:** {data['answer']}")
        else:
            st.error(f"Error with general query: {response.text}")

    #Clear Button
    if st.button('Clear Chat History and Index'):
        response = requests.get("http://localhost:8000/clear-index/")
        if response.status_code == 200:
            data = response.json()
            st.success(data['message'])
        else:
            st.error(f"Error clearing index: {response.text}")
    
    

if __name__ == "__main__":
    import requests
    # Run the Streamlit app
    #main()

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=PORT)

