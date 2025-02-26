import streamlit as st
import requests
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
    if response_status.json()["status"]:
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
    main()
