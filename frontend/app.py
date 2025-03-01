import streamlit as st
import requests
import os
import tempfile

def display_chat():
    """Displays chat messages stored in session state."""
    if "messages" not in st.session_state or not st.session_state.messages:
        # Initialize default general chat messages
        st.session_state.messages = [{"role": "assistant", "content": "Hello, How can i help you?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])       

def clear_chat_history():
    """Clears the chat history and resets the session state."""
    st.session_state.messages = []
    if "uploaded_file_path" in st.session_state:
        try:
            os.remove(st.session_state.uploaded_file_path)
        except FileNotFoundError:
            pass
        del st.session_state.uploaded_file_path


def main():
    # Get the FastAPI base URL from an environment variable
    FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")  # Default to localhost if not set

    st.set_page_config(page_title="AI Chatbot with RAG", page_icon="🔥")

    # Sidebar for file upload and settings
    with st.sidebar:
        st.title("AI Chatbot with RAG 🔥 🚀")
        st.markdown("#### Upload a document and start asking questions about it.")
        st.markdown("#### Or Ask general questions without uploading any document.")
        # File uploader for document (supports PDF and text files)
        uploaded_document = st.file_uploader("Upload Document (PDF or Text)", type=["pdf", "txt"])

        if uploaded_document is not None:
            try:
                files = {"file": (uploaded_document.name, uploaded_document.getvalue())}
                response = requests.post(f"{FASTAPI_BASE_URL}/upload-document/", files=files)
                response.raise_for_status()  # Raises an exception for bad status codes
                st.success(response.json()['message'])
            except requests.exceptions.RequestException as e:
                st.error(f"Error uploading or processing document: {e}")

        if st.sidebar.button('Clear Chat History'):
            try:
                response = requests.get(f"{FASTAPI_BASE_URL}/clear-index/")
                response.raise_for_status()
                data = response.json()
                st.success(data['message'])
                clear_chat_history()
            except requests.exceptions.RequestException as e:
                st.error(f"Error clearing index or chat history: {e}")


    # Main app logic
    try:
        status_response = requests.get(f"{FASTAPI_BASE_URL}/status/")
        status_response.raise_for_status()
        status_data = status_response.json()
        is_document_uploaded = status_data['status']

        display_chat()

        if prompt := st.chat_input("Ask a question"):
            # Append the user's message to session_state.messages
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)  # Display the user's message in the chat

            if is_document_uploaded:
                response = requests.post(f"{FASTAPI_BASE_URL}/document-query/", params={"question": prompt})
                if response.status_code == 200:
                    data = response.json()
                    with st.chat_message("assistant"):
                        st.write(data['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": data['answer']})
                else:
                    st.error(f"Error querying document: {response.text}")
            else:
                response = requests.post(f"{FASTAPI_BASE_URL}/general-query/", params={"question": prompt})
                if response.status_code == 200:
                    data = response.json()
                    with st.chat_message("assistant"):
                        st.write(data['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": data['answer']})
                else:
                    st.error(f"Error with general query: {response.text}")
    except requests.exceptions.RequestException as e:
                st.error(f"Error with general status check: {e}")

if __name__ == "__main__":
    main()
