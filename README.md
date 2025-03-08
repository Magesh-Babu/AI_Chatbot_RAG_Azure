# AI-Chatbot-Application

This repository contains an AI-powered chatbot application built using **Streamlit** and **FastAPI**. The chatbot supports both general queries and document-based queries, leveraging state-of-the-art **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** techniques to provide accurate and context-aware responses.

### 🚀 Deployment
The application is **containerized using Docker** and deployed on **Azure**. It follows a **microservices architecture** with a separate backend and frontend:
- **Backend:** Built using **FastAPI**, exposing RESTful endpoints.
- **Frontend:** Developed with **Streamlit** for an interactive user interface.
- **CI/CD:** Integrated **GitHub Actions** for automated builds, testing, and deployment.

### 🌐 Try the App
Check out the deployed app on **Azure**:  
[**Live Demo**](https://ai-chatbot-rag-magesh-babu.streamlit.app/)  

## 🔥 Features

1. **General Query Support:** The chatbot can handle a wide range of questions based on general knowledge.
2. **Document-Based Query Support:** Users can upload documents, and the chatbot retrieves and analyzes data from the uploaded files.
3. **Microservices Architecture:**
   - **FastAPI Backend**: Handles API requests and processes queries.
   - **Streamlit Frontend**: Provides an interactive UI.
4. **RAG Implementation:** Uses LlamaIndex for efficient data retrieval and augmentation.
5. **LLM Integration:** Powered by the Llama 3 (8B) model deployed on Azure.
6. **Embeddings:** Utilizes a Hugging Face embedding model for semantic search.
7. **CI/CD Pipeline:** Uses GitHub Actions to automate build, testing, and deployment.

## 🛠️ Technology Stack

| Technology   | Purpose |
|-------------|---------|
| **Docker**  | Containerization for deployment |
| **FastAPI** | Backend API service |
| **Streamlit** | Interactive web application frontend |
| **Llama 3 (8B)** | Large Language Model for chatbot responses |
| **Azure** | Cloud platform for hosting |
| **LlamaIndex** | Efficient document retrieval & processing |
| **Hugging Face** | Embedding models for semantic search |
| **GitHub Actions** | CI/CD automation |

---
