import streamlit as st
from llama_index.core.llms import ChatMessage
import time

def handle_general_query(prompt, llm):
    """
    Handles general-purpose queries using the Azure AI model.
    
    Args:
        prompt (str): The general input question.
        llm (AzureAICompletionsModel): An instance of a class or client that provides interaction with the llama 3 model in Azure.
        
    Returns:
        str: LLM generates response based on questions in the prompt.
    """
    # Add user prompt to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display the response
    with st.chat_message("assistant"):
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=prompt),
        ]
        assistant_response = llm.chat(messages)

        def stream_data():
            for word in assistant_response.message.content.split(" "):
                yield word + " "
                time.sleep(0.02)

        st.session_state.messages.append({"role": "assistant", "content": st.write_stream(stream_data)})
    
    
def handle_document_query(index, prompt, llm):
    """
    Handles document based queries using the Azure AI model.
    
    Args:
        index (float): vectorized form of input document
        prompt (str): The relevant question about the document.
        llm (AzureAICompletionsModel): An instance of a class or client that provides interaction with the llama 3 model in Azure.
        
    Returns:
        str: LLM generates response based on questions with given document in the prompt.
    """
    
    # Create and store the document-specific chat engine in session state
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True, llm=llm, streaming=True
    )
   
    # Add user prompt to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Generate assistant response
    with st.chat_message("assistant"):
        if "chat_engine" in st.session_state:
            # Use document-specific chat engine
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)  # Stream the response
            response_content = response_stream.response
            
    # Append assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": response_content})

