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

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content=prompt),
    ]
    assistant_response = llm.chat(messages)
    return assistant_response.message.content
    
    
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

    chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True, llm=llm, streaming=True
    )
    response_stream = chat_engine.stream_chat(prompt)

    full_response=""
    for token in response_stream.response_gen:
        full_response+=token
    return full_response
