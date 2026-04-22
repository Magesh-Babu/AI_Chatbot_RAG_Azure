from llama_index.core.llms import ChatMessage

def handle_general_query(prompt, llm):
    """
    Handles general-purpose queries using the Azure AI model.

    Args:
        prompt (str): The general input question.
        llm (AzureAICompletionsModel): Azure AI completions client.

    Returns:
        str: LLM response text.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if llm is None:
        raise ValueError("LLM instance is required.")

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content=prompt.strip()),
    ]
    try:
        assistant_response = llm.chat(messages)
        return assistant_response.message.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed for general query: {e}") from e


def handle_document_query(index, prompt, llm):
    """
    Handles document-based queries using the Azure AI model.

    Args:
        index (VectorStoreIndex): Vectorized form of the input document.
        prompt (str): The question about the document.
        llm (AzureAICompletionsModel): Azure AI completions client.

    Returns:
        str: LLM response text grounded in the document.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if index is None:
        raise ValueError("Index is required for document query.")
    if llm is None:
        raise ValueError("LLM instance is required.")

    try:
        chat_engine = index.as_chat_engine(
            chat_mode="context", verbose=True, llm=llm, streaming=True
        )
        response_stream = chat_engine.stream_chat(prompt.strip())
        return "".join(response_stream.response_gen)
    except Exception as e:
        raise RuntimeError(f"LLM call failed for document query: {e}") from e
