import time
from llama_index.core.llms import ChatMessage
from logging_config import get_logger

logger = get_logger(__name__)


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

    logger.info("Handling general query. prompt_length=%d", len(prompt.strip()))
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content=prompt.strip()),
    ]
    try:
        start = time.monotonic()
        assistant_response = llm.chat(messages)
        duration_ms = round((time.monotonic() - start) * 1000)
        logger.info("General query completed. duration_ms=%d", duration_ms)
        return assistant_response.message.content
    except Exception as e:
        logger.error("LLM call failed for general query: %s", e, exc_info=True)
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

    logger.info("Handling document query. prompt_length=%d", len(prompt.strip()))
    try:
        chat_engine = index.as_chat_engine(
            chat_mode="context", verbose=True, llm=llm, streaming=True
        )
        start = time.monotonic()
        response_stream = chat_engine.stream_chat(prompt.strip())
        result = "".join(response_stream.response_gen)
        duration_ms = round((time.monotonic() - start) * 1000)
        logger.info("Document query completed. duration_ms=%d", duration_ms)
        return result
    except Exception as e:
        logger.error("LLM call failed for document query: %s", e, exc_info=True)
        raise RuntimeError(f"LLM call failed for document query: {e}") from e
