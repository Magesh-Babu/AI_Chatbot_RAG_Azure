import pytest
from unittest.mock import MagicMock
from query_type import handle_general_query, handle_document_query


class TestHandleGeneralQuery:
    def test_valid_prompt_returns_response(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value.message.content = "Hello!"
        result = handle_general_query("What is Python?", mock_llm)
        assert result == "Hello!"

    def test_empty_prompt_raises_value_error(self):
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            handle_general_query("", mock_llm)

    def test_whitespace_only_prompt_raises_value_error(self):
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            handle_general_query("   ", mock_llm)

    def test_none_prompt_raises_value_error(self):
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            handle_general_query(None, mock_llm)

    def test_none_llm_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM instance is required"):
            handle_general_query("What is Python?", None)

    def test_llm_exception_raises_runtime_error(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API connection error")
        with pytest.raises(RuntimeError, match="LLM call failed for general query"):
            handle_general_query("What is Python?", mock_llm)

    def test_prompt_is_stripped_before_sending(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value.message.content = "Answer"
        handle_general_query("  What is Python?  ", mock_llm)
        call_messages = mock_llm.chat.call_args[0][0]
        assert call_messages[1].content == "What is Python?"


class TestHandleDocumentQuery:
    def test_valid_params_returns_joined_response(self):
        mock_index = MagicMock()
        mock_llm = MagicMock()
        mock_engine = MagicMock()
        mock_index.as_chat_engine.return_value = mock_engine
        mock_engine.stream_chat.return_value.response_gen = iter(["Hello", " World"])
        result = handle_document_query(mock_index, "What is this about?", mock_llm)
        assert result == "Hello World"

    def test_empty_prompt_raises_value_error(self):
        mock_index = MagicMock()
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            handle_document_query(mock_index, "", mock_llm)

    def test_whitespace_only_prompt_raises_value_error(self):
        mock_index = MagicMock()
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            handle_document_query(mock_index, "   ", mock_llm)

    def test_none_index_raises_value_error(self):
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Index is required"):
            handle_document_query(None, "What is this about?", mock_llm)

    def test_none_llm_raises_value_error(self):
        mock_index = MagicMock()
        with pytest.raises(ValueError, match="LLM instance is required"):
            handle_document_query(mock_index, "What is this about?", None)

    def test_engine_exception_raises_runtime_error(self):
        mock_index = MagicMock()
        mock_llm = MagicMock()
        mock_index.as_chat_engine.side_effect = Exception("Engine init failed")
        with pytest.raises(RuntimeError, match="LLM call failed for document query"):
            handle_document_query(mock_index, "What is this about?", mock_llm)

    def test_multiple_tokens_are_joined_correctly(self):
        mock_index = MagicMock()
        mock_llm = MagicMock()
        mock_engine = MagicMock()
        mock_index.as_chat_engine.return_value = mock_engine
        mock_engine.stream_chat.return_value.response_gen = iter(["token1", " token2", " token3"])
        result = handle_document_query(mock_index, "Question?", mock_llm)
        assert result == "token1 token2 token3"

    def test_prompt_is_stripped_before_sending(self):
        mock_index = MagicMock()
        mock_llm = MagicMock()
        mock_engine = MagicMock()
        mock_index.as_chat_engine.return_value = mock_engine
        mock_engine.stream_chat.return_value.response_gen = iter(["Answer"])
        handle_document_query(mock_index, "  Question?  ", mock_llm)
        mock_engine.stream_chat.assert_called_once_with("Question?")
