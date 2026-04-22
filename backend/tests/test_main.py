import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException


class TestGetDocumentAnswer:
    def test_no_document_loaded_raises_400(self):
        import main
        main.global_index = None
        with pytest.raises(HTTPException) as exc_info:
            main.get_document_answer("What is this?")
        assert exc_info.value.status_code == 400
        assert "No document" in exc_info.value.detail

    def test_valid_question_returns_answer(self):
        import main
        main.global_index = MagicMock()
        with patch("main.handle_document_query", return_value="The answer"):
            result = main.get_document_answer("What is this about?")
        assert result == "The answer"
        main.global_index = None

    def test_value_error_raises_422(self):
        import main
        main.global_index = MagicMock()
        with patch("main.handle_document_query", side_effect=ValueError("bad input")):
            with pytest.raises(HTTPException) as exc_info:
                main.get_document_answer("What is this?")
        assert exc_info.value.status_code == 422
        main.global_index = None

    def test_runtime_error_raises_500(self):
        import main
        main.global_index = MagicMock()
        with patch("main.handle_document_query", side_effect=RuntimeError("LLM failed")):
            with pytest.raises(HTTPException) as exc_info:
                main.get_document_answer("What is this?")
        assert exc_info.value.status_code == 500
        main.global_index = None


class TestGetGeneralAnswer:
    def test_valid_question_returns_answer(self):
        import main
        with patch("main.handle_general_query", return_value="General answer"):
            result = main.get_general_answer("What is Python?")
        assert result == "General answer"

    def test_value_error_raises_422(self):
        import main
        with patch("main.handle_general_query", side_effect=ValueError("bad input")):
            with pytest.raises(HTTPException) as exc_info:
                main.get_general_answer("What is Python?")
        assert exc_info.value.status_code == 422

    def test_runtime_error_raises_500(self):
        import main
        with patch("main.handle_general_query", side_effect=RuntimeError("LLM failed")):
            with pytest.raises(HTTPException) as exc_info:
                main.get_general_answer("What is Python?")
        assert exc_info.value.status_code == 500
