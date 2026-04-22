import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import main
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global index and document name before and after every test."""
    import main
    main.global_index = None
    main.global_document_name = None
    yield
    main.global_index = None
    main.global_document_name = None


class TestUploadDocument:
    def test_valid_txt_file_returns_200(self, client):
        with patch("main.create_index_from_document"):
            response = client.post(
                "/upload-document/",
                files={"file": ("test.txt", b"This is a test document.", "text/plain")}
            )
        assert response.status_code == 200
        assert "uploaded and processed successfully" in response.json()["message"]

    def test_valid_pdf_file_returns_200(self, client):
        with patch("main.create_index_from_document"):
            response = client.post(
                "/upload-document/",
                files={"file": ("report.pdf", b"%PDF-1.4 fake content", "application/pdf")}
            )
        assert response.status_code == 200

    def test_unsupported_file_type_returns_400(self, client):
        response = client.post(
            "/upload-document/",
            files={"file": ("script.exe", b"binary content", "application/octet-stream")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_oversized_file_returns_413(self, client):
        large_content = b"x" * (5 * 1024 * 1024 + 1)  # 1 byte over 5MB
        response = client.post(
            "/upload-document/",
            files={"file": ("big.txt", large_content, "text/plain")}
        )
        assert response.status_code == 413
        assert "5MB limit" in response.json()["detail"]

    def test_file_at_exact_size_limit_is_accepted(self, client):
        exact_content = b"x" * (5 * 1024 * 1024)  # exactly 5MB
        with patch("main.create_index_from_document"):
            response = client.post(
                "/upload-document/",
                files={"file": ("exact.txt", exact_content, "text/plain")}
            )
        assert response.status_code == 200


class TestDocumentQuery:
    def test_no_document_loaded_returns_400(self, client):
        response = client.post("/document-query/", params={"question": "What is this?"})
        assert response.status_code == 400
        assert "No document" in response.json()["detail"]

    def test_blank_question_returns_422(self, client):
        response = client.post("/document-query/", params={"question": "   "})
        assert response.status_code == 422

    def test_missing_question_returns_422(self, client):
        response = client.post("/document-query/")
        assert response.status_code == 422

    def test_valid_question_with_document_returns_answer(self, client):
        import main
        main.global_index = MagicMock()
        with patch("main.handle_document_query", return_value="The answer"):
            response = client.post("/document-query/", params={"question": "What is this about?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "The answer"


class TestGeneralQuery:
    def test_valid_question_returns_answer(self, client):
        with patch("main.handle_general_query", return_value="General answer"):
            response = client.post("/general-query/", params={"question": "What is Python?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "General answer"

    def test_blank_question_returns_422(self, client):
        response = client.post("/general-query/", params={"question": "   "})
        assert response.status_code == 422

    def test_missing_question_returns_422(self, client):
        response = client.post("/general-query/")
        assert response.status_code == 422


class TestStatus:
    def test_no_document_returns_false_status(self, client):
        response = client.get("/status/")
        assert response.status_code == 200
        assert response.json()["status"] is False
        assert response.json()["message"] == "No Document uploaded."

    def test_document_loaded_returns_true_status(self, client):
        import main
        main.global_document_name = "report.pdf"
        response = client.get("/status/")
        assert response.status_code == 200
        assert response.json()["status"] is True
        assert "report.pdf" in response.json()["message"]


class TestClearIndex:
    def test_clear_index_returns_200(self, client):
        with patch("main.clear_chromadb_db"):
            response = client.get("/clear-index/")
        assert response.status_code == 200
        assert "cleared" in response.json()["message"]

    def test_clear_index_resets_global_state(self, client):
        import main
        main.global_index = MagicMock()
        main.global_document_name = "test.pdf"
        with patch("main.clear_chromadb_db"):
            client.get("/clear-index/")
        assert main.global_index is None
        assert main.global_document_name is None

    def test_clear_index_chroma_error_returns_500(self, client):
        with patch("main.clear_chromadb_db", side_effect=RuntimeError("DB error")):
            response = client.get("/clear-index/")
        assert response.status_code == 500
