import pytest
from unittest.mock import MagicMock, patch
import chromadb


class TestConnectChromadbCreateIndex:
    def test_empty_list_raises_value_error(self):
        from chat import connect_chromadb_create_index
        with pytest.raises(ValueError, match="document list is empty"):
            connect_chromadb_create_index([])

    def test_none_raises_value_error(self):
        from chat import connect_chromadb_create_index
        with pytest.raises(ValueError, match="document list is empty"):
            connect_chromadb_create_index(None)

    def test_chroma_error_raises_runtime_error(self):
        from chat import connect_chromadb_create_index
        with patch("chat.chroma_client") as mock_client, \
             patch("chat.initialize_embed_model"):
            mock_client.get_or_create_collection.side_effect = chromadb.errors.ChromaError("DB error")
            with pytest.raises(RuntimeError, match="ChromaDB error while creating index"):
                connect_chromadb_create_index([MagicMock()])

    def test_unexpected_error_raises_runtime_error(self):
        from chat import connect_chromadb_create_index
        with patch("chat.chroma_client") as mock_client, \
             patch("chat.initialize_embed_model"):
            mock_client.get_or_create_collection.side_effect = Exception("Unexpected failure")
            with pytest.raises(RuntimeError, match="Unexpected error creating index"):
                connect_chromadb_create_index([MagicMock()])

    def test_returns_index_on_success(self):
        from chat import connect_chromadb_create_index
        mock_index = MagicMock()
        with patch("chat.chroma_client") as mock_client, \
             patch("chat.initialize_embed_model"), \
             patch("chat.Settings"), \
             patch("chat.ChromaVectorStore"), \
             patch("chat.StorageContext"), \
             patch("chat.VectorStoreIndex") as mock_vector_index:
            mock_vector_index.from_documents.return_value = mock_index
            result = connect_chromadb_create_index([MagicMock()])
        assert result == mock_index


class TestClearChromadbDb:
    def test_success_calls_delete_collection(self):
        from chat import clear_chromadb_db
        with patch("chat.chroma_client") as mock_client:
            clear_chromadb_db()
            mock_client.delete_collection.assert_called_once_with("given_doc")

    def test_chroma_error_raises_runtime_error(self):
        from chat import clear_chromadb_db
        with patch("chat.chroma_client") as mock_client:
            mock_client.delete_collection.side_effect = chromadb.errors.ChromaError("Delete failed")
            with pytest.raises(RuntimeError, match="ChromaDB error while clearing collection"):
                clear_chromadb_db()

    def test_unexpected_error_raises_runtime_error(self):
        from chat import clear_chromadb_db
        with patch("chat.chroma_client") as mock_client:
            mock_client.delete_collection.side_effect = Exception("Unexpected failure")
            with pytest.raises(RuntimeError, match="Unexpected error clearing ChromaDB"):
                clear_chromadb_db()
