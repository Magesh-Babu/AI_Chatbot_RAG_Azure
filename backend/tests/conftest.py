import os
import sys
from unittest.mock import MagicMock, patch

# Must be set before any backend module is imported — chat.py validates these at import time
os.environ.setdefault("AZURE_META_API", "test-meta-api-key")
os.environ.setdefault("AZURE_META_ENDPOINT", "https://test-meta-endpoint.azure.com")
os.environ.setdefault("AZURE_COHERE_API", "test-cohere-api-key")
os.environ.setdefault("AZURE_COHERE_ENDPOINT", "https://test-cohere-endpoint.azure.com")

# Add the backend directory to sys.path so test files can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Pre-import main with mocked LLM and LLMRerank to prevent Azure HTTP calls at module level.
# - initialize_llm is mocked so main.llm is a MagicMock (no real Azure client calls)
# - LLMRerank is mocked so pydantic never validates the mock LLM against the LLM base class,
#   and no llm.metadata access happens during init
_mock_llm = MagicMock()
with patch("chat.initialize_llm", return_value=_mock_llm), \
     patch("llama_index.core.postprocessor.LLMRerank"):
    import main  # noqa: E402
