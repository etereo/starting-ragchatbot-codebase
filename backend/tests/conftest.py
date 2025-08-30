import os
import shutil

# Import project modules
import sys
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock(spec=Config)
    config.LLM_PROVIDER = "anthropic"
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.OPENAI_API_KEY = "test-openai-key"
    config.OPENAI_MODEL = "gpt-4o-mini"
    config.GEMINI_API_KEY = "test-gemini-key"
    config.GEMINI_MODEL = "gemini-1.5-flash"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://example.com/ml-course/lesson0",
            ),
            Lesson(
                lesson_number=1,
                title="Linear Regression",
                lesson_link="https://example.com/ml-course/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Neural Networks",
                lesson_link="https://example.com/ml-course/lesson2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Lesson 0 content: This is the introduction to the machine learning course.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 1 content: Linear regression is a fundamental technique.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 2 content: Neural networks are powerful models.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()

    # Setup default search behavior
    mock_store.search.return_value = SearchResults(
        documents=["Test content about machine learning"],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1}
        ],
        distances=[0.8],
    )

    mock_store._resolve_course_name.return_value = "Introduction to Machine Learning"
    mock_store.get_lesson_link.return_value = "https://example.com/ml-course/lesson1"
    mock_store.get_course_link.return_value = "https://example.com/ml-course"

    return mock_store


@pytest.fixture
def mock_empty_search_results():
    """Mock empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_error_search_results():
    """Mock search results with error"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response for testing"""
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test AI response"
    mock_response.stop_reason = "end_turn"
    return mock_response


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic API response with tool use"""
    mock_response = Mock()

    # Mock tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "test query"}
    tool_block.id = "tool_123"

    # Mock text content block
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Using search tool..."

    mock_response.content = [text_block, tool_block]
    mock_response.stop_reason = "tool_use"
    return mock_response


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        }
    ]
    mock_manager.execute_tool.return_value = "Test search result"
    mock_manager.get_last_sources.return_value = ["Test Source"]
    mock_manager.get_last_links.return_value = [("Test Course", "https://example.com")]
    mock_manager.reset_sources.return_value = None
    return mock_manager


@pytest.fixture
def temp_chroma_db():
    """Temporary ChromaDB directory for integration tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection"""
    mock_collection = Mock()

    # Default query response
    mock_collection.query.return_value = {
        "documents": [["Test document content"]],
        "metadatas": [
            [{"course_title": "Introduction to Machine Learning", "lesson_number": 1}]
        ],
        "distances": [[0.8]],
    }

    # Default get response
    mock_collection.get.return_value = {
        "ids": ["Introduction to Machine Learning"],
        "metadatas": [
            {
                "title": "Introduction to Machine Learning",
                "instructor": "Dr. Jane Smith",
                "course_link": "https://example.com/ml-course",
                "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson1"}]',
            }
        ],
    }

    return mock_collection


class MockChromaClient:
    """Mock ChromaDB client for testing"""

    def __init__(self, path=None, settings=None):
        self.path = path
        self.settings = settings
        self.collections = {}

    def get_or_create_collection(self, name, embedding_function=None):
        if name not in self.collections:
            self.collections[name] = mock_chroma_collection()
        return self.collections[name]

    def delete_collection(self, name):
        if name in self.collections:
            del self.collections[name]


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client fixture"""
    return MockChromaClient()


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing"""
    mock_transformer = Mock()
    mock_transformer.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return mock_transformer


def create_test_search_results(
    documents: List[str] = None,
    metadata: List[Dict] = None,
    distances: List[float] = None,
    error: str = None,
) -> SearchResults:
    """Utility function to create test search results"""
    return SearchResults(
        documents=documents or [],
        metadata=metadata or [],
        distances=distances or [],
        error=error,
    )


def create_test_course_content() -> str:
    """Create test course content in expected format"""
    return """Course Title: Test Course
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/test-course/lesson0

This is the introduction lesson content.

Lesson 1: Advanced Topics
Lesson Link: https://example.com/test-course/lesson1

This is the advanced topics lesson content.
"""


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI app for testing without static file issues"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add same middleware as main app
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    return app


@pytest.fixture
def mock_session_manager():
    """Mock session manager for API tests"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test-session-123"
    mock_manager.clear_session.return_value = None
    mock_manager.get_history.return_value = []
    mock_manager.add_message.return_value = None
    return mock_manager


@pytest.fixture
def api_test_data():
    """Common test data for API tests"""
    return {
        "sample_query": "What is machine learning?",
        "sample_answer": "Machine learning is a subset of artificial intelligence.",
        "sample_sources": [
            "https://example.com/course1/lesson1",
            "https://example.com/course2/lesson2"
        ],
        "sample_session_id": "test-session-456",
        "sample_analytics": {
            "total_courses": 3,
            "course_titles": [
                "Introduction to Machine Learning",
                "Advanced Deep Learning",
                "Natural Language Processing"
            ]
        }
    }


@pytest.fixture(autouse=True)
def prevent_actual_api_calls():
    """Prevent actual API calls during testing"""
    with (
        patch("anthropic.Anthropic") as mock_anthropic,
        patch("openai.OpenAI") as mock_openai,
        patch("google.generativeai.configure") as mock_gemini_config,
        patch("google.generativeai.GenerativeModel") as mock_gemini_model,
        patch("chromadb.PersistentClient") as mock_chroma_client,
        patch("sentence_transformers.SentenceTransformer") as mock_transformer,
    ):
        # Setup mock responses
        mock_anthropic.return_value.messages.create.return_value = Mock()
        mock_openai.return_value.chat.completions.create.return_value = Mock()
        mock_gemini_model.return_value.generate_content.return_value = Mock()
        mock_chroma_client.return_value = MockChromaClient()
        mock_transformer.return_value.encode.return_value = [[0.1, 0.2, 0.3]]

        yield {
            "anthropic": mock_anthropic,
            "openai": mock_openai,
            "gemini_config": mock_gemini_config,
            "gemini_model": mock_gemini_model,
            "chroma_client": mock_chroma_client,
            "sentence_transformer": mock_transformer,
        }
