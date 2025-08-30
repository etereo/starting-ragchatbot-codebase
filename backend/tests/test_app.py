import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile
import sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Course, Lesson


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_system = Mock()
    
    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_session_manager.clear_session.return_value = None
    mock_system.session_manager = mock_session_manager
    
    # Mock query method
    mock_system.query.return_value = (
        "This is a test AI response about machine learning.",
        ["https://example.com/course1/lesson1", "https://example.com/course2/lesson2"]
    )
    
    # Mock analytics
    mock_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Machine Learning", "Advanced AI"]
    }
    
    # Mock add_course_folder
    mock_system.add_course_folder.return_value = (2, 15)
    
    return mock_system


@pytest.fixture
def mock_static_files():
    """Mock static files directory to avoid mounting errors"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple index.html for testing
        index_path = os.path.join(temp_dir, "index.html")
        with open(index_path, "w") as f:
            f.write("<html><body>Test Frontend</body></html>")
        yield temp_dir


@pytest.fixture
def test_client(mock_rag_system, mock_static_files):
    """Create test client with mocked dependencies"""
    with patch('app.RAGSystem') as mock_rag_class, \
         patch('app.StaticFiles') as mock_static_files_class:
        
        mock_rag_class.return_value = mock_rag_system
        
        # Mock StaticFiles to avoid directory issues
        mock_static_files_instance = Mock()
        mock_static_files_class.return_value = mock_static_files_instance
        
        # Import app after patching
        from app import app
        
        # Override the static files mount for testing
        app.router.routes = [route for route in app.router.routes if not hasattr(route, 'name') or route.name != 'static']
        
        client = TestClient(app)
        yield client


class TestQueryEndpoint:
    """Test /api/query endpoint"""
    
    def test_query_with_new_session(self, test_client, mock_rag_system):
        """Test query endpoint creates new session when none provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test AI response about machine learning."
        assert len(data["sources"]) == 2
        assert data["session_id"] == "test-session-123"
        
        # Verify RAG system was called correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is machine learning?", "test-session-123")
    
    def test_query_with_existing_session(self, test_client, mock_rag_system):
        """Test query endpoint uses existing session when provided"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "Tell me about neural networks",
                "session_id": "existing-session-456"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing-session-456"
        
        # Verify session manager was not called to create new session
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me about neural networks", "existing-session-456")
    
    def test_query_with_empty_query(self, test_client):
        """Test query endpoint with empty query"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200  # Should still work, RAG system handles empty queries
    
    def test_query_with_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_query_with_missing_query_field(self, test_client):
        """Test query endpoint with missing query field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test-session"}
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_query_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]


class TestCoursesEndpoint:
    """Test /api/courses endpoint"""
    
    def test_get_courses_success(self, test_client, mock_rag_system):
        """Test courses endpoint returns analytics successfully"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Machine Learning" in data["course_titles"]
        assert "Advanced AI" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_courses_rag_system_error(self, test_client, mock_rag_system):
        """Test courses endpoint when RAG system raises exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]


class TestSessionEndpoints:
    """Test session management endpoints"""
    
    def test_new_session(self, test_client, mock_rag_system):
        """Test new session endpoint"""
        response = test_client.post("/api/session/new")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_new_session_error(self, test_client, mock_rag_system):
        """Test new session endpoint when session manager raises exception"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation error")
        
        response = test_client.post("/api/session/new")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Session creation error" in data["detail"]
    
    def test_clear_session(self, test_client, mock_rag_system):
        """Test clear session endpoint"""
        response = test_client.post(
            "/api/session/clear",
            json={"session_id": "test-session-456"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ok" in data
        assert data["ok"] is True
        
        mock_rag_system.session_manager.clear_session.assert_called_once_with("test-session-456")
    
    def test_clear_session_missing_session_id(self, test_client):
        """Test clear session endpoint with missing session_id"""
        response = test_client.post(
            "/api/session/clear",
            json={}
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_clear_session_error(self, test_client, mock_rag_system):
        """Test clear session endpoint when session manager raises exception"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session clear error")
        
        response = test_client.post(
            "/api/session/clear",
            json={"session_id": "test-session-456"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Session clear error" in data["detail"]


class TestStartupEvent:
    """Test startup event handler"""
    
    def test_startup_with_docs_folder(self, mock_rag_system):
        """Test startup event when docs folder exists"""
        with patch('os.path.exists') as mock_exists, \
             patch('app.RAGSystem') as mock_rag_class:
            
            mock_exists.return_value = True
            mock_rag_class.return_value = mock_rag_system
            
            # Import and trigger startup
            from app import startup_event
            import asyncio
            
            asyncio.run(startup_event())
            
            mock_rag_system.add_course_folder.assert_called_once_with("../docs", clear_existing=False)
    
    def test_startup_without_docs_folder(self, mock_rag_system):
        """Test startup event when docs folder doesn't exist"""
        with patch('os.path.exists') as mock_exists, \
             patch('app.RAGSystem') as mock_rag_class:
            
            mock_exists.return_value = False
            mock_rag_class.return_value = mock_rag_system
            
            # Import and trigger startup
            from app import startup_event
            import asyncio
            
            asyncio.run(startup_event())
            
            mock_rag_system.add_course_folder.assert_not_called()
    
    def test_startup_with_loading_error(self, mock_rag_system):
        """Test startup event when document loading raises exception"""
        with patch('os.path.exists') as mock_exists, \
             patch('app.RAGSystem') as mock_rag_class, \
             patch('builtins.print') as mock_print:
            
            mock_exists.return_value = True
            mock_rag_class.return_value = mock_rag_system
            mock_rag_system.add_course_folder.side_effect = Exception("Loading error")
            
            # Import and trigger startup
            from app import startup_event
            import asyncio
            
            asyncio.run(startup_event())
            
            # Should print error message
            mock_print.assert_any_call("Error loading documents: Loading error")


class TestMiddleware:
    """Test middleware configuration"""
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set"""
        response = test_client.options("/api/query")
        
        # FastAPI should handle OPTIONS requests for CORS
        assert response.status_code == 405 or response.status_code == 200  # Method not allowed or OK
    
    def test_trusted_host_middleware(self, test_client):
        """Test trusted host middleware allows requests"""
        response = test_client.get("/api/courses")
        
        # Should not block requests (middleware allows all hosts)
        assert response.status_code != 400  # Not a bad request due to host issues


class TestResponseModels:
    """Test response models and validation"""
    
    def test_query_response_structure(self, test_client):
        """Test query response has correct structure"""
        response = test_client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        required_fields = {"answer", "sources", "session_id"}
        assert set(data.keys()) == required_fields
        
        # Verify field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
    
    def test_courses_response_structure(self, test_client):
        """Test courses response has correct structure"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        required_fields = {"total_courses", "course_titles"}
        assert set(data.keys()) == required_fields
        
        # Verify field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
    
    def test_session_new_response_structure(self, test_client):
        """Test new session response has correct structure"""
        response = test_client.post("/api/session/new")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = {"session_id"}
        assert set(data.keys()) == required_fields
        
        # Verify field types
        assert isinstance(data["session_id"], str)
    
    def test_session_clear_response_structure(self, test_client):
        """Test clear session response has correct structure"""
        response = test_client.post(
            "/api/session/clear",
            json={"session_id": "test-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = {"ok"}
        assert set(data.keys()) == required_fields
        
        # Verify field types
        assert isinstance(data["ok"], bool)
        assert data["ok"] is True