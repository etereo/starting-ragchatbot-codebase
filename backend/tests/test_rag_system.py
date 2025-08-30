import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class TestRAGSystemInitialization:
    """Test RAG system initialization"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore') 
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_init_components(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test that all components are properly initialized"""
        rag = RAGSystem(mock_config)
        
        # Verify all components were initialized
        mock_doc_proc.assert_called_once_with(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
        mock_vector_store.assert_called_once_with(
            mock_config.CHROMA_PATH,
            mock_config.EMBEDDING_MODEL,
            mock_config.MAX_RESULTS
        )
        mock_ai_gen.assert_called_once_with(mock_config)
        mock_session_mgr.assert_called_once_with(mock_config.MAX_HISTORY)
    
    def test_tool_registration(self, mock_config):
        """Test that search tools are properly registered"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag = RAGSystem(mock_config)
            
            # Verify tools are registered
            tool_names = list(rag.tool_manager.tools.keys())
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
            
            # Verify tools have access to vector store
            search_tool = rag.tool_manager.tools["search_course_content"]
            assert search_tool.store == rag.vector_store


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config, sample_course, sample_course_chunks):
        """Test successful course document addition"""
        # Setup mocks
        mock_processor = mock_doc_proc.return_value
        mock_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        mock_store = mock_vector_store.return_value
        
        rag = RAGSystem(mock_config)
        
        # Test adding document
        course, chunk_count = rag.add_course_document("/test/course.txt")
        
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)
        
        # Verify processing and storage
        mock_processor.process_course_document.assert_called_once_with("/test/course.txt")
        mock_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_store.add_course_content.assert_called_once_with(sample_course_chunks)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_processing_error(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test handling of document processing errors"""
        # Setup mock to raise exception
        mock_processor = mock_doc_proc.return_value
        mock_processor.process_course_document.side_effect = Exception("File processing error")
        
        rag = RAGSystem(mock_config)
        
        # Test error handling
        course, chunk_count = rag.add_course_document("/test/bad_file.txt")
        
        assert course is None
        assert chunk_count == 0
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config, sample_course, sample_course_chunks):
        """Test successful folder processing"""
        # Setup filesystem mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "readme.md"]
        
        # Setup processing mocks
        mock_processor = mock_doc_proc.return_value
        mock_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        mock_store = mock_vector_store.return_value
        mock_store.get_existing_course_titles.return_value = []  # No existing courses
        
        rag = RAGSystem(mock_config)
        
        with patch('os.path.isfile', return_value=True):
            courses, chunks = rag.add_course_folder("/test/docs")
        
        assert courses == 2  # Only PDF and TXT files processed
        assert chunks == len(sample_course_chunks) * 2
        
        # Verify processing calls
        assert mock_processor.process_course_document.call_count == 2
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    def test_add_course_folder_not_found(self, mock_exists, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test folder processing when folder doesn't exist"""
        mock_exists.return_value = False
        
        rag = RAGSystem(mock_config)
        courses, chunks = rag.add_course_folder("/nonexistent/path")
        
        assert courses == 0
        assert chunks == 0


class TestRAGSystemQuery:
    """Test RAG system query processing"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_anthropic_successful(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test successful query processing with Anthropic"""
        mock_config.LLM_PROVIDER = "anthropic"
        
        # Setup AI generator mock
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.return_value = "Here's what I found about machine learning..."
        
        # Setup tool manager mock
        mock_tool_mgr = Mock()
        mock_tool_mgr.get_last_sources.return_value = ["ML Course - Lesson 1"]
        mock_tool_mgr.get_last_links.return_value = [("ML Course", "https://example.com/ml")]
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_mgr
        
        response, sources = rag.query("What is machine learning?")
        
        assert "Here's what I found about machine learning" in response
        assert sources == ["ML Course - Lesson 1"]
        
        # Verify AI generator was called correctly
        mock_generator.generate_response.assert_called_once()
        call_args = mock_generator.generate_response.call_args
        assert "What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["tools"] == mock_tool_mgr.get_tool_definitions()
        assert call_args[1]["tool_manager"] == mock_tool_mgr
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_openai_with_manual_context(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query processing with OpenAI using manual context injection"""
        mock_config.LLM_PROVIDER = "openai"
        
        # Setup AI generator mock
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.return_value = "OpenAI response based on context"
        
        # Setup tool manager mock
        mock_tool_mgr = Mock()
        mock_tool_mgr.execute_tool.return_value = "Context: Machine learning is a subset of AI"
        mock_tool_mgr.get_last_sources.return_value = ["ML Course"]
        mock_tool_mgr.get_last_links.return_value = [("ML Course", "https://example.com/ml")]
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_mgr
        
        response, sources = rag.query("Explain machine learning")
        
        assert response == "OpenAI response based on context"
        assert sources == ["ML Course"]
        
        # Verify manual tool execution
        mock_tool_mgr.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Explain machine learning"
        )
        
        # Verify augmented query was sent to AI
        call_args = mock_generator.generate_response.call_args
        assert "Context:" in call_args[1]["query"]
        assert "Machine learning is a subset of AI" in call_args[1]["query"]
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_session_history(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query processing with session history"""
        # Setup session manager mock
        mock_session = mock_session_mgr.return_value
        mock_session.get_conversation_history.return_value = "Previous conversation context"
        
        # Setup AI generator mock
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.return_value = "Response with context"
        
        rag = RAGSystem(mock_config)
        
        response, sources = rag.query("Follow up question", session_id="session123")
        
        # Verify history was retrieved and used
        mock_session.get_conversation_history.assert_called_once_with("session123")
        call_args = mock_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation context"
        
        # Verify exchange was recorded
        mock_session.add_exchange.assert_called_once_with(
            "session123",
            "Follow up question",
            response
        )
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_ai_generator_exception(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query handling when AI generator raises exception"""
        # Setup AI generator to raise exception
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.side_effect = Exception("API Error")
        
        rag = RAGSystem(mock_config)
        
        # Should propagate exception (not caught by RAG system)
        with pytest.raises(Exception, match="API Error"):
            rag.query("Test question")
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_source_links_in_response(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test that source links are added to response"""
        # Setup AI generator mock
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.return_value = "AI response"
        
        # Setup tool manager with links
        mock_tool_mgr = Mock()
        mock_tool_mgr.get_last_sources.return_value = ["Test Source"]
        mock_tool_mgr.get_last_links.return_value = [("Test Course", "https://example.com/course")]
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_mgr
        
        response, sources = rag.query("Test question")
        
        # Verify link was added to response
        assert "AI response" in response
        assert "[Open source: Test Course](https://example.com/course)" in response
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_sources_reset_after_retrieval(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test that sources are reset after each query"""
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.return_value = "Response"
        
        mock_tool_mgr = Mock()
        mock_tool_mgr.get_last_sources.return_value = ["Source"]
        mock_tool_mgr.get_last_links.return_value = []
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_mgr
        
        rag.query("Test question")
        
        # Verify sources were reset
        mock_tool_mgr.reset_sources.assert_called_once()


class TestRAGSystemAnalytics:
    """Test RAG system analytics functionality"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test course analytics retrieval"""
        mock_store = mock_vector_store.return_value
        mock_store.get_course_count.return_value = 5
        mock_store.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C", "Course D", "Course E"]
        
        rag = RAGSystem(mock_config)
        analytics = rag.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]


class TestRAGSystemIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_complete_query_flow_with_mocks(self, mock_config):
        """Test complete query flow from start to finish"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:
            
            # Setup realistic mock responses
            mock_store = mock_vector_store.return_value
            mock_generator = mock_ai_gen.return_value
            mock_session = mock_session_mgr.return_value
            
            # Configure for Anthropic with tool calling
            mock_config.LLM_PROVIDER = "anthropic"
            mock_generator.generate_response.return_value = "Machine learning is a field of artificial intelligence that enables computers to learn without being explicitly programmed."
            
            # Setup tool manager behavior
            rag = RAGSystem(mock_config)
            
            # Create realistic tool manager mock
            mock_tool_mgr = Mock()
            mock_tool_mgr.get_tool_definitions.return_value = [
                {"name": "search_course_content", "description": "Search courses"}
            ]
            mock_tool_mgr.get_last_sources.return_value = ["<a href='https://example.com'>ML Course - Lesson 1</a>"]
            mock_tool_mgr.get_last_links.return_value = [("ML Course - Lesson 1", "https://example.com")]
            rag.tool_manager = mock_tool_mgr
            
            # Execute query
            response, sources = rag.query("What is machine learning?", session_id="test_session")
            
            # Verify complete flow
            assert "Machine learning is a field" in response
            assert "[Open source: ML Course - Lesson 1](https://example.com)" in response
            assert len(sources) == 1
            
            # Verify all components were called
            mock_generator.generate_response.assert_called_once()
            mock_tool_mgr.get_last_sources.assert_called_once()
            mock_tool_mgr.reset_sources.assert_called_once()
    
    def test_error_propagation_through_system(self, mock_config):
        """Test how errors propagate through the system"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager'):
            
            # Setup AI generator to raise specific error
            mock_generator = mock_ai_gen.return_value
            mock_generator.generate_response.side_effect = ValueError("Invalid API key")
            
            rag = RAGSystem(mock_config)
            
            # Error should propagate up
            with pytest.raises(ValueError, match="Invalid API key"):
                rag.query("Test question")
    
    def test_different_provider_paths(self, mock_config):
        """Test that different providers follow different code paths"""
        test_cases = ["anthropic", "openai", "gemini"]
        
        for provider in test_cases:
            with patch('rag_system.DocumentProcessor'), \
                 patch('rag_system.VectorStore'), \
                 patch('rag_system.AIGenerator') as mock_ai_gen, \
                 patch('rag_system.SessionManager'):
                
                mock_config.LLM_PROVIDER = provider
                mock_generator = mock_ai_gen.return_value
                mock_generator.generate_response.return_value = f"{provider} response"
                
                rag = RAGSystem(mock_config)
                mock_tool_mgr = Mock()
                mock_tool_mgr.execute_tool.return_value = "Context content"
                mock_tool_mgr.get_last_sources.return_value = []
                mock_tool_mgr.get_last_links.return_value = []
                rag.tool_manager = mock_tool_mgr
                
                response, _ = rag.query("Test question")
                
                assert f"{provider} response" in response
                
                # Verify appropriate call pattern
                if provider == "anthropic":
                    # Should use tool calling
                    call_args = mock_generator.generate_response.call_args
                    assert call_args[1]["tools"] is not None
                    assert call_args[1]["tool_manager"] is not None
                else:
                    # Should use manual context injection
                    mock_tool_mgr.execute_tool.assert_called()


class TestRAGSystemRealDocumentScenarios:
    """Test with real document processing scenarios"""
    
    def test_document_processing_integration(self, mock_config, tmp_path):
        """Test document processing with actual file system"""
        # Create test document
        test_doc = tmp_path / "test_course.txt"
        test_doc.write_text(
            "Course Title: Test ML Course\n"
            "Course Link: https://example.com/course\n"
            "Course Instructor: Dr. Test\n\n"
            "Lesson 0: Introduction\n"
            "This is the introduction to machine learning.\n\n"
            "Lesson 1: Algorithms\n"
            "Linear regression is a fundamental algorithm."
        )
        
        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            mock_store = mock_vector_store.return_value
            
            rag = RAGSystem(mock_config)
            
            # Test adding the document
            course, chunk_count = rag.add_course_document(str(test_doc))
            
            # Verify course was processed
            assert course is not None
            assert course.title == "Test ML Course"
            assert course.instructor == "Dr. Test"
            assert len(course.lessons) == 2
            assert chunk_count > 0
            
            # Verify vector store was called
            mock_store.add_course_metadata.assert_called_once()
            mock_store.add_course_content.assert_called_once()
    
    def test_folder_processing_with_mixed_files(self, mock_config, tmp_path):
        """Test folder processing with various file types"""
        # Create test files
        (tmp_path / "course1.txt").write_text("Course Title: Course 1\nContent here")
        (tmp_path / "course2.pdf").write_text("Course Title: Course 2\nPDF content")  # Mock PDF
        (tmp_path / "readme.md").write_text("This is a readme")
        (tmp_path / "data.json").write_text('{"not": "a course"}')
        
        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            mock_store = mock_vector_store.return_value
            mock_store.get_existing_course_titles.return_value = []
            
            rag = RAGSystem(mock_config)
            
            courses, chunks = rag.add_course_folder(str(tmp_path))
            
            # Should process 2 course files (txt and pdf), ignore others
            assert courses == 2
            assert chunks > 0
            
            # Verify calls
            assert mock_store.add_course_metadata.call_count == 2
            assert mock_store.add_course_content.call_count == 2


class TestRAGSystemErrorScenarios:
    """Test various error scenarios"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tool_execution_failure(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query when tool execution fails"""
        mock_config.LLM_PROVIDER = "openai"  # Uses manual tool execution
        
        # Setup tool manager to return error
        mock_tool_mgr = Mock()
        mock_tool_mgr.execute_tool.return_value = "Search error: Database connection failed"
        mock_tool_mgr.get_last_sources.return_value = []
        mock_tool_mgr.get_last_links.return_value = []
        
        # Setup AI generator
        mock_generator = mock_ai_gen.return_value
        mock_generator.generate_response.return_value = "I encountered an error searching the course materials."
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_mgr
        
        response, sources = rag.query("What is machine learning?")
        
        # Should still return a response even with tool error
        assert "I encountered an error" in response
        assert len(sources) == 0
        
        # Verify error context was passed to AI
        call_args = mock_generator.generate_response.call_args
        assert "Search error: Database connection failed" in call_args[1]["query"]