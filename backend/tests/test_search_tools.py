from unittest.mock import Mock, patch

import pytest
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool.execute() method"""

    def test_execute_successful_search(self, mock_vector_store):
        """Test successful search with results"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return successful results
        mock_vector_store.search.return_value = SearchResults(
            documents=["This is machine learning content about linear regression."],
            metadata=[
                {"course_title": "Introduction to Machine Learning", "lesson_number": 1}
            ],
            distances=[0.85],
        )

        # Execute
        result = tool.execute(query="What is linear regression?")

        # Verify
        assert isinstance(result, str)
        assert "Introduction to Machine Learning" in result
        assert "Lesson 1" in result
        assert "machine learning content" in result
        assert len(tool.last_sources) == 1
        assert len(tool.last_links) == 1

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="What is linear regression?", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test search with course name filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Content about neural networks"],
            metadata=[{"course_title": "Advanced ML Course", "lesson_number": 3}],
            distances=[0.90],
        )

        result = tool.execute(query="neural networks", course_name="Advanced ML")

        assert "Advanced ML Course" in result
        assert "neural networks" in result
        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name="Advanced ML", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test search with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson 2 specific content"],
            metadata=[{"course_title": "ML Basics", "lesson_number": 2}],
            distances=[0.75],
        )

        result = tool.execute(query="test query", lesson_number=2)

        assert "ML Basics" in result
        assert "Lesson 2" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=2
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test search with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.95],
        )

        result = tool.execute(
            query="test query", course_name="Test Course", lesson_number=1
        )

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Test Course", lesson_number=1
        )

    def test_execute_empty_results(self, mock_vector_store):
        """Test search that returns no results"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return empty results
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0
        assert len(tool.last_links) == 0

    def test_execute_empty_results_with_course_filter(self, mock_vector_store):
        """Test empty results with course filter shows course name"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool.execute(query="nonexistent topic", course_name="Some Course")

        assert "No relevant content found" in result
        assert "in course 'Some Course'" in result

    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store):
        """Test empty results with lesson filter shows lesson number"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool.execute(query="nonexistent topic", lesson_number=5)

        assert "No relevant content found" in result
        assert "in lesson 5" in result

    def test_execute_error_from_vector_store(self, mock_vector_store):
        """Test handling of error from vector store"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return error
        mock_vector_store.search.return_value = SearchResults.empty(
            "Database connection failed"
        )

        result = tool.execute(query="test query")

        assert result == "Database connection failed"
        assert len(tool.last_sources) == 0
        assert len(tool.last_links) == 0

    def test_execute_multiple_results(self, mock_vector_store):
        """Test search with multiple results"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=[
                "First result about ML algorithms",
                "Second result about deep learning",
            ],
            metadata=[
                {"course_title": "ML Basics", "lesson_number": 1},
                {"course_title": "Advanced ML", "lesson_number": 2},
            ],
            distances=[0.90, 0.85],
        )

        result = tool.execute(query="machine learning")

        assert "ML Basics" in result
        assert "Advanced ML" in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result
        assert "First result about ML" in result
        assert "Second result about deep" in result
        assert len(tool.last_sources) == 2

    def test_execute_result_with_links(self, mock_vector_store):
        """Test that links are properly generated in sources"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.90],
        )

        # Mock the link methods
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        result = tool.execute(query="test")

        # Check that lesson link was attempted first
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)

        # Check sources contain HTML link
        assert len(tool.last_sources) == 1
        assert '<a href="https://example.com/lesson1"' in tool.last_sources[0]
        assert "Test Course - Lesson 1" in tool.last_sources[0]

        # Check links tuple
        assert len(tool.last_links) == 1
        assert tool.last_links[0] == (
            "Test Course - Lesson 1",
            "https://example.com/lesson1",
        )

    def test_execute_result_fallback_to_course_link(self, mock_vector_store):
        """Test fallback to course link when lesson link unavailable"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.90],
        )

        # Mock lesson link to return None, course link to return URL
        mock_vector_store.get_lesson_link.return_value = None
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        result = tool.execute(query="test")

        # Check both methods were called
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)
        mock_vector_store.get_course_link.assert_called_once_with("Test Course")

        # Check sources use course link
        assert '<a href="https://example.com/course"' in tool.last_sources[0]

    def test_execute_result_no_links(self, mock_vector_store):
        """Test result formatting when no links are available"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.90],
        )

        # Mock both link methods to return None
        mock_vector_store.get_lesson_link.return_value = None
        mock_vector_store.get_course_link.return_value = None

        result = tool.execute(query="test")

        # Check sources contain plain text (no HTML links)
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0] == "Test Course - Lesson 1"
        assert "<a href=" not in tool.last_sources[0]
        assert len(tool.last_links) == 0

    def test_execute_result_missing_metadata(self, mock_vector_store):
        """Test handling of results with missing metadata"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{}],  # Empty metadata
            distances=[0.90],
        )

        result = tool.execute(query="test")

        assert "[unknown]" in result
        assert "Test content" in result

    def test_execute_vector_store_exception(self, mock_vector_store):
        """Test handling of exceptions from vector store"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to raise exception
        mock_vector_store.search.side_effect = Exception("ChromaDB error")

        result = tool.execute(query="test")

        # The search method should handle exceptions by returning SearchResults.empty()
        # If not handled, the exception would propagate
        assert isinstance(result, str)  # Should not raise exception

    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition is properly formed"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool.execute() method"""

    def test_execute_successful_outline(self, mock_vector_store):
        """Test successful outline retrieval"""
        tool = CourseOutlineTool(mock_vector_store)

        # Mock course catalog response
        mock_vector_store.course_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Test Course",
                    "instructor": "Test Instructor",
                    "course_link": "https://example.com/course",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Advanced Topics"}]',
                }
            ]
        }
        mock_vector_store._resolve_course_name.return_value = "Test Course"

        result = tool.execute(course_title="Test Course")

        assert "Course Title: Test Course" in result
        assert "Course Link: https://example.com/course" in result
        assert "1. Introduction" in result
        assert "2. Advanced Topics" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test outline when course is not found"""
        tool = CourseOutlineTool(mock_vector_store)

        mock_vector_store._resolve_course_name.return_value = None

        result = tool.execute(course_title="Nonexistent Course")

        assert "No course found matching 'Nonexistent Course'" in result

    def test_execute_no_metadata(self, mock_vector_store):
        """Test outline when no metadata is available"""
        tool = CourseOutlineTool(mock_vector_store)

        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.return_value = None

        result = tool.execute(course_title="Test Course")

        assert "No metadata found for course 'Test Course'" in result

    def test_get_tool_definition(self, mock_vector_store):
        """Test outline tool definition"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_title" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_title"]


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [defn["name"] for defn in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool_success(self, mock_vector_store):
        """Test successful tool execution through manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test result"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.8],
        )

        result = manager.execute_tool("search_course_content", query="test")

        assert isinstance(result, str)
        assert "Test result" in result

    def test_execute_tool_not_found(self, mock_vector_store):
        """Test executing non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test retrieving sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.8],
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert "Test Course - Lesson 1" in sources[0]

    def test_get_last_links(self, mock_vector_store):
        """Test retrieving links from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate links
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.8],
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        manager.execute_tool("search_course_content", query="test")
        links = manager.get_last_links()

        assert len(links) == 1
        assert links[0] == ("Test Course - Lesson 1", "https://example.com/lesson")

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources and links"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Populate sources first
        tool.last_sources = ["Test Source"]
        tool.last_links = [("Test", "https://example.com")]

        manager.reset_sources()

        assert len(tool.last_sources) == 0
        assert len(tool.last_links) == 0


class TestSearchToolsIntegration:
    """Integration tests for search tools with real behavior"""

    def test_search_tool_stores_sources_correctly(self, mock_vector_store):
        """Test that search tool properly stores sources for UI"""
        tool = CourseSearchTool(mock_vector_store)

        # Setup complex metadata scenario
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},  # No lesson number
            ],
            distances=[0.9, 0.8],
        )

        # Mock link resolution
        mock_vector_store.get_lesson_link.side_effect = lambda course, lesson: (
            "https://example.com/lesson" if lesson else None
        )
        mock_vector_store.get_course_link.side_effect = (
            lambda course: "https://example.com/course"
        )

        result = tool.execute(query="test")

        # Verify sources were stored
        assert len(tool.last_sources) == 2
        assert len(tool.last_links) == 2

        # First source should have lesson link
        assert '<a href="https://example.com/lesson"' in tool.last_sources[0]
        assert "Course A - Lesson 1" in tool.last_sources[0]

        # Second source should have course link (no lesson)
        assert '<a href="https://example.com/course"' in tool.last_sources[1]
        assert "Course B" in tool.last_sources[1]
