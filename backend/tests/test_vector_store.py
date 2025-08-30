import json
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Test cases for SearchResults class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course": "A"}, {"course": "B"}]],
            "distances": [[0.8, 0.9]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"course": "A"}, {"course": "B"}]
        assert results.distances == [0.8, 0.9]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [], "metadatas": [], "distances": []}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Database error")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Database error"

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.8])
        assert results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore class"""

    @patch("chromadb.PersistentClient")
    @patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction")
    def test_init_creates_collections(self, mock_embedding_fn, mock_chroma_client):
        """Test VectorStore initialization creates required collections"""
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore("/test/path", "test-model", max_results=10)

        assert store.max_results == 10
        assert mock_client.get_or_create_collection.call_count == 2

        # Verify collections were created
        calls = mock_client.get_or_create_collection.call_args_list
        collection_names = [call[1]["name"] for call in calls]
        assert "course_catalog" in collection_names
        assert "course_content" in collection_names

    def test_search_successful_query(self, mock_chroma_client):
        """Test successful search query"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            # Setup mock collection response
            mock_chroma_client.collections["course_content"].query.return_value = {
                "documents": [["Test content about ML"]],
                "metadatas": [[{"course_title": "ML Course", "lesson_number": 1}]],
                "distances": [[0.8]],
            }

            results = store.search("machine learning")

            assert not results.error
            assert len(results.documents) == 1
            assert "Test content about ML" in results.documents[0]
            assert results.metadata[0]["course_title"] == "ML Course"
            assert results.distances[0] == 0.8

    def test_search_with_course_name_resolution(self, mock_chroma_client):
        """Test search with course name that needs resolution"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            # Setup course catalog to resolve course name
            mock_chroma_client.collections["course_catalog"].query.return_value = {
                "documents": [["Introduction to Machine Learning"]],
                "metadatas": [[{"title": "Introduction to Machine Learning"}]],
                "distances": [[0.9]],
            }

            # Setup course content search
            mock_chroma_client.collections["course_content"].query.return_value = {
                "documents": [["Content from resolved course"]],
                "metadatas": [
                    [
                        {
                            "course_title": "Introduction to Machine Learning",
                            "lesson_number": 1,
                        }
                    ]
                ],
                "distances": [[0.85]],
            }

            results = store.search("neural networks", course_name="ML")

            assert not results.error
            assert len(results.documents) == 1

            # Verify course catalog was queried for resolution
            catalog_call = mock_chroma_client.collections[
                "course_catalog"
            ].query.call_args
            assert catalog_call[1]["query_texts"] == ["ML"]

            # Verify content search used resolved course name
            content_call = mock_chroma_client.collections[
                "course_content"
            ].query.call_args
            assert content_call[1]["where"] == {
                "course_title": "Introduction to Machine Learning"
            }

    def test_search_course_not_found(self, mock_chroma_client):
        """Test search with course name that cannot be resolved"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            # Setup course catalog to return empty results
            mock_chroma_client.collections["course_catalog"].query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            results = store.search("test query", course_name="Nonexistent Course")

            assert results.error == "No course found matching 'Nonexistent Course'"
            assert results.is_empty()

    def test_search_with_lesson_number_filter(self, mock_chroma_client):
        """Test search with lesson number filter"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_content"].query.return_value = {
                "documents": [["Lesson 3 content"]],
                "metadatas": [[{"course_title": "Test Course", "lesson_number": 3}]],
                "distances": [[0.9]],
            }

            results = store.search("test query", lesson_number=3)

            # Verify filter was applied
            content_call = mock_chroma_client.collections[
                "course_content"
            ].query.call_args
            assert content_call[1]["where"] == {"lesson_number": 3}

    def test_search_with_both_filters(self, mock_chroma_client):
        """Test search with both course name and lesson number filters"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            # Setup course resolution
            mock_chroma_client.collections["course_catalog"].query.return_value = {
                "documents": [["Test Course"]],
                "metadatas": [[{"title": "Test Course"}]],
                "distances": [[0.95]],
            }

            mock_chroma_client.collections["course_content"].query.return_value = {
                "documents": [["Specific lesson content"]],
                "metadatas": [[{"course_title": "Test Course", "lesson_number": 2}]],
                "distances": [[0.92]],
            }

            results = store.search(
                "test query", course_name="Test Course", lesson_number=2
            )

            # Verify combined filter was applied
            content_call = mock_chroma_client.collections[
                "course_content"
            ].query.call_args
            expected_filter = {
                "$and": [{"course_title": "Test Course"}, {"lesson_number": 2}]
            }
            assert content_call[1]["where"] == expected_filter

    def test_search_with_custom_limit(self, mock_chroma_client):
        """Test search with custom result limit"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model", max_results=5)

            mock_chroma_client.collections["course_content"].query.return_value = {
                "documents": [["Test doc"]],
                "metadatas": [[{"course_title": "Test"}]],
                "distances": [[0.8]],
            }

            results = store.search("test query", limit=10)

            # Verify custom limit was used
            content_call = mock_chroma_client.collections[
                "course_content"
            ].query.call_args
            assert content_call[1]["n_results"] == 10

    def test_search_chromadb_exception(self, mock_chroma_client):
        """Test search handling of ChromaDB exceptions"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            # Configure collection to raise exception
            mock_chroma_client.collections["course_content"].query.side_effect = (
                Exception("DB Error")
            )

            results = store.search("test query")

            assert results.error == "Search error: DB Error"
            assert results.is_empty()

    def test_build_filter_no_params(self, mock_chroma_client):
        """Test filter building with no parameters"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            filter_dict = store._build_filter(None, None)
            assert filter_dict is None

    def test_build_filter_course_only(self, mock_chroma_client):
        """Test filter building with course title only"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            filter_dict = store._build_filter("Test Course", None)
            assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, mock_chroma_client):
        """Test filter building with lesson number only"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            filter_dict = store._build_filter(None, 5)
            assert filter_dict == {"lesson_number": 5}

    def test_build_filter_both_params(self, mock_chroma_client):
        """Test filter building with both parameters"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            filter_dict = store._build_filter("Test Course", 3)
            expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
            assert filter_dict == expected

    def test_resolve_course_name_success(self, mock_chroma_client):
        """Test successful course name resolution"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].query.return_value = {
                "documents": [["Machine Learning Course"]],
                "metadatas": [[{"title": "Machine Learning Course"}]],
                "distances": [[0.9]],
            }

            resolved = store._resolve_course_name("ML Course")
            assert resolved == "Machine Learning Course"

    def test_resolve_course_name_not_found(self, mock_chroma_client):
        """Test course name resolution when not found"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            resolved = store._resolve_course_name("Nonexistent")
            assert resolved is None

    def test_resolve_course_name_exception(self, mock_chroma_client):
        """Test course name resolution with exception"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].query.side_effect = (
                Exception("DB Error")
            )

            resolved = store._resolve_course_name("Test")
            assert resolved is None

    def test_add_course_metadata(self, mock_chroma_client, sample_course):
        """Test adding course metadata to catalog"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            store.add_course_metadata(sample_course)

            # Verify catalog collection was called
            catalog_call = mock_chroma_client.collections[
                "course_catalog"
            ].add.call_args
            assert catalog_call[1]["documents"] == [sample_course.title]
            assert catalog_call[1]["ids"] == [sample_course.title]

            # Verify metadata structure
            metadata = catalog_call[1]["metadatas"][0]
            assert metadata["title"] == sample_course.title
            assert metadata["instructor"] == sample_course.instructor
            assert metadata["course_link"] == sample_course.course_link
            assert metadata["lesson_count"] == len(sample_course.lessons)

            # Verify lessons JSON
            lessons_json = json.loads(metadata["lessons_json"])
            assert len(lessons_json) == 3
            assert lessons_json[0]["lesson_number"] == 0
            assert lessons_json[0]["lesson_title"] == "Introduction"

    def test_add_course_content(self, mock_chroma_client, sample_course_chunks):
        """Test adding course content chunks"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            store.add_course_content(sample_course_chunks)

            # Verify content collection was called
            content_call = mock_chroma_client.collections[
                "course_content"
            ].add.call_args

            # Verify documents
            expected_docs = [chunk.content for chunk in sample_course_chunks]
            assert content_call[1]["documents"] == expected_docs

            # Verify metadata
            expected_metadata = [
                {
                    "course_title": chunk.course_title,
                    "lesson_number": chunk.lesson_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in sample_course_chunks
            ]
            assert content_call[1]["metadatas"] == expected_metadata

            # Verify IDs format
            expected_ids = [
                f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}"
                for chunk in sample_course_chunks
            ]
            assert content_call[1]["ids"] == expected_ids

    def test_add_course_content_empty(self, mock_chroma_client):
        """Test adding empty course content list"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            store.add_course_content([])

            # Verify content collection was not called
            mock_chroma_client.collections["course_content"].add.assert_not_called()

    def test_clear_all_data(self, mock_chroma_client):
        """Test clearing all data from collections"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            store.clear_all_data()

            # Verify collections were deleted
            assert mock_chroma_client.delete_collection.call_count == 2
            mock_chroma_client.delete_collection.assert_any_call("course_catalog")
            mock_chroma_client.delete_collection.assert_any_call("course_content")

            # Verify collections were recreated
            assert (
                mock_chroma_client.get_or_create_collection.call_count >= 4
            )  # 2 initial + 2 recreated

    def test_get_existing_course_titles(self, mock_chroma_client):
        """Test getting existing course titles"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "ids": ["Course A", "Course B", "Course C"]
            }

            titles = store.get_existing_course_titles()
            assert titles == ["Course A", "Course B", "Course C"]

    def test_get_existing_course_titles_empty(self, mock_chroma_client):
        """Test getting course titles when none exist"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.return_value = {}

            titles = store.get_existing_course_titles()
            assert titles == []

    def test_get_existing_course_titles_exception(self, mock_chroma_client):
        """Test getting course titles with exception"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.side_effect = (
                Exception("DB Error")
            )

            titles = store.get_existing_course_titles()
            assert titles == []

    def test_get_course_count(self, mock_chroma_client):
        """Test getting course count"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "ids": ["Course A", "Course B"]
            }

            count = store.get_course_count()
            assert count == 2

    def test_get_course_count_empty(self, mock_chroma_client):
        """Test getting course count when no courses exist"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.return_value = {}

            count = store.get_course_count()
            assert count == 0

    def test_get_course_link_success(self, mock_chroma_client):
        """Test getting course link successfully"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "metadatas": [{"course_link": "https://example.com/course"}]
            }

            link = store.get_course_link("Test Course")
            assert link == "https://example.com/course"

    def test_get_course_link_not_found(self, mock_chroma_client):
        """Test getting course link when course not found"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "metadatas": []
            }

            link = store.get_course_link("Nonexistent Course")
            assert link is None

    def test_get_lesson_link_success(self, mock_chroma_client):
        """Test getting lesson link successfully"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            lessons_json = json.dumps(
                [
                    {
                        "lesson_number": 1,
                        "lesson_title": "Intro",
                        "lesson_link": "https://example.com/lesson1",
                    },
                    {
                        "lesson_number": 2,
                        "lesson_title": "Advanced",
                        "lesson_link": "https://example.com/lesson2",
                    },
                ]
            )

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "metadatas": [{"lessons_json": lessons_json}]
            }

            link = store.get_lesson_link("Test Course", 2)
            assert link == "https://example.com/lesson2"

    def test_get_lesson_link_not_found(self, mock_chroma_client):
        """Test getting lesson link when lesson not found"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            lessons_json = json.dumps(
                [
                    {
                        "lesson_number": 1,
                        "lesson_title": "Intro",
                        "lesson_link": "https://example.com/lesson1",
                    }
                ]
            )

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "metadatas": [{"lessons_json": lessons_json}]
            }

            link = store.get_lesson_link("Test Course", 5)  # Lesson 5 doesn't exist
            assert link is None

    def test_get_all_courses_metadata(self, mock_chroma_client):
        """Test getting all courses metadata with lessons parsing"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            lessons_json = json.dumps(
                [{"lesson_number": 1, "lesson_title": "Introduction"}]
            )

            mock_chroma_client.collections["course_catalog"].get.return_value = {
                "metadatas": [
                    {
                        "title": "Test Course",
                        "instructor": "Test Instructor",
                        "lessons_json": lessons_json,
                    }
                ]
            }

            metadata = store.get_all_courses_metadata()

            assert len(metadata) == 1
            assert metadata[0]["title"] == "Test Course"
            assert metadata[0]["instructor"] == "Test Instructor"
            assert "lessons" in metadata[0]
            assert "lessons_json" not in metadata[0]  # Should be removed
            assert metadata[0]["lessons"][0]["lesson_number"] == 1


class TestVectorStoreIntegration:
    """Integration tests with more realistic scenarios"""

    def test_complete_search_workflow(self, mock_chroma_client):
        """Test complete search workflow from query to results"""
        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            store = VectorStore("/test", "test-model")

            # Setup course resolution
            mock_chroma_client.collections["course_catalog"].query.return_value = {
                "documents": [["Introduction to Machine Learning"]],
                "metadatas": [[{"title": "Introduction to Machine Learning"}]],
                "distances": [[0.95]],
            }

            # Setup content search
            mock_chroma_client.collections["course_content"].query.return_value = {
                "documents": [
                    "Linear regression is a fundamental algorithm in machine learning.",
                    "Neural networks are inspired by biological neurons.",
                ],
                "metadatas": [
                    {
                        "course_title": "Introduction to Machine Learning",
                        "lesson_number": 1,
                        "chunk_index": 5,
                    },
                    {
                        "course_title": "Introduction to Machine Learning",
                        "lesson_number": 3,
                        "chunk_index": 12,
                    },
                ],
                "distances": [[0.88, 0.82]],
            }

            results = store.search(
                query="machine learning algorithms", course_name="ML Intro"
            )

            # Verify successful results
            assert not results.error
            assert len(results.documents) == 2
            assert "Linear regression" in results.documents[0]
            assert "Neural networks" in results.documents[1]

            # Verify metadata preservation
            assert results.metadata[0]["lesson_number"] == 1
            assert results.metadata[1]["lesson_number"] == 3
            assert results.metadata[0]["chunk_index"] == 5
            assert results.metadata[1]["chunk_index"] == 12
