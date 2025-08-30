from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI (including lesson link if available)
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
                # Get lesson link from vector store
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    source += f"||{lesson_link}"  # Embed link invisibly with separator
            sources.append(source)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving the full outline of a course from metadata"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Render-ready source strings (may contain <a>)
        self.last_links = []    # List of (label, url) tuples

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": (
                "Return a course outline from the course metadata collection. "
                "Use this for outline-related queries (e.g., 'show the outline for ...')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title to fetch the outline for (partial names allowed)"
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """Fetch and format the outline (title, link, and all lessons)."""
        import json

        # Resolve the course title using the catalog's semantic search
        try:
            resolved_title = self.store._resolve_course_name(course_title) if course_title else None
        except Exception:
            resolved_title = None

        if not resolved_title:
            return f"No course found matching '{course_title}'."

        try:
            data = self.store.course_catalog.get(ids=[resolved_title])
        except Exception as e:
            return f"Error retrieving course outline: {e}"

        if not data or not data.get("metadatas"):
            return f"No metadata found for course '{resolved_title}'."

        meta = data["metadatas"][0]
        course_link = meta.get("course_link")

        # Parse lessons
        lessons = []
        lessons_json = meta.get("lessons_json")
        if lessons_json:
            try:
                lessons = json.loads(lessons_json)
            except Exception:
                lessons = []

        # Prepare sources/links for UI
        label = resolved_title
        if course_link:
            self.last_sources = [f'<a href="{course_link}" target="_blank" rel="noopener">{label}</a>']
            self.last_links = [(label, course_link)]
        else:
            self.last_sources = [label]
            self.last_links = []

        # Format output per requirements
        lines = [
            f"Course Title: {resolved_title}",
            f"Course Link: {course_link if course_link else 'N/A'}",
            "Lessons:",
        ]
        # Ensure lessons are sorted by lesson_number if possible
        try:
            lessons_sorted = sorted(
                lessons,
                key=lambda x: (x.get("lesson_number") is None, x.get("lesson_number"))
            )
        except Exception:
            lessons_sorted = lessons

        for lesson in lessons_sorted:
            num = lesson.get("lesson_number")
            title = lesson.get("lesson_title") or lesson.get("title") or "Untitled Lesson"
            if num is not None:
                lines.append(f"{num}. {title}")
            else:
                lines.append(f"- {title}")

        if len(lines) == 3:  # No lessons
            lines.append("(No lessons found)")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def get_last_links(self) -> list:
        """Get (label, url) tuples from the last search operation"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_links') and getattr(tool, 'last_links'):
                return tool.last_links
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
            if hasattr(tool, 'last_links'):
                tool.last_links = []