from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator class"""

    def test_init_anthropic(self, mock_config):
        """Test initialization with Anthropic provider"""
        mock_config.LLM_PROVIDER = "anthropic"

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_anthropic = Mock()
            mock_anthropic_class.return_value = mock_anthropic

            generator = AIGenerator(mock_config)

            assert generator.provider == "anthropic"
            assert generator._anthropic == mock_anthropic
            mock_anthropic_class.assert_called_once_with(
                api_key=mock_config.ANTHROPIC_API_KEY
            )

    def test_init_openai(self, mock_config):
        """Test initialization with OpenAI provider"""
        mock_config.LLM_PROVIDER = "openai"

        with patch("openai.OpenAI") as mock_openai_class:
            mock_openai = Mock()
            mock_openai_class.return_value = mock_openai

            generator = AIGenerator(mock_config)

            assert generator.provider == "openai"
            assert generator._openai == mock_openai
            mock_openai_class.assert_called_once_with(
                api_key=mock_config.OPENAI_API_KEY
            )

    def test_init_gemini(self, mock_config):
        """Test initialization with Gemini provider"""
        mock_config.LLM_PROVIDER = "gemini"

        with patch("google.generativeai") as mock_genai:
            generator = AIGenerator(mock_config)

            assert generator.provider == "gemini"
            mock_genai.configure.assert_called_once_with(
                api_key=mock_config.GEMINI_API_KEY
            )

    def test_init_unsupported_provider(self, mock_config):
        """Test initialization with unsupported provider raises error"""
        mock_config.LLM_PROVIDER = "unsupported"

        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            AIGenerator(mock_config)

    def test_generate_response_anthropic_without_tools(
        self, mock_config, prevent_actual_api_calls
    ):
        """Test Anthropic response generation without tools"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.return_value = mock_response

        generator = AIGenerator(mock_config)
        result = generator.generate_response("Test query")

        assert result == "Test response"

        # Verify API call parameters
        call_args = prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "Test query"
        assert "tools" not in call_args[1]

    def test_generate_response_anthropic_with_tools_no_tool_use(
        self, mock_config, mock_tool_manager, prevent_actual_api_calls
    ):
        """Test Anthropic response with tools but no tool use"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Setup mock response without tool use
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response without tools"
        mock_response.stop_reason = "end_turn"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.return_value = mock_response

        generator = AIGenerator(mock_config)
        tools = mock_tool_manager.get_tool_definitions()

        result = generator.generate_response(
            query="Test query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Direct response without tools"

        # Verify tools were passed
        call_args = prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == {"type": "auto"}

    def test_generate_response_anthropic_with_single_round_tool_use(
        self, mock_config, mock_tool_manager, prevent_actual_api_calls
    ):
        """Test Anthropic response with single round tool use (no additional rounds needed)"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Setup response with tool use
        tool_use_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test search"}
        tool_block.id = "tool_123"

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "I'll search for that information."

        tool_use_response.content = [text_block, tool_block]
        tool_use_response.stop_reason = "tool_use"

        # Setup final response after tool execution (no more tool use)
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = (
            "Based on the search results, here's the answer..."
        )
        final_response.stop_reason = "end_turn"

        # Configure mock to return different responses on subsequent calls
        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = [tool_use_response, final_response]

        # Configure tool manager
        mock_tool_manager.execute_tool.return_value = (
            "Search results: machine learning content"
        )

        generator = AIGenerator(mock_config)
        tools = mock_tool_manager.get_tool_definitions()

        result = generator.generate_response(
            query="Tell me about machine learning",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Based on the search results, here's the answer..."

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test search"
        )

        # Verify two API calls were made (round 1 + round 2)
        assert (
            prevent_actual_api_calls[
                "anthropic"
            ].return_value.messages.create.call_count
            == 2
        )

    def test_generate_response_anthropic_tool_execution_error(
        self, mock_config, mock_tool_manager, prevent_actual_api_calls
    ):
        """Test handling of tool execution errors"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Setup response with tool use
        initial_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool_123"

        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"

        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "I encountered an error while searching."

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = [initial_response, final_response]

        # Configure tool manager to return error
        mock_tool_manager.execute_tool.return_value = (
            "Search error: Database connection failed"
        )

        generator = AIGenerator(mock_config)
        tools = mock_tool_manager.get_tool_definitions()

        result = generator.generate_response(
            query="Test query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "I encountered an error while searching."

        # Verify tool was executed and error was handled
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test"
        )

        # Verify two API calls were made (round 1 + round 2)
        assert (
            prevent_actual_api_calls[
                "anthropic"
            ].return_value.messages.create.call_count
            == 2
        )

    def test_generate_response_openai(self, mock_config, prevent_actual_api_calls):
        """Test OpenAI response generation (tools are ignored)"""
        mock_config.LLM_PROVIDER = "openai"
        mock_config.OPENAI_API_KEY = "test-key"

        # Setup mock response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "OpenAI direct response"

        prevent_actual_api_calls[
            "openai"
        ].return_value.chat.completions.create.return_value = mock_completion

        generator = AIGenerator(mock_config)

        result = generator.generate_response(
            query="What is machine learning?",
            tools=None,  # Tools are ignored for OpenAI
            tool_manager=None,
        )

        assert result == "OpenAI direct response"

        # Verify direct API call was made without tools
        call_args = prevent_actual_api_calls[
            "openai"
        ].return_value.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2  # system + user
        assert messages[1]["content"] == "What is machine learning?"

    def test_generate_response_openai_missing_api_key(self, mock_config):
        """Test OpenAI with missing API key raises error"""
        mock_config.LLM_PROVIDER = "openai"
        mock_config.OPENAI_API_KEY = ""

        generator = AIGenerator(mock_config)

        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            generator.generate_response("Test query")

    def test_generate_response_gemini(self, mock_config, prevent_actual_api_calls):
        """Test Gemini response generation (tools are ignored)"""
        mock_config.LLM_PROVIDER = "gemini"
        mock_config.GEMINI_API_KEY = "test-key"

        # Setup mock model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini direct response"
        mock_model.generate_content.return_value = mock_response

        prevent_actual_api_calls["gemini_model"].return_value = mock_model

        generator = AIGenerator(mock_config)

        result = generator.generate_response(
            query="Explain neural networks",
            tools=None,  # Tools are ignored for Gemini
            tool_manager=None,
        )

        assert result == "Gemini direct response"

        # Verify direct API call was made
        mock_model.generate_content.assert_called_once_with("Explain neural networks")

    def test_generate_response_gemini_missing_api_key(self, mock_config):
        """Test Gemini with missing API key raises error"""
        mock_config.LLM_PROVIDER = "gemini"
        mock_config.GEMINI_API_KEY = ""

        generator = AIGenerator(mock_config)

        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            generator.generate_response("Test query")

    def test_generate_response_gemini_exception_fallback(
        self, mock_config, prevent_actual_api_calls
    ):
        """Test Gemini response exception handling with fallback"""
        mock_config.LLM_PROVIDER = "gemini"
        mock_config.GEMINI_API_KEY = "test-key"

        # Setup mock model with response that raises exception on .text
        mock_model = Mock()
        mock_response = Mock()

        # Mock the text property to raise an exception when accessed
        type(mock_response).text = property(
            lambda self: (_ for _ in ()).throw(Exception("Text access error"))
        )
        mock_response.__str__ = Mock(return_value="Fallback response")
        mock_model.generate_content.return_value = mock_response

        prevent_actual_api_calls["gemini_model"].return_value = mock_model

        generator = AIGenerator(mock_config)

        result = generator.generate_response("Test query")

        assert result == "Fallback response"

    def test_generate_response_with_conversation_history(
        self, mock_config, prevent_actual_api_calls
    ):
        """Test response generation with conversation history"""
        mock_config.LLM_PROVIDER = "anthropic"

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history"
        mock_response.stop_reason = "end_turn"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.return_value = mock_response

        generator = AIGenerator(mock_config)

        result = generator.generate_response(
            query="Follow-up question",
            conversation_history="Previous conversation context",
        )

        assert result == "Response with history"

        # Verify history was included in system message
        call_args = prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.call_args
        system_message = call_args[1]["system"]
        assert "Previous conversation:\nPrevious conversation context" in system_message

    def test_handle_tool_execution_multiple_tools(
        self, mock_config, mock_tool_manager, prevent_actual_api_calls
    ):
        """Test handling of multiple tool executions in one response"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Setup response with multiple tool uses
        initial_response = Mock()
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "first search"}
        tool_block1.id = "tool_1"

        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.input = {"course_title": "ML Course"}
        tool_block2.id = "tool_2"

        initial_response.content = [tool_block1, tool_block2]
        initial_response.stop_reason = "tool_use"

        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Results from both tools"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = [initial_response, final_response]

        # Configure tool manager responses
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Outline result 2",
        ]

        generator = AIGenerator(mock_config)
        tools = mock_tool_manager.get_tool_definitions()

        result = generator.generate_response(
            query="Test query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Results from both tools"

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="first search"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="ML Course"
        )

    def test_generate_response_anthropic_two_round_sequential_tool_use(
        self, mock_config, mock_tool_manager, prevent_actual_api_calls
    ):
        """Test Anthropic response with two rounds of sequential tool use"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Round 1: Get course outline
        round1_response = Mock()
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "get_course_outline"
        tool_block1.input = {"course_title": "Machine Learning Basics"}
        tool_block1.id = "tool_1"

        round1_response.content = [tool_block1]
        round1_response.stop_reason = "tool_use"

        # Round 2: Search based on outline results
        round2_response = Mock()
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "search_course_content"
        tool_block2.input = {"query": "neural networks", "course_name": "Deep Learning"}
        tool_block2.id = "tool_2"

        round2_response.content = [tool_block2]
        round2_response.stop_reason = "tool_use"

        # Final response after 2 rounds
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = (
            "Based on both the outline and search, here's a comprehensive answer..."
        )
        final_response.stop_reason = "end_turn"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Configure tool manager responses
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 4 covers neural networks",
            "Search results: Neural networks are computational models",
        ]

        generator = AIGenerator(mock_config)
        tools = mock_tool_manager.get_tool_definitions()

        result = generator.generate_response(
            query="Search for a course that discusses the same topic as lesson 4 of Machine Learning Basics",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert (
            result
            == "Based on both the outline and search, here's a comprehensive answer..."
        )

        # Verify both tools were executed in sequence
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Machine Learning Basics"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="neural networks",
            course_name="Deep Learning",
        )

        # Verify three API calls were made (round 1 + round 2 + final)
        assert (
            prevent_actual_api_calls[
                "anthropic"
            ].return_value.messages.create.call_count
            == 3
        )

    def test_generate_response_anthropic_max_rounds_reached(
        self, mock_config, mock_tool_manager, prevent_actual_api_calls
    ):
        """Test that system terminates after max rounds and makes final call without tools"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Round 1: Tool use
        round1_response = Mock()
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "first search"}
        tool_block1.id = "tool_1"

        round1_response.content = [tool_block1]
        round1_response.stop_reason = "tool_use"

        # Round 2: Tool use again
        round2_response = Mock()
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "search_course_content"
        tool_block2.input = {"query": "second search"}
        tool_block2.id = "tool_2"

        round2_response.content = [tool_block2]
        round2_response.stop_reason = "tool_use"

        # Final response (made without tools after max rounds)
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = (
            "Here's my final answer based on the search results..."
        )
        final_response.stop_reason = "end_turn"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Configure tool manager responses
        mock_tool_manager.execute_tool.side_effect = [
            "First search results",
            "Second search results",
        ]

        generator = AIGenerator(mock_config)
        tools = mock_tool_manager.get_tool_definitions()

        result = generator.generate_response(
            query="Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Here's my final answer based on the search results..."

        # Verify both tools were executed (2 rounds)
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify three API calls: round 1 + round 2 + final (without tools)
        assert (
            prevent_actual_api_calls[
                "anthropic"
            ].return_value.messages.create.call_count
            == 3
        )

        # Verify final call was made without tools
        final_call_args = prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.call_args_list[2]
        assert "tools" not in final_call_args[1]

    def test_anthropic_api_error_handling(self, mock_config, prevent_actual_api_calls):
        """Test handling of Anthropic API errors"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Configure API to raise exception
        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = Exception("API Error")

        generator = AIGenerator(mock_config)

        # Should propagate the exception (not caught by AIGenerator)
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("Test query")

    def test_system_prompt_content(self, mock_config, prevent_actual_api_calls):
        """Test that system prompt contains expected instructions"""
        mock_config.LLM_PROVIDER = "anthropic"

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.return_value = mock_response

        generator = AIGenerator(mock_config)
        generator.generate_response("Test query")

        # Verify system prompt contains expected content
        call_args = prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.call_args
        system_message = call_args[1]["system"]

        assert "AI assistant specialized in course materials" in system_message
        assert "search_course_content" in system_message
        assert "get_course_outline" in system_message
        assert "Tool Usage:" in system_message
        assert "You can make up to 2 rounds of tool calls" in system_message
        assert "Response Protocol:" in system_message

    def test_base_params_configuration(self, mock_config):
        """Test that base parameters are correctly configured"""
        mock_config.LLM_PROVIDER = "anthropic"
        mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

        with patch("anthropic.Anthropic"):
            generator = AIGenerator(mock_config)

            expected_params = {
                "model": "claude-sonnet-4-20250514",
                "temperature": 0,
                "max_tokens": 800,
            }

            assert generator._base_params == expected_params

    def test_provider_normalization(self, mock_config):
        """Test that provider names are normalized correctly"""
        test_cases = [
            ("ANTHROPIC", "anthropic"),
            ("  OpenAI  ", "openai"),
            ("Gemini", "gemini"),
        ]

        for input_provider, expected in test_cases:
            mock_config.LLM_PROVIDER = input_provider

            with (
                patch("anthropic.Anthropic"),
                patch("openai.OpenAI"),
                patch("google.generativeai"),
            ):

                generator = AIGenerator(mock_config)
                assert generator.provider == expected


class TestAIGeneratorToolIntegration:
    """Integration tests for AIGenerator with actual tool interactions"""

    def test_end_to_end_tool_flow(self, mock_config, prevent_actual_api_calls):
        """Test complete tool execution flow"""
        mock_config.LLM_PROVIDER = "anthropic"

        # Create a real tool manager with mock vector store
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import SearchResults

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Machine learning is a subset of artificial intelligence."],
            metadata=[{"course_title": "ML Basics", "lesson_number": 1}],
            distances=[0.9],
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Setup Anthropic responses
        initial_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "what is machine learning"}
        tool_block.id = "tool_123"

        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"

        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = (
            "Machine learning is a powerful technique for data analysis."
        )

        prevent_actual_api_calls[
            "anthropic"
        ].return_value.messages.create.side_effect = [initial_response, final_response]

        generator = AIGenerator(mock_config)

        result = generator.generate_response(
            query="Tell me about machine learning",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Machine learning is a powerful technique for data analysis."

        # Verify tool was actually executed
        mock_vector_store.search.assert_called_once_with(
            query="what is machine learning", course_name=None, lesson_number=None
        )

        # Verify sources were stored
        assert len(search_tool.last_sources) == 1
        assert "ML Basics - Lesson 1" in search_tool.last_sources[0]
