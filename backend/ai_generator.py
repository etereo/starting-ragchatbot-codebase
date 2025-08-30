from typing import List, Optional, Dict, Any


class AIGenerator:
    """Provider-agnostic LLM wrapper for generating responses.

    Supports Anthropic, OpenAI, and Gemini. Provider is selected via
    the `LLM_PROVIDER` environment variable (see config).
    """

    # System prompt for Anthropic (with tool calling)
    SYSTEM_PROMPT_ANTHROPIC = (
        " You are an AI assistant specialized in course materials and educational "
        "content with access to tools for course information.\n\n"
        "Tools:\n"
        "- search_course_content: Find relevant course passages; supports optional course and lesson filters.\n"
        "- get_course_outline: Retrieve a course outline (title, link, full lesson list).\n\n"
        "Tool Usage:\n"
        "- Use tools when needed. You can make up to 2 rounds of tool calls to gather information.\n"
        "- First round: Use tools to gather initial information.\n"
        "- Second round: Use tools to gather additional context if the first round's results suggest more information is needed.\n"
        "- Content questions (definitions, explanations, where-discussed): use search_course_content.\n"
        "- Outline questions (e.g., 'show outline', 'what are the lessons in <course>'): use get_course_outline.\n"
        "- If a tool yields no results, say so clearly without offering alternatives.\n\n"
        "Response Protocol:\n"
        "- General knowledge questions: Answer from your knowledge without tools.\n"
        "- Course-specific content: Call search_course_content first, then answer concisely.\n"
        "- Course outline requests: Ensure your final answer includes ALL of the following:\n"
        "  • Course title\n"
        "  • Course link (or 'N/A' if missing)\n"
        "  • Every lesson as a numbered list with its lesson number and title\n"
        "- No meta-commentary: Provide direct answers only — do not describe your tools or reasoning.\n\n"
        "All responses must be: 1) Brief and focused, 2) Educational, 3) Clear, 4) Example-supported when helpful."
    )
    
    # System prompt for OpenAI/Gemini (without tool calling)
    SYSTEM_PROMPT_NO_TOOLS = (
        "You are an AI assistant specialized in course materials and educational content. "
        "You will be provided with course materials context to help answer questions. "
        "Your responses should be: 1) Brief and focused, 2) Educational, 3) Clear, 4) Example-supported when helpful. "
        "Base your answers on the provided course materials when available."
    )

    def __init__(self, cfg):
        self.provider = (cfg.LLM_PROVIDER or "anthropic").strip().lower()
        self.cfg = cfg

        # Lazy-import provider SDKs to avoid hard dependency at import time.
        if self.provider == "anthropic":
            import anthropic  # type: ignore
            self._anthropic = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
            self._anthropic_model = cfg.ANTHROPIC_MODEL
            self._base_params = {"model": self._anthropic_model, "temperature": 0, "max_tokens": 800}
        elif self.provider == "openai":
            from openai import OpenAI  # type: ignore
            self._openai = OpenAI(api_key=cfg.OPENAI_API_KEY)
            self._openai_model = cfg.OPENAI_MODEL
        elif self.provider == "gemini":
            import google.generativeai as genai  # type: ignore
            if not cfg.GEMINI_API_KEY:
                # allow None; error will be raised on first call
                pass
            genai.configure(api_key=cfg.GEMINI_API_KEY)
            # We'll set system instruction at call time to include history.
            self._gemini = genai
            self._gemini_model_name = cfg.GEMINI_MODEL
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {cfg.LLM_PROVIDER}")

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Generate an AI response using the selected provider.

        For Anthropic, basic tool use is supported (as before). For OpenAI and Gemini,
        tools are ignored and a direct response is returned.
        """

        # Choose appropriate system prompt based on provider
        base_prompt = (
            self.SYSTEM_PROMPT_ANTHROPIC if self.provider == "anthropic" 
            else self.SYSTEM_PROMPT_NO_TOOLS
        )
        
        system_content = (
            f"{base_prompt}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else base_prompt
        )

        if self.provider == "anthropic":
            return self._call_anthropic(system_content, query, tools, tool_manager)
        if self.provider == "openai":
            return self._call_openai(system_content, query)
        if self.provider == "gemini":
            return self._call_gemini(system_content, query)
        # Should not reach here due to __init__ check
        raise RuntimeError("Invalid provider configuration")

    # --- Provider implementations ---

    def _call_anthropic(
        self,
        system_content: str,
        query: str,
        tools: Optional[List],
        tool_manager,
    ) -> str:
        max_rounds = 2
        round_count = 0
        messages = [{"role": "user", "content": query}]
        
        while round_count < max_rounds:
            api_params: Dict[str, Any] = {
                **self._base_params,
                "messages": messages,
                "system": system_content,
            }
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self._anthropic.messages.create(**api_params)
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Check if tool use occurred
            if getattr(response, "stop_reason", None) == "tool_use" and tool_manager:
                # Execute tools and add results to conversation
                tool_results = self._execute_tools_and_build_results(response, tool_manager)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                    round_count += 1
                    continue
            
            # No tool use - return final response
            return response.content[0].text
        
        # Max rounds reached - make final call without tools
        final_params = {
            **self._base_params,
            "messages": messages,
            "system": system_content,
        }
        final_response = self._anthropic.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_tools_and_build_results(self, response, tool_manager):
        """Execute all tools from a response and return formatted results."""
        tool_results = []
        for content_block in response.content:
            if getattr(content_block, "type", None) == "tool_use":
                tool_result = tool_manager.execute_tool(content_block.name, **content_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result,
                })
        return tool_results

    def _call_openai(self, system_content: str, query: str) -> str:
        if not getattr(self.cfg, "OPENAI_API_KEY", ""):
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        result = self._openai.chat.completions.create(
            model=self._openai_model,
            temperature=0,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ],
        )
        return (result.choices[0].message.content or "").strip()

    def _call_gemini(self, system_content: str, query: str) -> str:
        if not getattr(self.cfg, "GEMINI_API_KEY", ""):
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        model = self._gemini.GenerativeModel(self._gemini_model_name, system_instruction=system_content)
        resp = model.generate_content(query)
        # Handle candidates/safety
        try:
            return (resp.text or "").strip()
        except Exception:
            # Fallback for SDK variations
            return str(resp)
