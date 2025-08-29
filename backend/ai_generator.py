from typing import List, Optional, Dict, Any


class AIGenerator:
    """Provider-agnostic LLM wrapper for generating responses.

    Supports Anthropic, OpenAI, and Gemini. Provider is selected via
    the `LLM_PROVIDER` environment variable (see config).
    """

    SYSTEM_PROMPT = (
        " You are an AI assistant specialized in course materials and educational "
        "content with access to a comprehensive search tool for course information.\n\n"
        "Search Tool Usage:\n"
        "- Use the search tool only for questions about specific course content or detailed educational materials\n"
        "- One search per query maximum\n"
        "- Synthesize search results into accurate, fact-based responses\n"
        "- If search yields no results, state this clearly without offering alternatives\n\n"
        "Response Protocol:\n"
        "- General knowledge questions: Answer using existing knowledge without searching\n"
        "- Course-specific questions: Search first, then answer\n"
        "- No meta-commentary: Provide direct answers only — no reasoning process, search explanations, or question-type analysis\n"
        "  Do not mention 'based on the search results'\n\n"
        "All responses must be: 1) Brief and focused, 2) Educational, 3) Clear, 4) Example-supported when helpful."
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

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
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
        api_params: Dict[str, Any] = {
            **self._base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self._anthropic.messages.create(**api_params)

        if getattr(response, "stop_reason", None) == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        return response.content[0].text

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        tool_results = []
        for content_block in initial_response.content:
            if getattr(content_block, "type", None) == "tool_use":
                tool_result = tool_manager.execute_tool(content_block.name, **content_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        final_params = {**self._base_params, "messages": messages, "system": base_params["system"]}
        final_response = self._anthropic.messages.create(**final_params)
        return final_response.content[0].text

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
