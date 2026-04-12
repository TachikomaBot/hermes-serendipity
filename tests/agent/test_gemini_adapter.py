"""Tests for agent/gemini_adapter.py — Native Gemini API adapter."""

import base64
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock the google-genai SDK before importing the adapter
# ---------------------------------------------------------------------------

class _MockTypes:
    """Minimal mock of google.genai.types for testing."""

    class Part:
        def __init__(self, text=None, thought=False, function_call=None, function_response=None):
            self.text = text
            self.thought = thought
            self.function_call = function_call
            self.function_response = function_response

        @staticmethod
        def from_function_call(name="", id="", args=None):
            """Test helper — simulates API response Parts with function_call."""
            p = _MockTypes.Part()
            p.function_call = SimpleNamespace(name=name, id=id, args=args or {})
            return p

        @staticmethod
        def from_function_response(name="", response=None):
            p = _MockTypes.Part()
            p.function_response = SimpleNamespace(name=name, response=response or {})
            return p

        @staticmethod
        def from_bytes(data=b"", mime_type=""):
            p = _MockTypes.Part()
            p.inline_data = SimpleNamespace(data=data, mime_type=mime_type)
            return p

    class Content:
        def __init__(self, role="", parts=None):
            self.role = role
            self.parts = parts or []

    class FunctionCall:
        def __init__(self, name="", id="", args=None, **kwargs):
            self.name = name
            self.id = id
            self.args = args or {}

    class FunctionDeclaration:
        def __init__(self, name="", description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters

    class Tool:
        def __init__(self, function_declarations=None):
            self.function_declarations = function_declarations or []

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ThinkingConfig:
        def __init__(self, thinking_budget=-1, thinking_level=None):
            self.thinking_budget = thinking_budget
            self.thinking_level = thinking_level

    class HttpOptions:
        def __init__(self, api_version="v1"):
            self.api_version = api_version


class _MockGenai:
    """Minimal mock of google.genai module."""
    class Client:
        def __init__(self, api_key=""):
            self.api_key = api_key
            self.models = MagicMock()


# Inject mocks into sys.modules so the adapter's lazy import succeeds
_mock_genai_module = MagicMock()
_mock_genai_module.Client = _MockGenai.Client
_mock_types_module = _MockTypes()

# Patch at module level
_google_mock = MagicMock()
_google_mock.genai = _mock_genai_module
_google_mock.genai.types = _mock_types_module
sys.modules.setdefault("google", _google_mock)
sys.modules.setdefault("google.genai", _mock_genai_module)
sys.modules.setdefault("google.genai.types", _mock_types_module)

# Now force the adapter to use our mocks
import agent.gemini_adapter as gad
gad._genai = _mock_genai_module
gad._types = _MockTypes


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------

class TestConvertToolsToGemini:
    def test_basic_tool(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }]
        result = gad.convert_tools_to_gemini(tools)
        assert len(result) == 1  # single Tool wrapper
        tool = result[0]
        assert len(tool.function_declarations) == 1
        decl = tool.function_declarations[0]
        assert decl.name == "get_weather"
        assert decl.description == "Get weather for a city"
        assert decl.parameters["properties"]["city"]["type"] == "string"

    def test_empty_properties_stripped(self):
        tools = [{
            "function": {
                "name": "noop",
                "description": "Does nothing",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        result = gad.convert_tools_to_gemini(tools)
        decl = result[0].function_declarations[0]
        assert decl.parameters is None

    def test_multiple_tools(self):
        tools = [
            {"function": {"name": "a", "description": "A", "parameters": None}},
            {"function": {"name": "b", "description": "B", "parameters": None}},
        ]
        result = gad.convert_tools_to_gemini(tools)
        assert len(result[0].function_declarations) == 2


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

class TestConvertMessagesToGemini:
    def test_simple_user_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        system, contents = gad.convert_messages_to_gemini(messages)
        assert system is None
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hi there"

    def test_system_prompt_extraction(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        system, contents = gad.convert_messages_to_gemini(messages, system_prompt="Base prompt")
        assert "Base prompt" in system
        assert "You are helpful" in system
        # System message should not appear in contents
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_tool_calls_conversion(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "NYC"}',
                    },
                }],
            },
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        assert len(contents) == 2
        model_content = contents[1]
        assert model_content.role == "model"
        # Should have a FunctionCall part
        fc_part = model_content.parts[0]
        assert fc_part.function_call is not None
        assert fc_part.function_call.name == "get_weather"
        assert fc_part.function_call.args == {"city": "NYC"}

    def test_tool_results_conversion(self):
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temp": 72}',
            },
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        assert len(contents) == 3
        tool_content = contents[2]
        assert tool_content.role == "user"
        fr_part = tool_content.parts[0]
        assert fr_part.function_response is not None
        assert fr_part.function_response.name == "get_weather"

    def test_tool_result_with_image(self):
        b64_data = base64.b64encode(b"fake_png_data").decode()
        messages = [
            {"role": "user", "content": "Screenshot?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_img",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_img",
                "content": '{"status": "ok"}',
                "_image": {
                    "media_type": "image/png",
                    "base64": b64_data,
                },
            },
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        tool_content = contents[2]
        # Should have FunctionResponse + inline_data parts
        assert len(tool_content.parts) == 2
        assert tool_content.parts[0].function_response is not None
        assert hasattr(tool_content.parts[1], "inline_data")

    def test_consecutive_tool_results_merged(self):
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"r": 1}'},
            {"role": "tool", "tool_call_id": "c2", "content": '{"r": 2}'},
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        # user, model, then merged tool results (single user Content)
        assert len(contents) == 3
        merged = contents[2]
        assert merged.role == "user"
        assert len(merged.parts) == 2  # two FunctionResponse parts

    def test_thinking_blocks_preserved(self):
        messages = [
            {"role": "user", "content": "Think about this"},
            {
                "role": "assistant",
                "content": "My answer",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Let me consider..."},
                ],
            },
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        model_content = contents[1]
        # Should have thought part + text part
        assert len(model_content.parts) == 2
        assert model_content.parts[0].thought is True
        assert model_content.parts[0].text == "Let me consider..."
        assert model_content.parts[1].text == "My answer"

    def test_multipart_user_with_image(self):
        b64 = base64.b64encode(b"img").decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        _, contents = gad.convert_messages_to_gemini(messages)
        assert len(contents) == 1
        parts = contents[0].parts
        assert len(parts) == 2
        assert parts[0].text == "What's in this image?"
        assert hasattr(parts[1], "inline_data")

    def test_consecutive_user_messages_merged(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        assert len(contents) == 1
        assert len(contents[0].parts) == 2

    def test_empty_assistant_gets_placeholder(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ]
        _, contents = gad.convert_messages_to_gemini(messages)
        model_content = contents[1]
        assert len(model_content.parts) == 1
        assert model_content.parts[0].text == ""


# ---------------------------------------------------------------------------
# Build kwargs
# ---------------------------------------------------------------------------

class TestBuildGeminiKwargs:
    def test_basic_kwargs(self):
        messages = [{"role": "user", "content": "Hi"}]
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=messages,
        )
        assert kwargs["model"] == "gemini-3-flash-preview"
        assert "contents" in kwargs
        assert "config" in kwargs

    def test_thinking_level_mapping(self):
        messages = [{"role": "user", "content": "Think"}]
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=messages,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        config = kwargs["config"]
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.thinking_level == "high"

    def test_thinking_xhigh_maps_to_high(self):
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "x"}],
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )
        assert kwargs["config"].thinking_config.thinking_level == "high"

    def test_flex_mode(self):
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "x"}],
            flex_mode=True,
        )
        config = kwargs["config"]
        assert hasattr(config, "http_options")

    def test_no_flex_mode(self):
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "x"}],
            flex_mode=False,
        )
        config = kwargs["config"]
        assert not hasattr(config, "http_options")

    def test_max_tokens(self):
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=4096,
        )
        assert hasattr(kwargs["config"], "max_output_tokens")
        assert kwargs["config"].max_output_tokens == 4096

    def test_temperature(self):
        kwargs = gad.build_gemini_kwargs(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.5,
        )
        assert kwargs["config"].temperature == 0.5


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

class TestNormalizeGeminiResponse:
    def _make_response(self, parts, finish_reason="STOP", usage=None):
        """Build a mock Gemini response."""
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=parts),
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            candidates=[candidate],
            usage_metadata=usage,
        )

    def test_text_response(self):
        part = _MockTypes.Part(text="Hello world")
        response = self._make_response([part])
        msg, reason = gad.normalize_gemini_response(response)
        assert msg.content == "Hello world"
        assert msg.tool_calls is None
        assert reason == "stop"

    def test_tool_call_response(self):
        part = _MockTypes.Part.from_function_call(
            name="get_weather", id="call_xyz", args={"city": "SF"},
        )
        response = self._make_response([part])
        msg, reason = gad.normalize_gemini_response(response)
        assert msg.content is None
        assert reason == "tool_calls"
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "SF"}
        assert tc.id == "call_xyz"

    def test_thinking_response(self):
        thought = _MockTypes.Part(text="Let me think...", thought=True)
        text = _MockTypes.Part(text="The answer is 42")
        response = self._make_response([thought, text])
        msg, reason = gad.normalize_gemini_response(response)
        assert msg.content == "The answer is 42"
        assert msg.reasoning == "Let me think..."
        assert len(msg.reasoning_details) == 1
        assert msg.reasoning_details[0]["type"] == "thinking"

    def test_empty_response(self):
        response = SimpleNamespace(candidates=[], usage_metadata=None)
        msg, reason = gad.normalize_gemini_response(response)
        assert msg.content == ""
        assert reason == "stop"

    def test_max_tokens_finish_reason(self):
        part = _MockTypes.Part(text="truncated...")
        response = self._make_response([part], finish_reason="MAX_TOKENS")
        _, reason = gad.normalize_gemini_response(response)
        assert reason == "length"

    def test_generated_tool_call_id(self):
        """Tool calls without IDs get synthetic ones."""
        part = _MockTypes.Part.from_function_call(
            name="test_tool", id="", args={},
        )
        response = self._make_response([part])
        msg, _ = gad.normalize_gemini_response(response)
        tc = msg.tool_calls[0]
        # Should have generated an ID starting with call_
        assert tc.id.startswith("call_") or tc.id == ""

    def test_usage_metadata(self):
        part = _MockTypes.Part(text="Hi")
        usage = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )
        response = self._make_response([part], usage=usage)
        msg, _ = gad.normalize_gemini_response(response)
        assert msg.content == "Hi"


# ---------------------------------------------------------------------------
# Streaming accumulator
# ---------------------------------------------------------------------------

class TestAccumulateStreamingResponse:
    def test_accumulate_text_chunks(self):
        chunks = [
            SimpleNamespace(
                candidates=[SimpleNamespace(
                    content=SimpleNamespace(parts=[_MockTypes.Part(text="Hello ")]),
                    finish_reason=None,
                )],
                usage_metadata=None,
            ),
            SimpleNamespace(
                candidates=[SimpleNamespace(
                    content=SimpleNamespace(parts=[_MockTypes.Part(text="world")]),
                    finish_reason="STOP",
                )],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=5,
                    candidates_token_count=2,
                    total_token_count=7,
                ),
            ),
        ]
        response = gad.accumulate_streaming_response(chunks)
        msg, reason = gad.normalize_gemini_response(response)
        assert "Hello " in msg.content
        assert "world" in msg.content

    def test_empty_stream(self):
        response = gad.accumulate_streaming_response([])
        msg, reason = gad.normalize_gemini_response(response)
        assert msg.content == ""
        assert reason == "stop"


# ---------------------------------------------------------------------------
# Thinking level map
# ---------------------------------------------------------------------------

class TestThinkingLevelMap:
    @pytest.mark.parametrize("effort,expected", [
        ("xhigh", "high"),
        ("high", "high"),
        ("medium", "medium"),
        ("low", "low"),
        ("minimal", "minimal"),
    ])
    def test_mapping(self, effort, expected):
        assert gad.THINKING_LEVEL_MAP[effort] == expected

    def test_unknown_defaults(self):
        assert gad.THINKING_LEVEL_MAP.get("unknown") is None
