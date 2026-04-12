"""Native Google Gemini API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Google's native Gemini ``generateContent`` API via the ``google-genai`` SDK.

Follows the same pattern as ``anthropic_adapter.py`` — all provider-specific
logic is isolated here.  The adapter handles:

- Message conversion (OpenAI roles → Gemini Content/Part)
- Tool schema conversion (OpenAI function schemas → FunctionDeclaration)
- Response normalization (Gemini response → OpenAI-shaped SimpleNamespace)
- Images in tool results (via inline_data in FunctionResponse)
- Thinking / reasoning (thinkingLevel control, thought signature preservation)
- Flex inference (service_tier: "flex")
"""

import base64
import json
import logging
import re
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded SDK reference — import on first use.
_genai = None
_types = None


def _ensure_sdk():
    """Lazy-import the google-genai SDK."""
    global _genai, _types
    if _genai is None:
        try:
            from google import genai
            from google.genai import types
            _genai = genai
            _types = types
        except ImportError:
            raise ImportError(
                "google-genai package is required for native Gemini support. "
                "Install it with: pip install google-genai"
            )
    return _genai, _types


# ── Thinking level mapping ──────────────────────────────────────────

THINKING_LEVEL_MAP = {
    "xhigh": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "minimal",
}


# ── Client creation ─────────────────────────────────────────────────

def build_gemini_client(api_key: str):
    """Create a native Gemini client.

    Args:
        api_key: Google API key (GOOGLE_API_KEY or GEMINI_API_KEY).

    Returns:
        genai.Client instance.
    """
    genai, _ = _ensure_sdk()
    return genai.Client(api_key=api_key)


# ── Tool schema conversion ──────────────────────────────────────────

def convert_tools_to_gemini(tools: List[Dict]) -> List:
    """Convert OpenAI-format tool schemas to Gemini FunctionDeclarations.

    Args:
        tools: List of OpenAI tool dicts with ``function`` key.

    Returns:
        List containing a single ``types.Tool`` with all declarations.
    """
    _, types = _ensure_sdk()
    declarations = []
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters")
        # Gemini doesn't accept empty properties — strip if absent
        if params and not params.get("properties"):
            params = None
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=params,
        ))
    return [types.Tool(function_declarations=declarations)]


# ── Message conversion ──────────────────────────────────────────────

def _parse_data_url(url: str) -> Tuple[str, bytes]:
    """Parse a data: URL into (mime_type, raw_bytes)."""
    # data:image/png;base64,iVBOR...
    match = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
    if match:
        return match.group(1), base64.b64decode(match.group(2))
    return "application/octet-stream", b""


def _build_tool_call_id_to_name(messages: List[Dict]) -> Dict[str, str]:
    """Build a mapping from tool_call_id → function name.

    Scans assistant messages for tool_calls to resolve names for tool results.
    """
    mapping = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                tc_id = tc.get("id", "")
                fn = tc.get("function", {})
                name = fn.get("name", "")
            else:
                # SimpleNamespace from normalized responses
                tc_id = getattr(tc, "id", "")
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn else ""
            if tc_id and name:
                mapping[tc_id] = name
    return mapping


def convert_messages_to_gemini(
    messages: List[Dict],
    system_prompt: str = "",
) -> Tuple[str, List]:
    """Convert OpenAI-format messages to Gemini Content/Part structure.

    Returns (system_instruction, gemini_contents) where:
    - system_instruction: extracted system messages (string)
    - gemini_contents: list of types.Content objects

    Key conversions:
    - role: assistant → model, tool → user (FunctionResponse part)
    - tool_calls: function.arguments JSON string → dict
    - images: data: URLs → inline_data parts
    - consecutive tool results merged into single Content
    """
    _, types = _ensure_sdk()

    # Collect system messages
    system_parts = []
    if system_prompt:
        system_parts.append(system_prompt)

    # Build tool_call_id → name mapping for FunctionResponse
    tc_id_to_name = _build_tool_call_id_to_name(messages)

    contents = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        # ── System messages → extracted ──
        if role == "system":
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part["text"])
            elif content:
                system_parts.append(str(content))
            continue

        # ── Assistant messages → model role ──
        if role == "assistant":
            parts = []

            # Preserve thinking blocks from previous responses
            for detail in m.get("reasoning_details") or []:
                if isinstance(detail, dict) and detail.get("type") == "thinking":
                    thinking_text = detail.get("thinking", "")
                    if thinking_text:
                        parts.append(types.Part(text=thinking_text, thought=True))

            # Text content
            if content:
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(types.Part(text=block["text"]))
                else:
                    parts.append(types.Part(text=str(content)))

            # Tool calls → FunctionCall parts
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    tc_id = tc.get("id", "")
                    args_raw = fn.get("arguments", "{}")
                else:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "") if fn else ""
                    tc_id = getattr(tc, "id", "")
                    args_raw = getattr(fn, "arguments", "{}") if fn else "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, ValueError):
                    args = {}
                # Build FunctionCall with id, then wrap in Part.
                # Part.from_function_call() doesn't accept id, but
                # FunctionCall() does — and we need to preserve it
                # for tool_call_id matching on round-trip.
                fc = types.FunctionCall(name=name, id=tc_id, args=args)
                parts.append(types.Part(function_call=fc))

            # Gemini rejects empty model content
            if not parts:
                parts = [types.Part(text="")]

            contents.append(types.Content(role="model", parts=parts))
            continue

        # ── Tool results → user role with FunctionResponse ──
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tool_name = tc_id_to_name.get(tc_id, "unknown_tool")
            result_text = content if isinstance(content, str) else json.dumps(content)
            if not result_text:
                result_text = "(no output)"

            # Parse result as JSON for structured response
            try:
                result_obj = json.loads(result_text)
            except (json.JSONDecodeError, TypeError):
                result_obj = {"result": result_text}

            parts = [types.Part.from_function_response(
                name=tool_name, response=result_obj,
            )]

            # Handle image data attached to tool result
            image_data = m.get("_image")
            if image_data:
                mime_type = image_data.get("media_type", "image/png")
                b64 = image_data.get("base64", "")
                if b64:
                    parts.append(types.Part.from_bytes(
                        data=base64.b64decode(b64),
                        mime_type=mime_type,
                    ))

            # Merge consecutive tool results into single Content
            # (Gemini requires alternating user/model turns)
            if (
                contents
                and contents[-1].role == "user"
                and contents[-1].parts
                and hasattr(contents[-1].parts[0], "function_response")
                and contents[-1].parts[0].function_response is not None
            ):
                contents[-1].parts.extend(parts)
            else:
                contents.append(types.Content(role="user", parts=parts))
            continue

        # ── User messages ──
        parts = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(types.Part(text=text))
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        mime_type, raw = _parse_data_url(url)
                        parts.append(types.Part.from_bytes(
                            data=raw, mime_type=mime_type,
                        ))
                    # Non-data URLs (http) could use file_data, skip for now
        else:
            if content:
                parts.append(types.Part(text=str(content)))

        if not parts:
            parts = [types.Part(text="")]

        # Merge consecutive user messages (Gemini requires alternating turns)
        if contents and contents[-1].role == "user":
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role="user", parts=parts))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


# ── Request building ────────────────────────────────────────────────

def build_gemini_kwargs(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    reasoning_config: Optional[Dict] = None,
    system_prompt: str = "",
    temperature: Optional[float] = None,
    media_resolution: Optional[str] = None,
    flex_mode: bool = True,
) -> Dict[str, Any]:
    """Build kwargs for ``client.models.generate_content()``.

    Returns a dict ready to be unpacked: ``client.models.generate_content(**kwargs)``.
    """
    _, types = _ensure_sdk()

    system_instruction, contents = convert_messages_to_gemini(messages, system_prompt)

    # Build GenerateContentConfig
    config_kwargs: Dict[str, Any] = {}

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    if tools:
        config_kwargs["tools"] = convert_tools_to_gemini(tools)

    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    if temperature is not None:
        config_kwargs["temperature"] = temperature

    if media_resolution:
        config_kwargs["media_resolution"] = media_resolution

    # Thinking / reasoning — use thinking_level (not thinking_budget;
    # the API rejects requests that set both).
    if reasoning_config and reasoning_config.get("enabled"):
        effort = reasoning_config.get("effort", "medium")
        thinking_level = THINKING_LEVEL_MAP.get(effort, "medium")
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
        )

    # Flex inference
    if flex_mode:
        config_kwargs["http_options"] = types.HttpOptions(
            api_version="v1alpha",
        )

    config = types.GenerateContentConfig(**config_kwargs)

    return {
        "model": model,
        "contents": contents,
        "config": config,
    }


# ── Response normalization ──────────────────────────────────────────

def normalize_gemini_response(response) -> Tuple[SimpleNamespace, str]:
    """Normalize Gemini response to match the shape expected by AIAgent.

    Returns (assistant_message, finish_reason) where assistant_message has
    .content, .tool_calls, .reasoning, and .reasoning_details attributes.
    """
    text_parts = []
    reasoning_parts = []
    reasoning_details = []
    tool_calls = []

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return (
            SimpleNamespace(
                content="",
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            "stop",
        )

    candidate = candidates[0]
    content = getattr(candidate, "content", None)

    if content and content.parts:
        for part in content.parts:
            # Check for thinking/thought parts
            if getattr(part, "thought", False):
                text = getattr(part, "text", "")
                if text:
                    reasoning_parts.append(text)
                    reasoning_details.append({
                        "type": "thinking",
                        "thinking": text,
                    })
                continue

            # Function calls
            fc = getattr(part, "function_call", None)
            if fc is not None:
                fc_id = getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                fc_args = getattr(fc, "args", {}) or {}
                tool_calls.append(
                    SimpleNamespace(
                        id=fc_id,
                        type="function",
                        function=SimpleNamespace(
                            name=getattr(fc, "name", ""),
                            arguments=json.dumps(fc_args),
                        ),
                    )
                )
                continue

            # Text parts
            text = getattr(part, "text", None)
            if text is not None:
                text_parts.append(text)

    # Map finish reason
    finish_reason_raw = getattr(candidate, "finish_reason", None)
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason_raw:
        # Gemini uses enum names like STOP, MAX_TOKENS, SAFETY, etc.
        reason_str = str(finish_reason_raw).upper()
        if "STOP" in reason_str:
            finish_reason = "stop"
        elif "MAX_TOKENS" in reason_str:
            finish_reason = "length"
        else:
            finish_reason = "stop"
    else:
        finish_reason = "stop"

    # Usage tracking
    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.debug(
            "Gemini usage: prompt=%s, completion=%s, total=%s",
            getattr(usage, "prompt_token_count", "?"),
            getattr(usage, "candidates_token_count", "?"),
            getattr(usage, "total_token_count", "?"),
        )

    return (
        SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=reasoning_details or None,
        ),
        finish_reason,
    )


# ── Streaming helpers ───────────────────────────────────────────────

def accumulate_streaming_response(chunks):
    """Accumulate streaming chunks into a final response-like object.

    Gemini streaming yields GenerateContentResponse objects where each
    chunk has complete parts (not deltas).  We merge them into a single
    response structure.

    Args:
        chunks: Iterable of GenerateContentResponse chunks.

    Returns:
        A response-like object compatible with ``normalize_gemini_response``.
    """
    _, types = _ensure_sdk()
    all_parts = []
    final_candidate = None
    final_usage = None

    for chunk in chunks:
        candidates = getattr(chunk, "candidates", None)
        if candidates:
            final_candidate = candidates[0]
            content = getattr(final_candidate, "content", None)
            if content and content.parts:
                all_parts.extend(content.parts)
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            final_usage = usage

    # Build a synthetic response
    return SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=all_parts),
            finish_reason=getattr(final_candidate, "finish_reason", None) if final_candidate else None,
        )] if all_parts or final_candidate else [],
        usage_metadata=final_usage,
    )
