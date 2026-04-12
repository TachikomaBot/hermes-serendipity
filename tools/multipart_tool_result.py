"""Multipart tool result support — extract images from tool results for vision models.

Tools signal "this result includes an image" by including an ``_image`` key in
their JSON result:

    {"status": "ok", "analysis": "...", "_image": {
        "media_type": "image/png",
        "base64": "<base64-data>",
        "resolution": "high"
    }}

The harness calls ``extract_image_from_result()`` after tool execution.  The
``_image`` key is always stripped from the tool message text.  For vision-capable
providers, the image is injected as a **separate user message** immediately after
the tool result — Gemini's OpenAI-compatible endpoint does not support images in
tool-role messages, but does support them in user-role messages.

The ``resolution`` field is advisory — it maps to Gemini's ``media_resolution``
levels (low=280 tokens, medium=560, high=1120).  A future native Gemini API path
could use it for per-image resolution control.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Providers that support vision via user-message image injection.
_VISION_PROVIDERS = {"gemini", "google"}

# Approximate token costs for image parts (Gemini high resolution).
IMAGE_TOKEN_ESTIMATE = 1120


def extract_image_from_result(function_result: str, provider: str = None):
    """Parse tool result JSON, extract ``_image`` if present.

    Always strips ``_image`` from the text content (so no base64 bloats the
    tool message).  For vision providers, returns the raw image data dict
    so the caller can attach it to the tool message for native handling.

    Args:
        function_result: Raw tool result string (expected to be JSON).
        provider: Provider name (e.g. "gemini", "anthropic", "openai").

    Returns:
        tuple: (cleaned_text: str, image_data: dict | None)
            - cleaned_text: tool result with _image stripped
            - image_data: raw image dict with media_type/base64 keys, or None
    """
    if not function_result or not isinstance(function_result, str):
        return function_result, None

    try:
        parsed = json.loads(function_result)
    except (json.JSONDecodeError, TypeError):
        return function_result, None

    if not isinstance(parsed, dict) or "_image" not in parsed:
        return function_result, None

    # Pop the image data — always strip it from the text portion.
    image_data = parsed.pop("_image")
    text_content = json.dumps(parsed, ensure_ascii=False)

    # Non-vision providers get text only, no image data.
    if provider not in _VISION_PROVIDERS:
        return text_content, None

    # Validate image payload.
    media_type = image_data.get("media_type", "image/png")
    b64 = image_data.get("base64", "")
    if not b64:
        logger.warning("_image key present but base64 data is empty")
        return text_content, None

    logger.debug(
        "Extracted image from tool result: %s, ~%d KB",
        media_type, len(b64) * 3 // 4 // 1024,
    )

    # Return the raw image data — the adapter handles native injection.
    return text_content, {"media_type": media_type, "base64": b64}


def strip_images_from_content(content):
    """Strip image parts from multipart content, returning text only.

    Useful before persisting to DB or logging.  Idempotent on strings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts) if texts else "[image content stripped]"
    return str(content)


def estimate_content_tokens(content) -> int:
    """Estimate token count for content that may include images.

    - String content: len // 4 (standard rough estimate).
    - List content: sum text tokens + IMAGE_TOKEN_ESTIMATE per image part.
    """
    if isinstance(content, str):
        return (len(content) + 3) // 4
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                total += (len(part.get("text", "")) + 3) // 4
            elif part.get("type") == "image_url":
                total += IMAGE_TOKEN_ESTIMATE
        return total
    return (len(str(content)) + 3) // 4
