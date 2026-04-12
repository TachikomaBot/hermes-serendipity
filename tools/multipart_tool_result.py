"""Multipart tool result support — extract images from tool results for vision models.

Tools signal "this result includes an image" by including an ``_image`` key in
their JSON result:

    {"status": "ok", "analysis": "...", "_image": {
        "media_type": "image/png",
        "base64": "<base64-data>",
        "resolution": "high"
    }}

The harness calls ``extract_multipart_content()`` after tool execution but before
message construction.  For vision-capable providers (Gemini), the image is converted
to multipart content.  For others, the ``_image`` key is silently stripped so only
clean text reaches the model.

The ``resolution`` field is advisory — it maps to Gemini's ``media_resolution``
levels (low=280 tokens, medium=560, high=1120).  Currently passed through as-is;
a future native Gemini API path could use it for per-image resolution control.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Providers that support vision via multipart content in tool results.
_VISION_PROVIDERS = {"gemini", "google"}

# Approximate token costs for image parts (Gemini high resolution).
IMAGE_TOKEN_ESTIMATE = 1120


def extract_multipart_content(function_result: str, provider: str = None):
    """Parse tool result JSON, extract ``_image`` if present, return content.

    For vision providers (Gemini):
        Returns a list with text + image_url parts (OpenAI-compatible format).

    For other providers or when no image is present:
        Returns a plain string (``_image`` stripped to save context).

    Args:
        function_result: Raw tool result string (expected to be JSON).
        provider: Provider name (e.g. "gemini", "anthropic", "openai").

    Returns:
        str or list: Content suitable for the ``tool`` message's ``content`` field.
    """
    if not function_result or not isinstance(function_result, str):
        return function_result

    try:
        parsed = json.loads(function_result)
    except (json.JSONDecodeError, TypeError):
        return function_result

    if not isinstance(parsed, dict) or "_image" not in parsed:
        return function_result

    # Pop the image data — always strip it from the text portion.
    image_data = parsed.pop("_image")
    text_content = json.dumps(parsed, ensure_ascii=False)

    # Non-vision providers get text only.
    if provider not in _VISION_PROVIDERS:
        return text_content

    # Validate image payload.
    media_type = image_data.get("media_type", "image/png")
    b64 = image_data.get("base64", "")
    if not b64:
        logger.warning("_image key present but base64 data is empty")
        return text_content

    logger.debug(
        "Multipart tool result: %s, ~%d KB image",
        media_type, len(b64) * 3 // 4 // 1024,
    )

    return [
        {"type": "text", "text": text_content},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        },
    ]


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
