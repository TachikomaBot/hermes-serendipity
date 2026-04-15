"""Autonomous continuation — lets the agent self-prompt after a delay.

The ``self_continue`` tool sets a signal that the gateway picks up after
the current turn completes.  The gateway waits *delay* seconds (checking
for user interruption each second), then injects a synthetic continuation
message into the same session.  The agent can keep calling ``self_continue``
each turn to stay in autonomous mode; omitting it ends the loop.

Signal channel
--------------
``request_continuation`` / ``pop_continuation`` / ``cancel_continuation``
operate on a thread-safe module-level dict keyed by session key.  The tool
handler writes; the gateway reads and pops.
"""

import json
import logging
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal channel (tool → gateway)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_continuation_requests: dict[str, dict] = {}  # session_key -> {delay, status}


def request_continuation(session_key: str, delay: int = 45, status: str = ""):
    """Signal that the current session wants another turn after *delay* seconds."""
    with _lock:
        _continuation_requests[session_key] = {"delay": delay, "status": status}


def pop_continuation(session_key: str) -> dict | None:
    """Pop and return the continuation request for *session_key*, if any."""
    with _lock:
        return _continuation_requests.pop(session_key, None)


def cancel_continuation(session_key: str):
    """Cancel any pending continuation request for *session_key*."""
    with _lock:
        _continuation_requests.pop(session_key, None)


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def self_continue(args: dict, **kw) -> str:
    """Tool handler for self_continue."""
    from gateway.session_context import get_session_env

    session_key = get_session_env("HERMES_SESSION_KEY")
    if not session_key:
        return json.dumps({"error": "self_continue is only available in gateway sessions"})

    delay = max(15, min(300, args.get("delay", 45)))
    status = args.get("status", "")

    request_continuation(session_key, delay=delay, status=status)
    logger.info("self_continue requested: session=%s delay=%ds status=%r", session_key, delay, status)

    return json.dumps({"ok": True, "delay": delay, "status": status or "(no status)"})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SELF_CONTINUE_SCHEMA = {
    "name": "self_continue",
    "description": (
        "Continue autonomously after this turn. Call this when you have more "
        "to do and don't need to wait for the user. Your handler can interrupt "
        "at any time by sending a message. If you're done or want to stop, "
        "simply don't call this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "delay": {
                "type": "integer",
                "description": "Seconds to wait before continuing (default 45, min 15, max 300).",
            },
            "status": {
                "type": "string",
                "description": "Brief note on what you plan to do next (shown in Discord).",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="self_continue",
    toolset="session",
    schema=SELF_CONTINUE_SCHEMA,
    handler=self_continue,
    emoji="🔄",
    description="Continue autonomously",
)
