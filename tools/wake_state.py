"""Wake/sleep state management for Serendipity.

Manages a simple JSON state file that tracks whether the agent is awake,
when she woke/slept, cycle count, and current activity.  Used by the
/wake and /sleep Discord slash commands and the wake-cycle cron pre-run script.
"""

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cross-platform file locking (same pattern as cron/scheduler.py)
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

logger = logging.getLogger("tools.wake_state")

# Activity rotation for wake cycles
_ACTIVITIES = ["gaming", "social", "research"]

_DEFAULT_STATE = {
    "awake": False,
    "woke_at": None,
    "slept_at": None,
    "cycle_count": 0,
    "current_activity": None,
}


def _state_dir() -> Path:
    """Return the state directory (~/.hermes/state/)."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "state"
    except ImportError:
        return Path.home() / ".hermes" / "state"


def _state_file() -> Path:
    return _state_dir() / "wake_state.json"


@contextmanager
def _file_lock(path: Path):
    """Acquire an exclusive file lock for atomic read-modify-write."""
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        elif msvcrt:
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        fd.close()


def get_wake_state() -> Dict[str, Any]:
    """Read and return the current wake state."""
    path = _state_file()
    if not path.exists():
        return dict(_DEFAULT_STATE)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt wake state file, returning defaults")
        return dict(_DEFAULT_STATE)


def set_wake_state(**kwargs) -> Dict[str, Any]:
    """Atomically update specific fields in the wake state file.

    Returns the updated state dict.
    """
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _file_lock(path):
        state = get_wake_state()
        state.update(kwargs)

        # Atomic write via temp + rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".wake_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    return state


def get_activity_for_cycle(cycle_count: int) -> str:
    """Return the suggested activity for a given cycle number."""
    return _ACTIVITIES[(cycle_count - 1) % len(_ACTIVITIES)]


def increment_cycle() -> Dict[str, Any]:
    """Increment the cycle count and rotate the activity. Returns updated state."""
    state = get_wake_state()
    new_count = state.get("cycle_count", 0) + 1
    activity = get_activity_for_cycle(new_count)
    return set_wake_state(cycle_count=new_count, current_activity=activity)
