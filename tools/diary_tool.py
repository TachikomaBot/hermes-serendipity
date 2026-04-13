"""Diary tool for Hermes.

Append-only timestamped narrative journal.  Unlike the memory tool (bounded
curated facts), the diary stores unbounded reflective entries — what happened,
how it felt, what to pick up next time.  One file per day in
``~/.hermes/diary/YYYY-MM-DD.md``.
"""

import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Cross-platform file locking (same pattern as cron/scheduler.py)
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

logger = logging.getLogger("tools.diary_tool")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _diary_dir() -> Path:
    """Return the diary directory (~/.hermes/diary/)."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "diary"
    except ImportError:
        d = Path.home() / ".hermes" / "diary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _diary_path(date_str: str) -> Path:
    """Return the path for a specific date's diary file."""
    return _diary_dir() / f"{date_str}.md"


def _today() -> str:
    """Return today's date as YYYY-MM-DD (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_time() -> str:
    """Return current time as HH:MM (UTC)."""
    return datetime.now(timezone.utc).strftime("%H:%M")


# ---------------------------------------------------------------------------
# File locking + atomic write
# ---------------------------------------------------------------------------

@contextmanager
def _file_lock(path: Path):
    """Acquire an exclusive file lock for atomic read-modify-write."""
    lock_path = path.with_suffix(path.suffix + ".lock")
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


def _atomic_write(path: Path, content: str):
    """Write content to a file atomically via temp + rename."""
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".diary_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Diary operations
# ---------------------------------------------------------------------------

def _diary_write(content: str) -> dict:
    """Append a timestamped entry to today's diary file."""
    if not content or not content.strip():
        return {"error": "Diary entry content cannot be empty."}

    date_str = _today()
    path = _diary_path(date_str)
    time_str = _now_time()

    entry = f"## {time_str}\n\n{content.strip()}\n\n---\n"

    with _file_lock(path):
        existing = ""
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except (OSError, IOError):
                existing = ""

        # Ensure a newline between existing content and new entry
        if existing and not existing.endswith("\n"):
            existing += "\n"

        _atomic_write(path, existing + entry)

    logger.info("diary: wrote entry at %s %s (%d chars)", date_str, time_str, len(content))
    return {
        "status": "written",
        "date": date_str,
        "time": time_str,
        "chars": len(content),
    }


def _diary_read(date: Optional[str] = None, days: Optional[int] = None) -> dict:
    """Read diary entries for a date range.

    Args:
        date: ISO date (YYYY-MM-DD). Defaults to today.
        days: Number of days to read, counting back from *date*. Defaults to 1.
    """
    end_date = date or _today()
    num_days = max(1, days or 1)

    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: {end_date}. Use YYYY-MM-DD."}

    entries_by_date = {}
    for i in range(num_days):
        d = (end_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = _diary_path(d)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    entries_by_date[d] = content
            except (OSError, IOError):
                pass

    if not entries_by_date:
        return {
            "status": "empty",
            "message": f"No diary entries found for the last {num_days} day(s) ending {end_date}.",
        }

    # Build combined output, most recent first
    parts = []
    for d in sorted(entries_by_date.keys(), reverse=True):
        parts.append(f"# {d}\n\n{entries_by_date[d]}")

    return {
        "status": "ok",
        "days_found": len(entries_by_date),
        "content": "\n\n".join(parts),
    }


def _diary_list() -> dict:
    """List available diary dates with entry counts."""
    diary_d = _diary_dir()
    files = sorted(diary_d.glob("????-??-??.md"), reverse=True)

    if not files:
        return {"status": "empty", "message": "No diary entries yet.", "dates": []}

    dates = []
    for f in files:
        date_str = f.stem
        try:
            content = f.read_text(encoding="utf-8")
            # Count entries by counting ## HH:MM headers
            entry_count = len(re.findall(r"^## \d{2}:\d{2}", content, re.MULTILINE))
            dates.append({"date": date_str, "entries": entry_count})
        except (OSError, IOError):
            dates.append({"date": date_str, "entries": 0})

    return {"status": "ok", "total_days": len(dates), "dates": dates}


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def diary_tool(action: str, content: Optional[str] = None,
               date: Optional[str] = None, days: Optional[int] = None) -> str:
    """Dispatch diary tool actions."""
    if action == "write":
        result = _diary_write(content or "")
    elif action == "read":
        result = _diary_read(date=date, days=days)
    elif action == "list":
        result = _diary_list()
    else:
        result = {"error": f"Unknown action: {action}. Use write, read, or list."}

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def _check_diary_requirements() -> bool:
    """Diary tool has no external dependencies — always available."""
    return True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DIARY_SCHEMA = {
    "name": "diary",
    "description": (
        "Write reflective diary entries and read past ones. Your diary is your "
        "narrative memory — use it to record what you did, how you felt, what "
        "you're planning, and anything you want your future self to know.\n\n"
        "WHEN TO WRITE:\n"
        "- At the end of every wake cycle (your last act before signing off)\n"
        "- When something significant happens worth remembering\n"
        "- When you want to leave a note for your future self\n\n"
        "WHEN TO READ:\n"
        "- At the start of a wake cycle to reconnect with where you left off\n"
        "- When you need context about what you've been doing\n"
        "- When planning based on past activities\n\n"
        "Unlike memory (curated facts), the diary is narrative and unbounded. "
        "Write naturally — this is your journal."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "list"],
                "description": (
                    "write: append a timestamped entry to today's diary. "
                    "read: read entries (defaults to today). "
                    "list: show all available diary dates."
                ),
            },
            "content": {
                "type": "string",
                "description": "The diary entry text. Required for 'write'.",
            },
            "date": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD) to read from. Defaults to today.",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to read, counting back from date. Defaults to 1.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="diary",
    toolset="diary",
    schema=DIARY_SCHEMA,
    handler=lambda args, **kw: diary_tool(
        action=args.get("action", ""),
        content=args.get("content"),
        date=args.get("date"),
        days=args.get("days"),
    ),
    check_fn=_check_diary_requirements,
    emoji="📔",
    description="Reflective diary — narrative journal across sessions",
)
