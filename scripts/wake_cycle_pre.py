#!/usr/bin/env python3
"""Pre-run script for Serendipity's wake-cycle cron job.

Runs before every wake-cycle cron fire.  Handles both:
  - Autonomous wake (awake=False → sets awake=True, creates daily thread, schedules reminders)
  - Mid-day cycle (awake=True → increments cycle, reuses daily thread, refreshes reminders)

Outputs context that gets prepended to the wake-cycle prompt:
  - Cycle number, suggested activity, time awake
  - Discord thread name (if created/reused)

The cron scheduler re-reads the job after this script runs, so updating
the job's origin.thread_id here ensures delivery goes to the right thread.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "/opt/hermes-serendipity")

STATE_FILE = Path.home() / ".hermes" / "state" / "wake_state.json"
JOBS_FILE = Path.home() / ".hermes" / "cron" / "jobs.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"awake": False, "cycle_count": 0}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _create_discord_thread(channel_id: str, name: str) -> str | None:
    """Create a public Discord thread via REST API.  Returns thread ID or None."""
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token or not channel_id:
        return None
    url = f"https://discord.com/api/v10/channels/{channel_id}/threads"
    data = json.dumps({
        "name": name[:100],
        "type": 11,  # PUBLIC_THREAD
        "auto_archive_duration": 1440,
    }).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            return result.get("id")
    except Exception as e:
        print(f"(thread creation failed: {e})", file=sys.stderr)
        return None


def _check_thread_exists(thread_id: str) -> bool:
    """Check if a Discord thread exists and is not archived."""
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token or not thread_id:
        return False
    url = f"https://discord.com/api/v10/channels/{thread_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bot {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            metadata = data.get("thread_metadata", {})
            return not metadata.get("archived", False)
    except Exception:
        return False


def _update_wake_cycle_job_origin(thread_id: str, chat_id: str):
    """Update the wake-cycle job's origin so delivery routes to the thread."""
    try:
        if not JOBS_FILE.exists():
            return
        jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        for job in jobs:
            if job.get("name") == "wake-cycle":
                if not job.get("origin"):
                    job["origin"] = {}
                job["origin"]["thread_id"] = thread_id
                job["origin"]["chat_id"] = chat_id
                job["origin"]["platform"] = "discord"
                # Ensure deliver is "origin" so thread_id is used
                job["deliver"] = "origin"
                break
        JOBS_FILE.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"(job origin update failed: {e})", file=sys.stderr)


def _schedule_reminders(thread_id: str | None, chat_id: str):
    """Create one-shot reminder cron jobs at +5m, +10m, +15m."""
    try:
        from cron.jobs import list_jobs, remove_job, create_job

        # Clean up stale reminders from previous cycle
        for job in list_jobs(include_disabled=True):
            if job.get("name", "").startswith("wake-reminder-"):
                remove_job(job["id"])

        now = datetime.now(timezone.utc)
        origin = {
            "platform": "discord",
            "chat_id": chat_id,
        }
        if thread_id:
            origin["thread_id"] = thread_id

        reminders = [
            (5,  "wake-reminder-5m",
             "\u23f0 10 minutes remaining in this wake cycle."),
            (10, "wake-reminder-10m",
             "\u23f0 5 minutes remaining \u2014 start thinking about your diary entry."),
            (15, "wake-reminder-15m",
             "\u23f0 Time\u2019s up. Write your diary entry now and wind down. "
             "Wrap up whatever you\u2019re doing \u2014 save your game if applicable, "
             "post any final thoughts, then write a diary entry reflecting on "
             "this cycle. This is your last act before sleep."),
        ]
        for offset_min, name, prompt in reminders:
            fire_at = (now + timedelta(minutes=offset_min)).isoformat()
            create_job(
                prompt=prompt,
                schedule=fire_at,
                name=name,
                deliver="origin",
                origin=origin,
                persona=True,
            )

        # Hard cutoff at +18 minutes (3 min grace after wind-down)
        # This job runs a script that forces awake=False and pauses the cron
        fire_at = (now + timedelta(minutes=18)).isoformat()
        create_job(
            prompt="Force-sleep backstop. If the agent is still running, stop now.",
            schedule=fire_at,
            name="wake-reminder-cutoff",
            deliver="origin",
            origin=origin,
            persona=False,
            script="wake_cycle_force_sleep.py",
        )
    except Exception as e:
        print(f"(reminder setup failed: {e})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    state = _load_state()
    home_channel = os.getenv("DISCORD_HOME_CHANNEL", "")
    now_utc = datetime.now(timezone.utc)

    # --- Determine if this is a fresh wake or a mid-day cycle ---
    was_awake = state.get("awake", False)

    if not was_awake:
        # Autonomous wake: transition from sleep → awake
        state["awake"] = True
        state["woke_at"] = now_utc.isoformat()
        state["cycle_count"] = 1
    else:
        # Mid-day cycle: increment
        state["cycle_count"] = state.get("cycle_count", 0) + 1

    cycle = state["cycle_count"]
    activities = ["gaming", "social", "research"]
    activity = activities[(cycle - 1) % len(activities)]
    state["current_activity"] = activity

    # --- Daily thread: reuse or create ---
    thread_id = state.get("daily_thread_id")
    thread_name = None

    if thread_id and _check_thread_exists(thread_id):
        thread_name = "(reused daily thread)"
    else:
        # Create new daily thread
        today_str = now_utc.strftime("%B %d")
        thread_name = f"\u2600\ufe0f {today_str} \u2014 Open Chat"
        thread_id = _create_discord_thread(home_channel, thread_name)
        if thread_id:
            state["daily_thread_id"] = thread_id
        else:
            thread_name = "(thread creation failed)"

    _save_state(state)

    # --- Update wake-cycle job delivery to target the thread ---
    if thread_id:
        _update_wake_cycle_job_origin(thread_id, home_channel)

    # --- Schedule reminders ---
    _schedule_reminders(thread_id, home_channel)

    # --- Output context for the prompt ---
    print(f"Wake cycle #{cycle}")
    print(f"Suggested activity: {activity}")
    if was_awake:
        print(f"Awake since: {state.get('woke_at', 'unknown')}")
    else:
        print("Status: just woke up (autonomous)")
    if thread_name:
        print(f"Discord thread: {thread_name}")


if __name__ == "__main__":
    main()
