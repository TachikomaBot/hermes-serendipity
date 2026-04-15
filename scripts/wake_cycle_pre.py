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
    """Create a public Discord thread via discord.py client.  Returns thread ID or None.

    Uses a short-lived discord.py client (websocket) because the VPS is
    blocked from Discord's REST API by Cloudflare.  The client connects,
    creates the thread, and disconnects.
    """
    import asyncio
    try:
        import discord
    except ImportError:
        print("(discord.py not installed)", file=sys.stderr)
        return None

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token or not channel_id:
        return None

    result_id = None

    async def _run():
        nonlocal result_id
        intents = discord.Intents.default()
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            nonlocal result_id
            try:
                ch = client.get_channel(int(channel_id))
                if ch is None:
                    ch = await client.fetch_channel(int(channel_id))
                thread = await ch.create_thread(
                    name=name[:100],
                    auto_archive_duration=1440,
                )
                result_id = str(thread.id)
            except Exception as e:
                print(f"(thread creation failed: {e})", file=sys.stderr)
            finally:
                await client.close()

        try:
            await asyncio.wait_for(client.start(token), timeout=30)
        except asyncio.TimeoutError:
            print("(discord client timed out)", file=sys.stderr)
        except Exception:
            pass  # client.close() raises on normal shutdown

    asyncio.run(_run())
    return result_id


def _check_thread_exists(thread_id: str) -> bool:
    """Check if a Discord thread exists and is not archived via discord.py."""
    import asyncio
    try:
        import discord
    except ImportError:
        return False

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token or not thread_id:
        return False

    exists = False

    async def _run():
        nonlocal exists
        intents = discord.Intents.default()
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            nonlocal exists
            try:
                ch = await client.fetch_channel(int(thread_id))
                if isinstance(ch, discord.Thread) and not ch.archived:
                    exists = True
            except Exception:
                pass
            finally:
                await client.close()

        try:
            await asyncio.wait_for(client.start(token), timeout=15)
        except Exception:
            pass

    asyncio.run(_run())
    return exists


def _update_wake_cycle_job_origin(thread_id: str, chat_id: str):
    """Update the wake-cycle job's origin so delivery routes to the thread."""
    try:
        if not JOBS_FILE.exists():
            return
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        # Jobs file is {"jobs": [...]} or bare [...]
        jobs = data.get("jobs", data) if isinstance(data, dict) else data
        for job in jobs:
            if job.get("name") == "wake-cycle":
                if not job.get("origin"):
                    job["origin"] = {}
                job["origin"]["thread_id"] = thread_id
                job["origin"]["chat_id"] = chat_id
                job["origin"]["platform"] = "discord"
                job["deliver"] = "origin"
                break
        JOBS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
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

def _get_home_channel() -> str:
    """Resolve DISCORD_HOME_CHANNEL from env or config.yaml."""
    val = os.getenv("DISCORD_HOME_CHANNEL", "")
    if val:
        return val
    try:
        import yaml
        cfg_path = Path.home() / ".hermes" / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            return str(cfg.get("DISCORD_HOME_CHANNEL", ""))
    except Exception:
        pass
    return ""


def main():
    # Load .env so DISCORD_BOT_TOKEN is available
    try:
        from dotenv import load_dotenv
        env_path = Path("/opt/hermes-serendipity/.env")
        if env_path.exists():
            load_dotenv(str(env_path), override=False)
    except ImportError:
        pass

    state = _load_state()
    home_channel = _get_home_channel()
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
