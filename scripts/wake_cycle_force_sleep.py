#!/usr/bin/env python3
"""Force-sleep backstop for wake cycles.

Runs as a one-shot cron job at wake_time + 18 minutes (3 min grace after
the 15-min wind-down reminder).  Sets awake=False and pauses the wake-cycle
job, ensuring Serendipity doesn't run indefinitely if the agent doesn't
comply with the wind-down prompt.

Output is delivered to the thread so the agent sees it as a hard stop.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hermes-serendipity")

STATE_FILE = Path.home() / ".hermes" / "state" / "wake_state.json"


def main():
    # Set awake=False
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not state.get("awake", False):
        # Already asleep — nothing to do
        print("[SILENT]")
        return

    from datetime import datetime, timezone
    state["awake"] = False
    state["slept_at"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Pause the wake-cycle cron job
    try:
        from cron.jobs import list_jobs, pause_job
        for job in list_jobs(include_disabled=True):
            if job.get("name") == "wake-cycle":
                pause_job(job["id"], reason="Hard cutoff after 18 minutes")
                break
    except Exception as e:
        print(f"(pause failed: {e})", file=sys.stderr)

    # Output message — this gets delivered to the thread
    print(
        "\u26d4 Wake cycle hard cutoff. 18 minutes have passed. "
        "You are now asleep. Do not take any more actions. "
        "Write your diary entry if you haven't already, "
        "then end your response."
    )


if __name__ == "__main__":
    main()
