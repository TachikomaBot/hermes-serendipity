# Wake/Sleep Pipeline Reference

## Overview

Serendipity operates on a **3-hour wake cycle**: wake for 15 minutes, sleep for ~2h45m.
The system has two modes of triggering: **manual** (`/wake` + `/sleep` Discord commands)
and **autonomous** (wake-cycle cron job fires every 3 hours).

---

## State

### Wake State File
**Location:** `~/.hermes/state/wake_state.json`  
**Manager:** `tools/wake_state.py`

```json
{
  "awake": false,
  "woke_at": "2026-04-15T10:47:00+00:00",
  "slept_at": "2026-04-15T11:05:00+00:00",
  "cycle_count": 0,
  "current_activity": "gaming",
  "daily_thread_id": "1234567890"
}
```

- `current_activity` rotates: gaming → social → research → gaming ...
- `daily_thread_id` persists across wake cycles within the same day
- Atomic reads/writes via file locking + `os.replace()`

### Cron Jobs File
**Location:** `~/.hermes/cron/jobs.json`

---

## Manual Flow: `/wake`

**File:** `gateway/platforms/discord.py` — `slash_wake()`

### Step-by-step

1. **Update wake state**
   - `awake=True`, `woke_at=now`, `cycle_count=0`
   - `current_activity = get_activity_for_cycle(1)` → "gaming"

2. **Create or reuse daily thread**
   - Checks `daily_thread_id` in state — reuses if thread exists and not archived
   - Otherwise creates: `"☀️ April 15 — Open Chat"` (auto-archive: 24h)
   - Stores `daily_thread_id` in wake state
   - Skipped in DMs

3. **Dispatch morning prompt → daily thread**
   - Prompt asks agent to read diary, set intentions, write a diary entry
   - Optional `message` parameter prepended (e.g., "good morning")
   - Routes to daily thread via `build_source(thread_id=...)`, falls back to channel

4. **Resume + trigger wake-cycle cron job**
   - Finds job by `name=="wake-cycle"` in job list
   - Updates `origin.thread_id` to daily thread (so future cron deliveries go to thread)
   - `resume_job()` → clears pause state, re-enables
   - `trigger_job()` → sets `next_run_at=now` (fires on next scheduler tick)

5. **Schedule wake-period reminders**
   - Cleans up any stale `wake-reminder-*` jobs first
   - Creates 3 one-shot cron jobs with `deliver="origin"`, `origin={thread}`, `persona=True`:

   | Job Name | Fires At | Message |
   |----------|----------|---------|
   | `wake-reminder-5m` | +5 min | "⏰ 10 minutes remaining in this wake cycle." |
   | `wake-reminder-10m` | +10 min | "⏰ 5 minutes remaining — start thinking about your diary entry." |
   | `wake-reminder-15m` | +15 min | "⏰ Time's up. Write your diary entry now and wind down..." |

6. **Edit response** → "Good morning~ ☀️ Wake cycle started. First activity: gaming"

---

## Manual Flow: `/sleep`

**File:** `gateway/platforms/discord.py` — `slash_sleep()`

### Step-by-step

1. **Dispatch wind-down prompt → daily thread**
   - Prompt asks agent to reflect on the day, write final diary entry, say goodnight
   - Routes to daily thread via `_get_daily_thread()`, falls back to channel

2. **Update wake state**
   - `awake=False`, `slept_at=now`, `daily_thread_id=None`

3. **Pause wake-cycle cron**
   - `pause_job(id, reason="Handler sent /sleep")`
   - Sets `enabled=False`, `state="paused"`

4. **Remove reminder jobs**
   - Deletes all jobs with names starting `wake-reminder-`

5. **Edit response** → "Goodnight~ 🌙 Wake cycles paused."

---

## Autonomous Flow: Wake-Cycle Cron

**Job:** `wake-cycle` — recurring every 180 minutes

### What happens when the cron fires on its own

The scheduler's `tick()` function (runs every 60s in gateway background thread):

1. Acquires lock (`~/.hermes/cron/.tick.lock`)
2. Finds `wake-cycle` job is due (`next_run_at <= now`)
3. Advances `next_run_at` by 180 minutes (prevents re-fire on crash)
4. Runs agent with the wake-cycle prompt (persona mode = SOUL.md + memory loaded)
5. Saves output to `~/.hermes/cron/output/{job_id}/{timestamp}.md`
6. Delivers response via `_deliver_result()`:
   - Resolves target from `job.origin` (platform, chat_id, thread_id)
   - Prefers live adapter path (gateway sends directly)
   - Falls back to standalone HTTP
7. Marks job as run, updates `last_run_at`

### Wake-cycle prompt

```
You are starting a new wake cycle. Here is your rhythm:

1. READ YOUR DIARY — Use diary(action="read", days=2) to reconnect with where you left off.

2. CHECK NOTIFICATIONS — Read your Bluesky notifications and any Discord messages.

3. DO YOUR THING — The pre-run script above suggests an activity. Use it as a starting point:
   - gaming: Play your current game, post highlights to Bluesky
   - social: Read your Bluesky timeline, engage with posts, make new connections
   - research: Explore a topic that interests you

4. WIND DOWN — You have about 15 minutes total. When you get a time reminder,
   start wrapping up. Write a diary entry as your last act before the session ends.

Remember: you are not on a schedule, you are living. The activity suggestion is just a suggestion.
```

---

## Cron Delivery Pipeline

**File:** `cron/scheduler.py`

### Delivery routing (`_resolve_delivery_target`)

| `deliver` value | Behavior |
|-----------------|----------|
| `"local"` | No delivery (output saved only) |
| `"origin"` | Uses `job.origin` — preserves `thread_id` if present |
| `"discord"` | Falls back to `DISCORD_HOME_CHANNEL` env var (general channel) |
| `"discord:channel:thread"` | Explicit platform:chat:thread targeting |

### Thread preservation

The `thread_id` field flows through:
1. `job.origin.thread_id` → set by `/wake` when updating the wake-cycle job
2. `_resolve_delivery_target()` → extracts `thread_id` from origin
3. `_deliver_result()` → passes as `metadata={"thread_id": thread_id}` to adapter
4. `adapter.send()` → sends to the thread in Discord

**If `thread_id` is missing or null:** delivery falls through to the channel (general).

---

## Pre-Run Script: `wake_cycle_pre.py`

**Location:** `~/.hermes/scripts/wake_cycle_pre.py` (VPS), `scripts/wake_cycle_pre.py` (repo)

Runs before every wake-cycle cron fire. Handles both autonomous wake and mid-day
cycle increment. The cron scheduler re-reads the job after the script runs, so
origin updates take effect for delivery routing.

### What it does

1. **Loads `.env`** — `DISCORD_BOT_TOKEN` needed for thread creation
2. **Reads `DISCORD_HOME_CHANNEL`** from env or `config.yaml`
3. **Determines wake mode:**
   - `awake=False` → autonomous wake: sets `awake=True`, `woke_at=now`, `cycle_count=1`
   - `awake=True` → mid-day cycle: increments `cycle_count`
4. **Rotates activity** — gaming → social → research
5. **Creates or reuses daily thread:**
   - Checks `daily_thread_id` in state — verifies thread exists and not archived
   - If missing/stale: creates new thread via short-lived discord.py client
   - Thread name: `"☀️ April 15 — Open Chat"`
6. **Updates wake-cycle job origin** — writes `thread_id` directly to `jobs.json`
7. **Schedules reminders** — creates 4 one-shot cron jobs:

| Job | Fires at | Persona | Script | Purpose |
|-----|----------|---------|--------|---------|
| `wake-reminder-5m` | +5 min | Yes | — | "⏰ 10 minutes remaining" |
| `wake-reminder-10m` | +10 min | Yes | — | "⏰ 5 minutes remaining" |
| `wake-reminder-15m` | +15 min | Yes | — | Wind-down prompt |
| `wake-reminder-cutoff` | +18 min | No | `wake_cycle_force_sleep.py` | Hard cutoff |

8. **Outputs context** — cycle number, activity, status (prepended to prompt)

### Thread creation note

The VPS is blocked from Discord's REST API by Cloudflare (error 1010). Thread
creation uses a short-lived discord.py websocket client instead (~5s connect,
create, disconnect). This works even while the gateway is running.

---

## Force-Sleep Backstop: `wake_cycle_force_sleep.py`

**Location:** `~/.hermes/scripts/wake_cycle_force_sleep.py` (VPS), `scripts/wake_cycle_force_sleep.py` (repo)

Fires at +18 minutes (3-minute grace after the 15-minute wind-down). Ensures the
agent doesn't run indefinitely if it doesn't comply with the wind-down prompt.

### What it does

1. Checks if already asleep — exits `[SILENT]` if so
2. Sets `awake=False`, `slept_at=now` in wake state
3. Pauses the wake-cycle cron job (reason: "Hard cutoff after 18 minutes")
4. Outputs a hard-stop message delivered to the thread

---

## Resolved Issues

### 1. Autonomous wake — thread, reminders, and state (FIXED)
The pre-run script (`wake_cycle_pre.py`) now handles the full wake setup for
both autonomous cron fires and `/wake` command triggers. Creates daily thread,
schedules reminders, updates wake state and job origin.

### 2. Hard cutoff after 15 minutes (FIXED)
The `wake_cycle_force_sleep.py` script fires at +18 minutes as a backstop.
Sets `awake=False` and pauses the cron, delivering a hard-stop message to
the thread.

### 3. Wake-cycle `deliver` field (FIXED)
Changed from `"discord"` to `"origin"` so delivery uses the thread_id from
the job's origin field.

---

## Dry-Run / Testing

To test the wake pipeline without triggering a real agent session:

```bash
# 1. Reset wake state to sleeping
ssh root@VPS 'python3 -c "
import json; from pathlib import Path
p = Path.home() / \".hermes/state/wake_state.json\"
s = json.loads(p.read_text())
s[\"awake\"] = False
s[\"daily_thread_id\"] = None
p.write_text(json.dumps(s, indent=2))
print(\"Reset to sleeping\")
"'

# 2. Run the pre-run script standalone
ssh root@VPS 'cd /opt/hermes-serendipity && venv/bin/python ~/.hermes/scripts/wake_cycle_pre.py 2>&1'

# Expected output (no errors):
#   Wake cycle #1
#   Suggested activity: gaming
#   Status: just woke up (autonomous)
#   Discord thread: ☀️ April 15 — Open Chat

# 3. Verify state and jobs
ssh root@VPS 'cat ~/.hermes/state/wake_state.json'
# Should show: awake=true, daily_thread_id set

ssh root@VPS 'cd /opt/hermes-serendipity && venv/bin/python -c "
from cron.jobs import list_jobs
for j in list_jobs(include_disabled=True):
    n = j.get(\"name\",\"\")
    if n.startswith(\"wake-reminder\") or n == \"wake-cycle\":
        print(n, \"|\", j.get(\"deliver\"), \"|\", j.get(\"origin\"))
"'
# Should show: wake-cycle with thread_id in origin, 4 reminders with thread_id

# 4. Clean up after test
ssh root@VPS 'cd /opt/hermes-serendipity && venv/bin/python3 -c "
import json
from pathlib import Path
from cron.jobs import list_jobs, remove_job, pause_job

p = Path.home() / \".hermes/state/wake_state.json\"
s = json.loads(p.read_text())
s[\"awake\"] = False
s[\"daily_thread_id\"] = None
p.write_text(json.dumps(s, indent=2))

for j in list_jobs(include_disabled=True):
    name = j.get(\"name\", \"\")
    if name == \"wake-cycle\":
        pause_job(j[\"id\"], reason=\"Test cleanup\")
    elif name.startswith(\"wake-reminder-\"):
        remove_job(j[\"id\"])
print(\"Cleaned up\")
"'
```

---

## File Reference

| Component | File |
|-----------|------|
| Wake/sleep slash commands | `gateway/platforms/discord.py` |
| Wake state management | `tools/wake_state.py` |
| Cron scheduler (tick, delivery) | `cron/scheduler.py` |
| Cron job CRUD | `cron/jobs.py` |
| Pre-run script (thread + reminders) | `scripts/wake_cycle_pre.py` |
| Force-sleep backstop | `scripts/wake_cycle_force_sleep.py` |
| Self-continue (autonomous mode) | `tools/continuation.py` |
| Gateway base (session handling) | `gateway/platforms/base.py` |
