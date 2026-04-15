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

## Known Gaps

### 1. Autonomous wake has no thread or reminders
When the wake-cycle cron fires on its own (not via `/wake`):
- No daily thread is created
- No reminders are scheduled
- Wake state is NOT updated (`awake` stays `False`)
- The cron prompt tells the agent about reminders that won't come

**Impact:** Agent runs with no time awareness, no thread for conversation, and
delivers output to whatever `origin` was last set (possibly stale thread from
previous day, or general channel).

**Fix needed:** Either:
- Add a pre-run script to the wake-cycle job that creates the thread and reminders
- Or move the thread/reminder logic into the cron execution path

### 2. No hard cutoff after 15 minutes
The 15-minute reminder tells the agent to stop, but nothing forces it. If the
agent is mid-game-turn with many tool calls, the reminder arrives as a cron
delivery — it doesn't interrupt the active session.

**Impact:** Agent can run well past 15 minutes if it's in a long tool chain.

**Fix consideration:** The reminder could trigger `/sleep` automatically, or
set a flag that the gateway checks to interrupt the session.

### 3. Wake-cycle `deliver` field
The wake-cycle job has `deliver: "discord"` (from original creation), not
`deliver: "origin"`. This means `_resolve_delivery_target()` may fall back
to `DISCORD_HOME_CHANNEL` instead of using the origin's thread_id.

**Fix:** Update the job's `deliver` field to `"origin"` so it uses the
thread_id that `/wake` sets.

---

## File Reference

| Component | File |
|-----------|------|
| Wake/sleep slash commands | `gateway/platforms/discord.py` |
| Wake state management | `tools/wake_state.py` |
| Cron scheduler (tick, delivery) | `cron/scheduler.py` |
| Cron job CRUD | `cron/jobs.py` |
| Self-continue (autonomous mode) | `tools/continuation.py` |
| Gateway base (session handling) | `gateway/platforms/base.py` |
