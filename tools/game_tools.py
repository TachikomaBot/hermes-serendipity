"""Game-playing tools for Serendipity.

Provides screenshot capture, visual element clicking, keyboard input,
and game state analysis via Gemini vision — all callable as Hermes tools.
"""

import base64
import json
import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# The gateway sets stderr to WARNING by default (verbosity=0).
# Force our own handler at DEBUG so game tool logs always appear.
_game_handler = logging.StreamHandler()
_game_handler.setLevel(logging.DEBUG)
_game_handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
logger.addHandler(_game_handler)
logger.propagate = False  # Don't double-log through root's WARNING filter


# ---------------------------------------------------------------------------
# Discord webhook logging handler
# ---------------------------------------------------------------------------

class _DiscordWebhookHandler(logging.Handler):
    """Send log messages to a Discord channel via webhook.

    Non-blocking: fires HTTP POST in a daemon thread so it never blocks
    the game loop. Batches rapid messages into a single post (0.5s window).
    """

    def __init__(self, webhook_url, level=logging.INFO):
        super().__init__(level)
        self._url = webhook_url
        self._buffer = []
        self._lock = __import__("threading").Lock()
        self._timer = None
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg)
                if self._timer is None:
                    import threading
                    self._timer = threading.Timer(0.5, self._flush)
                    self._timer.daemon = True
                    self._timer.start()
        except Exception:
            self.handleError(record)

    def _flush(self):
        with self._lock:
            lines = self._buffer[:]
            self._buffer.clear()
            self._timer = None
        if not lines:
            return
        # Discord message limit is 2000 chars; truncate if needed
        content = "```\n" + "\n".join(lines)
        if len(content) > 1990:
            content = content[:1987] + "..."
        content += "\n```"
        try:
            import urllib.request
            data = json.dumps({"content": content}).encode()
            req = urllib.request.Request(
                self._url, data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Serendipity/1.0",
                },
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            import sys
            print(f"[DiscordWebhook] flush failed: {exc}", file=sys.stderr)


_DISCORD_LOG_WEBHOOK = os.environ.get("GAME_TOOLS_LOG_WEBHOOK", "")
if _DISCORD_LOG_WEBHOOK:
    _discord_handler = _DiscordWebhookHandler(_DISCORD_LOG_WEBHOOK, level=logging.INFO)
    logger.addHandler(_discord_handler)
    logger.info("Discord webhook logging enabled")


DISPLAY = os.environ.get("DISPLAY", ":99")

# ---------------------------------------------------------------------------
# game_turn constants
# ---------------------------------------------------------------------------
FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-3.1-pro-preview"
DEFAULT_DOWNSCALE = (960, 540)
MAX_OBSERVATIONS = 6          # screenshot-analyze cycles per game_turn call
MAX_ACTIONS_HARD_CAP = 30     # total actions (clicks+keys) per call
WALL_CLOCK_TIMEOUT = 120      # seconds
STRATEGIC_REVIEW_INTERVAL = 10  # Pro review every N turns

# xdotool key name normalization — Flash often generates lowercase
_KEY_NORMALIZE = {
    "escape": "Escape", "return": "Return", "enter": "Return",
    "space": "space", "tab": "Tab", "backspace": "BackSpace",
    "delete": "Delete", "home": "Home", "end": "End",
    "pageup": "Page_Up", "pagedown": "Page_Down",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "shift+return": "Shift+Return", "shift+enter": "Shift+Return",
}


def _normalize_key(key):
    """Normalize key names to xdotool format."""
    return _KEY_NORMALIZE.get(key.lower().strip(), key)

# ---------------------------------------------------------------------------
# Structured output schemas for Gemini
# ---------------------------------------------------------------------------

_STATE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string", "enum": [
            "main_map", "city_view", "dialog", "unit_selection", "menu", "other",
        ]},
        "turn_info": {"type": "string"},
        "selected_unit": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "at": {"type": "string"},
            },
            "nullable": True,
        },
        "units_needing_orders": {
            "type": "array", "items": {"type": "string"},
        },
        "unit_count_on_tile": {"type": "integer", "nullable": True},
        "units_visible_in_panel": {"type": "integer"},
        "cities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "production": {"type": "string"},
                    "turns_left": {"type": "integer"},
                },
            },
        },
        "threats": {"type": "array", "items": {"type": "string"}},
        "resources": {
            "type": "object",
            "properties": {
                "gold": {"type": "integer"},
                "researching": {"type": "string"},
            },
        },
        "notifications": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "surprise_level": {"type": "string", "enum": ["none", "low", "high"]},
        "mismatches": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["screen_type", "turn_info", "mismatches"],
}

_ACTION_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": [
                        "click", "key", "end_turn",
                    ]},
                    "target": {"type": "string"},
                    "key": {"type": "string"},
                },
                "required": ["action"],
            },
        },
        "turn_complete": {"type": "boolean"},
        "intent": {"type": "string"},
    },
    "required": ["actions", "turn_complete"],
}

_LOCATE_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "y0": {"type": "integer", "nullable": True},
        "x0": {"type": "integer", "nullable": True},
        "y1": {"type": "integer", "nullable": True},
        "x1": {"type": "integer", "nullable": True},
        "confidence": {"type": "number"},
        "description": {"type": "string"},
    },
    "required": ["x0", "y0", "x1", "y1", "confidence", "description"],
}

ESCALATION_KEYWORDS = [
    "enemy", "barbarian", "attack", "war", "declare",
    "disaster", "famine", "revolt", "unknown civilization",
    "wonder", "unexpected", "surprise", "urgent",
]


def _check_game_requirements() -> bool:
    """Check if game tools can run (Xvfb, xdotool, imagemagick)."""
    try:
        for cmd in ["xdotool", "import", "identify"]:
            subprocess.run(
                ["which", cmd], capture_output=True, check=True,
            )
        return True
    except Exception:
        return False


def _focus_window(window_name=None):
    """Focus a window by name, or no-op if None (use whatever is active)."""
    if window_name is None:
        return True
    import time
    env = {**os.environ, "DISPLAY": DISPLAY}
    for name in (window_name.split("|") if "|" in window_name else [window_name]):
        result = subprocess.run(
            ["xdotool", "search", "--name", name],
            capture_output=True, text=True, env=env,
        )
        wids = result.stdout.strip().split()
        if wids:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", wids[-1]],
                env=env,
            )
            time.sleep(0.2)
            return True
    return False


def _focus_game_window():
    """Focus the active game window before sending input."""
    return _focus_window("Freeciv|Unciv")


def _capture_and_downscale(target_w=None, target_h=None):
    """Capture screenshot, return (b64, full_w, full_h).

    Sends full resolution (1920x1080) to give vision models maximum detail
    for UI element identification. The name is kept for backward compat.
    """
    path_full = tempfile.mktemp(suffix=".png")
    try:
        subprocess.run(
            ["import", "-display", DISPLAY, "-window", "root", path_full],
            check=True,
        )
        result = subprocess.run(
            ["identify", "-format", "%w %h", path_full],
            capture_output=True, text=True, check=True,
        )
        full_w, full_h = map(int, result.stdout.strip().split())

        with open(path_full, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        return img_b64, full_w, full_h
    finally:
        try:
            os.unlink(path_full)
        except OSError:
            pass


def _capture_and_crop(region, target_w=None, target_h=None):
    """Capture screenshot, crop to region (0-1000 normalized), resize for vision.

    Returns (b64, full_w, full_h, crop_offset_dict).
    crop_offset has px_x0, px_y0, px_x1, px_y1 for coordinate remapping.
    """
    tw, th = target_w or DEFAULT_DOWNSCALE[0], target_h or DEFAULT_DOWNSCALE[1]
    path_full = tempfile.mktemp(suffix=".png")
    path_crop = tempfile.mktemp(suffix="_crop.png")
    try:
        subprocess.run(
            ["import", "-display", DISPLAY, "-window", "root", path_full],
            check=True,
        )
        result = subprocess.run(
            ["identify", "-format", "%w %h", path_full],
            capture_output=True, text=True, check=True,
        )
        full_w, full_h = map(int, result.stdout.strip().split())

        # Convert normalized 0-1000 to pixels
        px0 = int(region["x0"] / 1000 * full_w)
        py0 = int(region["y0"] / 1000 * full_h)
        px1 = int(region["x1"] / 1000 * full_w)
        py1 = int(region["y1"] / 1000 * full_h)
        crop_w = max(px1 - px0, 1)
        crop_h = max(py1 - py0, 1)

        subprocess.run(
            ["convert", path_full, "-crop", f"{crop_w}x{crop_h}+{px0}+{py0}",
             "+repage", "-resize", f"{tw}x{th}", path_crop],
            check=True,
        )
        with open(path_crop, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        return img_b64, full_w, full_h, {
            "px_x0": px0, "px_y0": py0, "px_x1": px1, "px_y1": py1,
        }
    finally:
        for p in (path_full, path_crop):
            try:
                os.unlink(p)
            except OSError:
                pass


def _gemini_call(model, prompt, image_b64, system=None, response_schema=None):
    """Call Gemini with an image + prompt. Returns response text. Retries once.

    If response_schema is provided, enables structured output (JSON mode)
    so the response is guaranteed valid JSON matching the schema.
    """
    import time as _time
    from google import genai
    client = genai.Client()
    config = {
        "temperature": 0.0,
        "http_options": {"api_version": "v1alpha"},
    }
    if system:
        config["system_instruction"] = system
    if response_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = response_schema
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
            {"text": prompt},
        ],
    }]
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            return response.text
        except Exception as e:
            if attempt == 0:
                logger.warning("Gemini call failed (%s), retrying: %s", model, e)
                _time.sleep(2)
            else:
                raise


def _parse_json_response(text):
    """Strip markdown code fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _should_escalate(game_state_text, turn_number, force_review,
                     already_escalated, consecutive_failures):
    """Decide whether to escalate to Pro model for strategic review."""
    if already_escalated:
        return False
    if force_review:
        return True
    if turn_number <= 1:
        return True
    if turn_number > 0 and turn_number % STRATEGIC_REVIEW_INTERVAL == 0:
        return True
    if consecutive_failures >= 2:
        return True
    state_lower = game_state_text.lower() if isinstance(game_state_text, str) else ""
    if any(kw in state_lower for kw in ESCALATION_KEYWORDS):
        return True
    # Escalate on expectation mismatches
    if '"mismatches": [' in game_state_text:
        try:
            state = json.loads(game_state_text) if isinstance(game_state_text, str) else {}
            if state.get("mismatches"):
                return True
        except (json.JSONDecodeError, ValueError):
            if "mismatch" in state_lower:
                return True
    return False


_LOCATE_SYSTEM_PROMPT = (
    "You identify UI elements in screenshots and return bounding box "
    "coordinates on a NORMALIZED 0-1000 scale. Respond with ONLY a JSON "
    "object: {\"y0\": N, \"x0\": N, \"y1\": N, \"x1\": N, \"confidence\": "
    "F, \"description\": \"...\"}\n"
    "If target not visible: {\"y0\": null, \"x0\": null, \"y1\": null, "
    "\"x1\": null, \"confidence\": 0, \"description\": \"...\"}"
)

# Bounding box area threshold: if Flash's first-pass bounding box is
# smaller than this fraction of the screen, do a zoom pass for precision.
_ZOOM_THRESHOLD = 0.04  # 4% of screen area → ~250x170px at 1920x1080


def _precision_click(target, full_w, full_h, focus_fn):
    """Locate a UI element and click it, using two-pass zoom for small targets.

    Pass 1: Full screenshot → Flash locates target → bounding box.
    If bounding box is small (< _ZOOM_THRESHOLD of screen), do Pass 2:
      Crop region around target → zoom to 960x540 → Flash re-locates →
      remap coordinates → click.
    Otherwise: click directly from Pass 1 coordinates.

    Returns result dict with status, target, pixel, confidence.
    """
    import time as _time

    # ── PASS 1: full screenshot ──
    img_b64, _, _ = _capture_and_downscale()
    resp = _gemini_call(FLASH_MODEL, f"Locate: {target}", img_b64,
                        system=_LOCATE_SYSTEM_PROMPT,
                        response_schema=_LOCATE_ELEMENT_SCHEMA)
    coords = _parse_json_response(resp)

    if coords.get("x0") is None:
        return {"status": "not_found", "target": target,
                "description": coords.get("description", "")}
    confidence = coords.get("confidence", 0)
    if confidence < 0.5:
        return {"status": "low_confidence", "target": target,
                "confidence": confidence}

    x0, y0 = coords["x0"], coords["y0"]
    x1, y1 = coords["x1"], coords["y1"]
    box_area = (x1 - x0) * (y1 - y0) / 1_000_000  # fraction of screen

    # ── PASS 2: zoom if target is small ──
    if box_area < _ZOOM_THRESHOLD:
        # Expand bounding box by 50% padding for context, clamped to 0-1000
        pad_w = max((x1 - x0) * 0.5, 50)
        pad_h = max((y1 - y0) * 0.5, 50)
        region = {
            "x0": max(0, x0 - pad_w),
            "y0": max(0, y0 - pad_h),
            "x1": min(1000, x1 + pad_w),
            "y1": min(1000, y1 + pad_h),
        }
        try:
            zoomed_b64, _, _, crop_off = _capture_and_crop(region)
            zoom_resp = _gemini_call(
                FLASH_MODEL, f"Locate: {target}", zoomed_b64,
                system=_LOCATE_SYSTEM_PROMPT,
                response_schema=_LOCATE_ELEMENT_SCHEMA,
            )
            zoom_coords = _parse_json_response(zoom_resp)

            if (zoom_coords.get("x0") is not None
                    and zoom_coords.get("confidence", 0) >= 0.5):
                # Remap from crop-space (0-1000) to full-screen pixels
                crop_w = crop_off["px_x1"] - crop_off["px_x0"]
                crop_h = crop_off["px_y1"] - crop_off["px_y0"]
                zx = int((zoom_coords["x0"] + zoom_coords["x1"]) / 2
                         / 1000 * crop_w + crop_off["px_x0"])
                zy = int((zoom_coords["y0"] + zoom_coords["y1"]) / 2
                         / 1000 * crop_h + crop_off["px_y0"])

                focus_fn()
                env = {**os.environ, "DISPLAY": DISPLAY}
                subprocess.run(
                    ["xdotool", "mousemove", "--screen", "0",
                     str(zx), str(zy), "click", "1"],
                    env=env, check=True,
                )
                _time.sleep(0.3)
                return {"status": "clicked", "target": target,
                        "pixel": [zx, zy],
                        "confidence": zoom_coords.get("confidence", 0),
                        "zoom_pass": True}
        except Exception:
            logger.warning("precision_click: zoom pass failed, falling back",
                           exc_info=True)

    # ── Direct click from Pass 1 ──
    cx = int((x0 + x1) / 2 / 1000 * full_w)
    cy = int((y0 + y1) / 2 / 1000 * full_h)

    focus_fn()
    env = {**os.environ, "DISPLAY": DISPLAY}
    subprocess.run(
        ["xdotool", "mousemove", "--screen", "0", str(cx), str(cy), "click", "1"],
        env=env, check=True,
    )
    _time.sleep(0.3)
    return {"status": "clicked", "target": target, "pixel": [cx, cy],
            "confidence": confidence}


def _execute_action(action, full_w, full_h):
    """Execute a single game action (click, key, or end_turn). Returns result dict."""
    import time as _time
    act_type = action.get("action", "")

    if act_type == "end_turn":
        _focus_game_window()
        env = {**os.environ, "DISPLAY": DISPLAY}
        subprocess.run(["xdotool", "key", "Shift+Return"], env=env, check=True)
        _time.sleep(0.5)
        return {"status": "pressed", "key": "Shift+Return"}

    if act_type == "key":
        key = _normalize_key(action.get("key", ""))
        if not key:
            return {"status": "error", "error": "no key specified"}
        _focus_game_window()
        env = {**os.environ, "DISPLAY": DISPLAY}
        subprocess.run(["xdotool", "key", key], env=env, check=True)
        _time.sleep(0.3)
        return {"status": "pressed", "key": key}

    if act_type == "click":
        target = action.get("target", "")
        if not target:
            return {"status": "error", "error": "no click target"}
        try:
            return _precision_click(target, full_w, full_h, _focus_game_window)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    return {"status": "error", "error": f"unknown action type: {act_type}"}


def _execute_ui_action(action, full_w, full_h, window_name=None):
    """Execute a UI action (click or key) in any window. Returns result dict."""
    import time as _time
    act_type = action.get("action", "")

    if act_type == "key":
        key = _normalize_key(action.get("key", ""))
        if not key:
            return {"status": "error", "error": "no key specified"}
        _focus_window(window_name)
        env = {**os.environ, "DISPLAY": DISPLAY}
        subprocess.run(["xdotool", "key", key], env=env, check=True)
        _time.sleep(0.3)
        return {"status": "pressed", "key": key}

    if act_type == "click":
        target = action.get("target", "")
        if not target:
            return {"status": "error", "error": "no click target"}
        try:
            return _precision_click(
                target, full_w, full_h,
                lambda: _focus_window(window_name),
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

    return {"status": "error", "error": f"unknown action type: {act_type}"}


# ---------------------------------------------------------------------------
# game_turn prompts
# ---------------------------------------------------------------------------

_STATE_ANALYSIS_PROMPT = """\
Analyze this game screenshot. Return ONLY a JSON object:
{{
  "screen_type": "main_map|city_view|dialog|unit_selection|menu|other",
  "turn_info": "Turn N, Year",
  "selected_unit": {{"type": "Settler|Warrior|...", "at": "(x,y)"}} or null,
  "units_needing_orders": ["unit_type at (x,y)"],
  "unit_count_on_tile": null,
  "units_visible_in_panel": 0,
  "cities": [{{"name": "...", "production": "...", "turns_left": 0}}],
  "threats": [],
  "resources": {{"gold": 0, "researching": "Tech (N turns)"}},
  "notifications": [],
  "observations": [],
  "surprise_level": "none|low|high",
  "mismatches": []
}}

IMPORTANT — Mismatch detection:
- Check if a number on a map tile (e.g. "4") exceeds visible units in a panel/dialog
- Check if the selected unit type matches what the strategy requires
- Check if expected UI elements are missing or if dialogs need scrolling
- Report ANY discrepancy between what you see and what you'd expect in "mismatches"
  e.g. ["tile shows 4 units but only 3 visible in selection dialog",
        "strategy says found city but selected unit is Diplomat not Settler"]

STRATEGY CONTEXT: {strategy}
Previous context: {prev_summary}
Recent actions: {recent_actions}"""

_ACTION_DECISION_PROMPT = """\
You are executing a turn in a strategy game. Pick 1-5 actions.

STRATEGY: {strategy}
{guidance_line}
GAME STATE: {game_state}
RECENT ACTIONS: {recent_actions}
ACTION BUDGET: {remaining} remaining
{game_knowledge_line}

Return JSON:
{{
  "actions": [
    {{"action": "click", "target": "description of element"}},
    {{"action": "key", "key": "xdotool key name"}},
    {{"action": "end_turn"}}
  ],
  "turn_complete": false,
  "intent": "brief description of what this action sequence is trying to accomplish"
}}

Rules:
- Max 5 actions per batch
- Handle dialogs/popups FIRST (Escape or click Close)
- VERIFY the correct unit is selected before issuing unit commands
- If state reports mismatches, resolve them FIRST (e.g. scroll to find hidden \
units, select the correct unit before acting)
- Actions often require multi-step sequences. Think in terms of goals:
  e.g. "found city" = select Settler → verify Settler is active → press B
  e.g. "move unit" = select unit → press G (or movement key) → click destination
- If a list or panel might have hidden items (count mismatch), try scrolling
- Set turn_complete: true when all units have orders and turn should end"""

_STRATEGIC_REVIEW_PROMPT = """\
You are a strategic advisor for a turn-based strategy game.

PLAYER STRATEGY: {strategy}
CURRENT GAME STATE: {game_state}
PREVIOUS CONTEXT: {prev_summary}

Assess in under 150 words:
1. TACTICAL: Immediate threats or opportunities the current strategy misses?
2. STRATEGIC: Does the grand strategy (e.g. victory condition path) still \
make sense given what's happening on the ground? If the player is pursuing \
a science victory but keeps winning wars, should they consider pivoting \
to domination? If going military but falling behind in tech, reconsider?
3. MISMATCHES: If the game state reports mismatches (expected vs actual), \
diagnose the likely cause and suggest specific actions to resolve them. \
Common issues: hidden scrollbars in selection dialogs, wrong unit selected, \
unit stacks where clicking a tile selects the wrong unit, UI elements that \
require scrolling to reveal all options.
4. RECOMMENDATIONS: Enumerate priorities for the next 5-10 turns.

Your assessment will be returned to the player's high-level reasoning, \
which may revise the grand strategy for future turns. Be direct."""


def _build_summary(action_log, final_state_text, turn_number, escalated,
                   strategic_guidance):
    """Build a compact summary of the turn for the Hermes agent."""
    # Count outcomes
    clicks = sum(1 for a in action_log if a.get("action") == "click")
    keys = sum(1 for a in action_log if a.get("action") == "key")
    errors = sum(1 for a in action_log
                 if a.get("result", {}).get("status") not in ("clicked", "pressed"))

    # Collect action descriptions
    descriptions = []
    for a in action_log:
        if a.get("action") == "click" and a.get("target"):
            status = a.get("result", {}).get("status", "?")
            if status == "clicked":
                descriptions.append(f"clicked {a['target']}")
            else:
                descriptions.append(f"tried to click {a['target']} ({status})")
        elif a.get("action") == "key" and a.get("key"):
            descriptions.append(f"pressed {a['key']}")
        elif a.get("action") == "end_turn":
            descriptions.append("ended turn")

    parts = [f"Turn {turn_number} complete."]
    if descriptions:
        parts.append("Actions: " + "; ".join(descriptions[:8]) + ".")

    # Extract observations from final state
    if isinstance(final_state_text, str):
        try:
            state = json.loads(final_state_text)
            res = state.get("resources", {})
            if res:
                res_parts = []
                if "gold" in res:
                    res_parts.append(f"Gold: {res['gold']}")
                if "researching" in res:
                    res_parts.append(f"Research: {res['researching']}")
                if res_parts:
                    parts.append(". ".join(res_parts) + ".")
            threats = state.get("threats", [])
            if threats:
                parts.append("THREATS: " + "; ".join(threats) + ".")
            else:
                parts.append("No threats.")
            observations = state.get("observations", [])
            if observations:
                parts.append("Observed: " + "; ".join(observations[:5]) + ".")
            mismatches = state.get("mismatches", [])
            if mismatches:
                parts.append("MISMATCHES: " + "; ".join(mismatches[:3]) + ".")
        except (json.JSONDecodeError, AttributeError):
            pass

    if errors:
        parts.append(f"({errors} action(s) failed.)")

    if escalated and strategic_guidance:
        # Truncate to keep summary compact
        guidance_short = strategic_guidance[:300]
        if len(strategic_guidance) > 300:
            guidance_short += "..."
        parts.append(f"STRATEGIC REVIEW: {guidance_short}")

    # Suggest exploration if many actions failed
    if errors > 0 and clicks > 0 and errors / max(clicks, 1) > 0.5:
        parts.append(
            "HINT: Many actions failed. Consider using menu_navigate in explore "
            "mode to map available game actions before continuing."
        )

    return " ".join(parts)


def game_turn(args: dict, **kwargs) -> str:
    """Execute a complete game turn using an OODA loop.

    Observes game state, orients via Flash (escalates to Pro on surprise),
    decides on actions, executes in batches, and returns a compact summary.
    """
    import time as _time

    strategy = args.get("strategy", "")
    if not strategy:
        return json.dumps({"status": "error", "error": "strategy is required"})
    turn_number = args.get("turn_number", 0)
    force_review = args.get("force_strategic_review", False)
    max_actions = min(args.get("max_actions", 15), MAX_ACTIONS_HARD_CAP)
    prev_summary = args.get("previous_summary", "")
    game_knowledge = args.get("game_knowledge", "")

    action_log = []
    total_actions = 0
    observations = 0
    escalated = False
    strategic_guidance = ""
    consecutive_failures = 0
    start_time = _time.monotonic()
    final_state_text = ""

    try:
        while (total_actions < max_actions
               and observations < MAX_OBSERVATIONS
               and _time.monotonic() - start_time < WALL_CLOCK_TIMEOUT):

            observations += 1

            # ── OBSERVE ──
            img_b64, full_w, full_h = _capture_and_downscale()

            # ── ORIENT ── (Flash analyzes state)
            state_prompt = _STATE_ANALYSIS_PROMPT.format(
                strategy=strategy,
                prev_summary=prev_summary or "(none)",
                recent_actions=json.dumps(
                    [{"action": a.get("action"), "target": a.get("target"),
                      "key": a.get("key"), "result": a.get("result", {}).get("status")}
                     for a in action_log[-5:]]),
            )
            final_state_text = _gemini_call(FLASH_MODEL, state_prompt, img_b64,
                                           response_schema=_STATE_ANALYSIS_SCHEMA)
            logger.info("game_turn obs %d: %s", observations, final_state_text[:200])

            # ── ESCALATE? ──
            if _should_escalate(final_state_text, turn_number, force_review,
                                escalated, consecutive_failures):
                review_prompt = _STRATEGIC_REVIEW_PROMPT.format(
                    strategy=strategy,
                    game_state=final_state_text[:2000],
                    prev_summary=prev_summary or "(none)",
                )
                strategic_guidance = _gemini_call(PRO_MODEL, review_prompt, img_b64)
                escalated = True
                logger.info("game_turn Pro review: %s", strategic_guidance[:200])

            # ── DECIDE ── (Flash picks actions)
            guidance_line = (f"STRATEGIC GUIDANCE: {strategic_guidance}"
                            if strategic_guidance else "")
            game_knowledge_line = (
                f"GAME KNOWLEDGE:\n{game_knowledge}" if game_knowledge else ""
            )
            decide_prompt = _ACTION_DECISION_PROMPT.format(
                strategy=strategy,
                guidance_line=guidance_line,
                game_state=final_state_text[:2000],
                recent_actions=json.dumps(
                    [{"action": a.get("action"), "target": a.get("target"),
                      "key": a.get("key"), "status": a.get("result", {}).get("status")}
                     for a in action_log[-3:]]),
                remaining=max_actions - total_actions,
                game_knowledge_line=game_knowledge_line,
            )
            decide_text = _gemini_call(FLASH_MODEL, decide_prompt, img_b64,
                                       response_schema=_ACTION_DECISION_SCHEMA)
            try:
                plan = _parse_json_response(decide_text)
            except (json.JSONDecodeError, ValueError):
                logger.warning("game_turn: could not parse action plan: %s",
                               decide_text[:200])
                break

            actions = plan.get("actions", [])
            turn_complete = plan.get("turn_complete", False)

            if not actions or turn_complete:
                break

            # ── ACT ── (execute batch)
            batch_failures = 0
            for action in actions:
                result = _execute_action(action, full_w, full_h)
                action_log.append({**action, "result": result})
                total_actions += 1
                if result.get("status") not in ("clicked", "pressed"):
                    batch_failures += 1
                if total_actions >= max_actions:
                    break
                _time.sleep(0.3)

            if batch_failures == len(actions):
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            if any(a.get("action") == "end_turn" for a in actions):
                break

        summary = _build_summary(action_log, final_state_text, turn_number,
                                 escalated, strategic_guidance)
        return json.dumps({
            "status": "ok",
            "summary": summary,
            "actions_taken": total_actions,
            "observations": observations,
            "escalated_to_pro": escalated,
            "strategic_signal": strategic_guidance if escalated else None,
        })
    except Exception as e:
        logger.exception("game_turn error")
        partial = _build_summary(action_log, final_state_text, turn_number,
                                 escalated, strategic_guidance)
        return json.dumps({
            "status": "error",
            "error": str(e),
            "partial_summary": partial,
            "actions_taken": total_actions,
        })


# ---------------------------------------------------------------------------
# menu_navigate prompts
# ---------------------------------------------------------------------------

_MENU_ANALYSIS_PROMPT = """\
Analyze this UI screenshot. Return ONLY a JSON object:
{{
  "screen_description": "brief description of current screen/dialog",
  "screen_id": "short_stable_id e.g. main_menu, game_config, nation_picker",
  "ui_elements": [
    {{"label": "...", "type": "button|dropdown|tab|field|checkbox|menu_item",
      "state": "enabled|disabled|selected|checked"}}
  ],
  "goal_reached": false,
  "decision_needed": null,
  "region_of_interest": null,
  "stuck": false
}}

If there's a choice to make (dropdown, radio buttons, picker, config option)
where preferences don't specify what to pick, set decision_needed to:
{{"description": "what choice", "options": ["opt1", "opt2"], "default": "pre-selected"}}

region_of_interest: {{"x0": N, "y0": N, "x1": N, "y1": N}} normalized 0-1000
for the area most relevant to the current goal. Useful if we get stuck.

{mode_context}

Previous actions: {recent_actions}
Step {step} of {max_steps}"""

_MENU_ACTION_PROMPT = """\
You are navigating a UI to reach this goal: {goal}
User preferences: {preferences}
{guidance_line}

Current screen: {screen_analysis}
Previous actions: {recent_actions}

Pick exactly 1 action (or 2 if the second is an obvious follow-up like pressing \
Enter after typing).

Return JSON:
{{
  "actions": [
    {{"action": "click", "target": "description of element to click"}}
  ],
  "reasoning": "brief explanation"
}}

Rules:
- Prefer CLICKING buttons/menu items over keyboard shortcuts
- After clicking a menu item, STOP and wait for verification
- If a dialog is blocking, dismiss it first (click OK/Close or press Escape)
- If you need to type, use {{"action": "key", "key": "xdotool key sequence"}}"""

_MENU_EXPLORE_ACTION_PROMPT = """\
You are exploring an unfamiliar application interface to learn what it can do.

CURRENT SCREEN: {screen_analysis}
ALREADY EXPLORED: {explored_screens}
UNEXPLORED ELEMENTS: {unexplored}
DEPTH: {depth}/{max_depth}
PRO GUIDANCE: {guidance}

Pick 1 action to learn more about this interface. Prefer:
1. Unexplored menu items, buttons, or tabs that lead to new screens
2. Dropdowns (click to see options, then Escape to close)
3. Skip elements that are decorative or unlikely to reveal new information

Return JSON:
{{
  "action": "click|key",
  "target": "description of element",
  "expect": "dialog|dropdown|new_screen|toggle",
  "back_action": "Escape|click Close|click Back",
  "done_exploring_screen": false
}}

Set done_exploring_screen to true if all interesting elements on this screen \
have been explored or cataloged."""

_MENU_DECISION_PROMPT = """\
You are helping navigate a UI. A choice needs to be made.

GOAL: {goal}
USER PREFERENCES: {preferences}
CURRENT SCREEN: {screen_description}

DECISION NEEDED: {decision_description}
AVAILABLE OPTIONS: {options}
DEFAULT/PRE-SELECTED: {default}

Pick the best option considering:
1. User preferences (if they specified anything relevant)
2. Reasonable defaults for the goal
3. Common/recommended settings

Return JSON:
{{"choice": "the option to select", "reasoning": "brief explanation"}}"""

_MENU_EXPLORE_PRIORITY_PROMPT = """\
You are helping explore an unfamiliar application interface.

APPLICATION CONTEXT: {context}
SCREENS MAPPED SO FAR: {interface_map_summary}
CURRENT SCREEN ELEMENTS: {ui_elements}
EXPLORATION BUDGET: {steps_remaining} steps left

Which elements are most worth exploring? Consider:
- What would a user need to know to operate this application?
- Which elements likely lead to new screens vs. simple toggles?
- What's the most efficient exploration order?

Return JSON:
{{
  "priority_elements": ["element1", "element2"],
  "skip_elements": ["element3"],
  "observations": "any high-level observations about the application",
  "enough": false
}}

Set "enough" to true if the interface map is already comprehensive enough \
to write a useful guide."""

_MENU_STUCK_PROMPT = """\
Flash vision model is stuck navigating a UI. Same screen {repeat_count} times.

GOAL: {goal}
CURRENT SCREEN: {screen_description}
RECENT ACTIONS (all failed to make progress): {recent_actions}

Examine the screenshot carefully. What should we try?
Return JSON:
{{
  "diagnosis": "what's likely going wrong",
  "actions": [
    {{"action": "click", "target": "very specific element description"}}
  ],
  "give_up": false,
  "give_up_reason": null
}}

Set give_up to true only if the goal is genuinely unreachable from this screen."""


# ---------------------------------------------------------------------------
# menu_navigate constants
# ---------------------------------------------------------------------------

MENU_NAV_TIMEOUT_EXPLOIT = 90
MENU_NAV_TIMEOUT_EXPLORE = 120
MENU_NAV_MAX_STEPS_CAP = 40


# ---------------------------------------------------------------------------
# menu_navigate handler
# ---------------------------------------------------------------------------

def _menu_exploit_loop(goal, preferences, max_steps, window_name):
    """Goal-directed UI navigation loop. Returns result dict."""
    import time as _time

    action_log = []
    step = 0
    prev_screen_id = None
    same_screen_count = 0
    pro_decisions = []
    start_time = _time.monotonic()

    while step < max_steps and (_time.monotonic() - start_time) < MENU_NAV_TIMEOUT_EXPLOIT:
        step += 1

        # ── CAPTURE & ANALYZE ──
        img_b64, full_w, full_h = _capture_and_downscale()

        mode_context = f"GOAL: {goal}\nPREFERENCES: {preferences or '(none)'}"
        analysis_prompt = _MENU_ANALYSIS_PROMPT.format(
            mode_context=mode_context,
            recent_actions=json.dumps([
                {"action": a.get("action"), "target": a.get("target"),
                 "key": a.get("key"), "status": a.get("result", {}).get("status")}
                for a in action_log[-5:]
            ]),
            step=step, max_steps=max_steps,
        )
        analysis_text = _gemini_call(FLASH_MODEL, analysis_prompt, img_b64)

        try:
            analysis = _parse_json_response(analysis_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("menu_navigate: bad analysis JSON: %s", analysis_text[:200])
            analysis = {"screen_description": analysis_text[:200],
                        "screen_id": "unknown", "goal_reached": False}

        logger.info("menu_navigate step %d: screen=%s", step,
                     analysis.get("screen_id", "?"))

        # ── GOAL REACHED? ──
        if analysis.get("goal_reached"):
            return {
                "status": "goal_reached",
                "summary": f"Reached goal in {step} steps: "
                           f"{analysis.get('screen_description', '')}",
                "steps_taken": step,
                "actions": len(action_log),
                "pro_decisions": len(pro_decisions),
            }

        # ── STUCK DETECTION ──
        screen_id = analysis.get("screen_id", "")
        if screen_id == prev_screen_id:
            same_screen_count += 1
        else:
            same_screen_count = 0
            prev_screen_id = screen_id

        if same_screen_count >= 3:
            # Phase 1: Crop and zoom
            roi = analysis.get("region_of_interest")
            if roi and same_screen_count == 3:
                try:
                    cropped_b64, _, _, crop_off = _capture_and_crop(roi)
                    zoom_prompt = _MENU_ACTION_PROMPT.format(
                        goal=goal, preferences=preferences or "(none)",
                        guidance_line="NOTE: This is a ZOOMED view of the area of interest.",
                        screen_analysis=analysis_text[:1000],
                        recent_actions="(stuck — retrying with zoom)",
                    )
                    zoom_text = _gemini_call(FLASH_MODEL, zoom_prompt, cropped_b64)
                    zoom_plan = _parse_json_response(zoom_text)
                    for action in zoom_plan.get("actions", [])[:1]:
                        result = _execute_ui_action(action, full_w, full_h, window_name)
                        action_log.append({**action, "result": result, "zoomed": True})
                    _time.sleep(0.4)
                    continue
                except Exception:
                    logger.warning("menu_navigate: zoom retry failed", exc_info=True)

            # Phase 2: Escalate to Pro
            if same_screen_count >= 4:
                try:
                    stuck_prompt = _MENU_STUCK_PROMPT.format(
                        goal=goal,
                        screen_description=analysis.get("screen_description", ""),
                        recent_actions=json.dumps([
                            {"action": a.get("action"), "target": a.get("target"),
                             "status": a.get("result", {}).get("status")}
                            for a in action_log[-5:]
                        ]),
                        repeat_count=same_screen_count,
                    )
                    pro_text = _gemini_call(PRO_MODEL, stuck_prompt, img_b64)
                    pro_plan = _parse_json_response(pro_text)
                    if pro_plan.get("give_up"):
                        return {
                            "status": "stuck",
                            "summary": f"Unable to reach goal after {step} steps. "
                                       f"Diagnosis: {pro_plan.get('give_up_reason', 'unknown')}",
                            "steps_taken": step,
                            "actions": len(action_log),
                        }
                    for action in pro_plan.get("actions", [])[:2]:
                        result = _execute_ui_action(action, full_w, full_h, window_name)
                        action_log.append({**action, "result": result, "pro_directed": True})
                    same_screen_count = 0
                    _time.sleep(0.4)
                    continue
                except Exception:
                    logger.warning("menu_navigate: Pro stuck resolution failed",
                                   exc_info=True)

            # Phase 3: Unrecoverable
            if same_screen_count >= 6:
                return {
                    "status": "stuck",
                    "summary": f"Stuck after {step} steps on: "
                               f"{analysis.get('screen_description', '')}",
                    "steps_taken": step,
                    "actions": len(action_log),
                }

        # ── DECISION POINT? ──
        guidance_line = ""
        decision = analysis.get("decision_needed")
        if decision and isinstance(decision, dict):
            try:
                decision_prompt = _MENU_DECISION_PROMPT.format(
                    goal=goal,
                    preferences=preferences or "(none specified)",
                    screen_description=analysis.get("screen_description", ""),
                    decision_description=decision.get("description", ""),
                    options=json.dumps(decision.get("options", [])),
                    default=decision.get("default", "unknown"),
                )
                pro_text = _gemini_call(PRO_MODEL, decision_prompt, img_b64)
                pro_decision = _parse_json_response(pro_text)
                pro_decisions.append(pro_decision)
                guidance_line = (
                    f"PRO DECISION: Select '{pro_decision.get('choice', '')}' "
                    f"— {pro_decision.get('reasoning', '')}"
                )
            except Exception:
                logger.warning("menu_navigate: Pro decision failed", exc_info=True)

        # ── DECIDE ACTION (Flash) ──
        action_prompt = _MENU_ACTION_PROMPT.format(
            goal=goal,
            preferences=preferences or "(none)",
            guidance_line=guidance_line,
            screen_analysis=analysis_text[:1000],
            recent_actions=json.dumps([
                {"action": a.get("action"), "target": a.get("target"),
                 "status": a.get("result", {}).get("status")}
                for a in action_log[-3:]
            ]),
        )
        action_text = _gemini_call(FLASH_MODEL, action_prompt, img_b64)

        try:
            plan = _parse_json_response(action_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("menu_navigate: bad action JSON: %s", action_text[:200])
            continue

        actions = plan.get("actions", [])[:2]
        if not actions:
            continue

        # ── EXECUTE ──
        for action in actions:
            result = _execute_ui_action(action, full_w, full_h, window_name)
            action_log.append({**action, "result": result})
            _time.sleep(0.4)

    return {
        "status": "max_steps_reached",
        "summary": f"Exhausted {step} steps without reaching goal: {goal}",
        "steps_taken": step,
        "actions": len(action_log),
        "pro_decisions": len(pro_decisions),
    }


def _menu_explore_loop(context, max_steps, max_depth, window_name):
    """Exploration loop — map an unfamiliar interface. Returns result dict."""
    import time as _time

    interface_map = {}        # screen_id → {description, elements, parent}
    visited_screens = set()
    explore_stack = []        # [(screen_id, back_action)]
    action_log = []
    step = 0
    current_depth = 0
    pro_guidance = ""
    pro_observations = ""
    start_time = _time.monotonic()

    while step < max_steps and (_time.monotonic() - start_time) < MENU_NAV_TIMEOUT_EXPLORE:
        step += 1

        # ── CAPTURE & ANALYZE ──
        img_b64, full_w, full_h = _capture_and_downscale()

        mode_context = (
            "MODE: exploration. Catalog ALL visible UI elements thoroughly. "
            "Include element types, states, and any visible options/values.\n"
            f"APPLICATION CONTEXT: {context}"
        )
        analysis_prompt = _MENU_ANALYSIS_PROMPT.format(
            mode_context=mode_context,
            recent_actions=json.dumps([
                {"action": a.get("action"), "target": a.get("target"),
                 "status": a.get("result", {}).get("status")}
                for a in action_log[-5:]
            ]),
            step=step, max_steps=max_steps,
        )
        analysis_text = _gemini_call(FLASH_MODEL, analysis_prompt, img_b64)

        try:
            analysis = _parse_json_response(analysis_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("menu_explore: bad analysis JSON: %s", analysis_text[:200])
            analysis = {"screen_description": analysis_text[:200],
                        "screen_id": f"unknown_{step}", "ui_elements": []}

        screen_id = analysis.get("screen_id", f"screen_{step}")
        logger.info("menu_explore step %d: screen=%s depth=%d",
                     step, screen_id, current_depth)

        # ── RECORD SCREEN ──
        if screen_id not in interface_map:
            parent = explore_stack[-1][0] if explore_stack else None
            interface_map[screen_id] = {
                "description": analysis.get("screen_description", ""),
                "elements": analysis.get("ui_elements", []),
                "parent": parent,
            }

        # ── PRO PRIORITIZATION (every 3 new screens) ──
        if len(interface_map) % 3 == 0 and screen_id not in visited_screens:
            try:
                map_summary = {sid: {"desc": info["description"],
                                     "n_elements": len(info["elements"])}
                               for sid, info in interface_map.items()}
                priority_prompt = _MENU_EXPLORE_PRIORITY_PROMPT.format(
                    context=context,
                    interface_map_summary=json.dumps(map_summary),
                    ui_elements=json.dumps(analysis.get("ui_elements", [])),
                    steps_remaining=max_steps - step,
                )
                pro_text = _gemini_call(PRO_MODEL, priority_prompt, img_b64)
                pro_priority = _parse_json_response(pro_text)
                pro_guidance = json.dumps(pro_priority.get("priority_elements", []))
                if pro_priority.get("observations"):
                    pro_observations = pro_priority["observations"]
                if pro_priority.get("enough"):
                    logger.info("menu_explore: Pro says enough")
                    break
            except Exception:
                logger.warning("menu_explore: Pro priority failed", exc_info=True)

        # ── CHECK IF SCREEN FULLY EXPLORED ──
        if screen_id in visited_screens:
            # Back out
            if explore_stack:
                _, back_action = explore_stack.pop()
                current_depth = max(0, current_depth - 1)
                if back_action:
                    back = {"action": "key", "key": back_action} if back_action in (
                        "Escape", "Return", "BackSpace"
                    ) else {"action": "click", "target": back_action}
                    result = _execute_ui_action(back, full_w, full_h, window_name)
                    action_log.append({**back, "result": result, "back_nav": True})
                    _time.sleep(0.4)
                continue
            else:
                break  # Nothing left to explore

        # ── DEPTH CHECK ──
        if current_depth >= max_depth:
            visited_screens.add(screen_id)
            if explore_stack:
                _, back_action = explore_stack.pop()
                current_depth = max(0, current_depth - 1)
                if back_action:
                    back = {"action": "key", "key": back_action} if back_action in (
                        "Escape", "Return", "BackSpace"
                    ) else {"action": "click", "target": back_action}
                    result = _execute_ui_action(back, full_w, full_h, window_name)
                    action_log.append({**back, "result": result, "back_nav": True})
                    _time.sleep(0.4)
                continue
            else:
                break

        # ── PICK NEXT ELEMENT TO EXPLORE (Flash) ──
        explored_labels = [e.get("label", "") for e in
                           interface_map.get(screen_id, {}).get("elements", [])
                           if e.get("_explored")]
        all_labels = [e.get("label", "") for e in analysis.get("ui_elements", [])]
        unexplored = [l for l in all_labels if l and l not in explored_labels]

        if not unexplored:
            visited_screens.add(screen_id)
            continue

        explore_prompt = _MENU_EXPLORE_ACTION_PROMPT.format(
            screen_analysis=analysis_text[:1000],
            explored_screens=json.dumps(list(visited_screens)),
            unexplored=json.dumps(unexplored),
            depth=current_depth, max_depth=max_depth,
            guidance=pro_guidance or "(none)",
        )
        explore_text = _gemini_call(FLASH_MODEL, explore_prompt, img_b64)

        try:
            explore_plan = _parse_json_response(explore_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("menu_explore: bad explore JSON: %s", explore_text[:200])
            visited_screens.add(screen_id)
            continue

        if explore_plan.get("done_exploring_screen"):
            visited_screens.add(screen_id)
            continue

        # ── EXECUTE EXPLORATION ACTION ──
        action = {
            "action": explore_plan.get("action", "click"),
            "target": explore_plan.get("target", ""),
            "key": explore_plan.get("key", explore_plan.get("target", "")),
        }
        back_action = explore_plan.get("back_action", "Escape")
        expected = explore_plan.get("expect", "new_screen")

        result = _execute_ui_action(action, full_w, full_h, window_name)
        action_log.append({**action, "result": result})
        _time.sleep(0.5)

        # Mark element as explored in the map
        target_label = explore_plan.get("target", "")
        for elem in interface_map.get(screen_id, {}).get("elements", []):
            if elem.get("label", "") == target_label:
                elem["_explored"] = True
                break

        # If we expect a new screen, push onto stack
        if expected in ("new_screen", "dialog"):
            explore_stack.append((screen_id, back_action))
            current_depth += 1
        elif expected == "dropdown":
            # Catalog dropdown options on next iteration, then close
            explore_stack.append((screen_id, back_action))
            # Don't increment depth — dropdown is same logical screen

    # ── BUILD NAVIGATION HINTS ──
    nav_hints = []
    for sid, info in interface_map.items():
        parent = info.get("parent")
        if parent:
            # Find which element led here
            for elem in interface_map.get(parent, {}).get("elements", []):
                if elem.get("_explored"):
                    nav_hints.append(f"{parent} → {elem.get('label', '?')} → {sid}")
                    break

    # Clean internal markers from output
    clean_map = {}
    for sid, info in interface_map.items():
        clean_elements = []
        for elem in info.get("elements", []):
            clean_elem = {k: v for k, v in elem.items() if not k.startswith("_")}
            clean_elements.append(clean_elem)
        clean_map[sid] = {
            "description": info.get("description", ""),
            "elements": clean_elements,
            "parent": info.get("parent"),
        }

    return {
        "status": "explored",
        "application": context,
        "screens_visited": len(visited_screens),
        "interface_map": clean_map,
        "navigation_hints": nav_hints,
        "pro_observations": pro_observations,
        "steps_taken": step,
        "actions": len(action_log),
    }


def menu_navigate(args: dict, **kwargs) -> str:
    """Navigate application menus or explore an unfamiliar interface.

    Two modes:
    - exploit (default): Goal-directed navigation to a target screen/state.
    - explore: Systematically map an unfamiliar interface for learning.
    """
    mode = args.get("mode", "exploit")

    if mode == "explore":
        context = args.get("context", "")
        if not context:
            return json.dumps({"status": "error",
                               "error": "context is required for explore mode"})
        max_steps = min(args.get("max_steps", 25), MENU_NAV_MAX_STEPS_CAP)
        max_depth = args.get("max_depth", 3)
        window_name = args.get("window_name")
        try:
            result = _menu_explore_loop(context, max_steps, max_depth, window_name)
            return json.dumps(result)
        except Exception as e:
            logger.exception("menu_navigate explore error")
            return json.dumps({"status": "error", "error": str(e)})

    # Exploit mode (default)
    goal = args.get("goal", "")
    if not goal:
        return json.dumps({"status": "error",
                           "error": "goal is required for exploit mode"})
    preferences = args.get("preferences", "")
    max_steps = min(args.get("max_steps", 20), MENU_NAV_MAX_STEPS_CAP)
    window_name = args.get("window_name")
    try:
        result = _menu_exploit_loop(goal, preferences, max_steps, window_name)
        return json.dumps(result)
    except Exception as e:
        logger.exception("menu_navigate exploit error")
        return json.dumps({"status": "error", "error": str(e)})


def game_screenshot(args: dict, **kwargs) -> str:
    """Capture a screenshot of the game and return a description of the game state.

    Sends the screenshot to Gemini for analysis and returns a structured
    text description of what's visible on screen.
    """
    import time
    try:
        path = tempfile.mktemp(suffix=".png")
        subprocess.run(
            ["import", "-display", DISPLAY, "-window", "root", path],
            check=True,
        )
        result = subprocess.run(
            ["identify", "-format", "%w %h", path],
            capture_output=True, text=True, check=True,
        )
        w, h = result.stdout.strip().split()

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        os.unlink(path)

        from google import genai
        client = genai.Client()

        context = args.get("context", "")
        prompt = (
            "Analyze this game screenshot. Describe:\n"
            "1. What game is this and what screen/state are we on?\n"
            "2. What units, cities, or UI elements are visible?\n"
            "3. What resources, stats, or turn info can you read?\n"
            "4. What actions are available right now?\n"
            "Be specific and concise."
        )
        if context:
            prompt += f"\n\nAdditional context: {context}"

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": prompt},
                ],
            }],
            config={
                "temperature": 0.0,
                "http_options": {"api_version": "v1alpha"},
            },
        )
        return json.dumps({
            "status": "ok",
            "resolution": f"{w}x{h}",
            "analysis": response.text,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def game_click(args: dict, **kwargs) -> str:
    """Click on a game UI element identified by visual description.

    Takes a screenshot, asks Gemini to locate the target element,
    converts normalized coordinates to pixels, and clicks.
    """
    import time
    target = args.get("target", "")
    if not target:
        return json.dumps({"status": "error", "error": "target is required"})

    try:
        # Capture screenshot
        path = tempfile.mktemp(suffix=".png")
        subprocess.run(
            ["import", "-display", DISPLAY, "-window", "root", path],
            check=True,
        )
        result = subprocess.run(
            ["identify", "-format", "%w %h", path],
            capture_output=True, text=True, check=True,
        )
        w, h = map(int, result.stdout.strip().split())

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        os.unlink(path)

        # Ask Gemini for coordinates
        from google import genai
        client = genai.Client()

        system = (
            "You identify UI elements in screenshots and return bounding box "
            "coordinates on a NORMALIZED 0-1000 scale. Respond with ONLY a JSON "
            "object: {\"y0\": N, \"x0\": N, \"y1\": N, \"x1\": N, \"confidence\": "
            "F, \"description\": \"...\"}\n"
            "If target not visible: {\"y0\": null, \"x0\": null, \"y1\": null, "
            "\"x1\": null, \"confidence\": 0, \"description\": \"...\"}"
        )

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": f"Locate: {target}"},
                ],
            }],
            config={
                "system_instruction": system,
                "temperature": 0.0,
                "http_options": {"api_version": "v1alpha"},
            },
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        coords = json.loads(text)

        if coords.get("x0") is None:
            return json.dumps({
                "status": "not_found",
                "target": target,
                "description": coords.get("description", ""),
            })

        confidence = coords.get("confidence", 0)
        if confidence < 0.5:
            return json.dumps({
                "status": "low_confidence",
                "target": target,
                "confidence": confidence,
                "description": coords.get("description", ""),
            })

        # Convert normalized to pixel coords
        cx_norm = (coords["x0"] + coords["x1"]) // 2
        cy_norm = (coords["y0"] + coords["y1"]) // 2
        cx = int(cx_norm / 1000 * w)
        cy = int(cy_norm / 1000 * h)

        # Focus and click
        _focus_game_window()
        env = {**os.environ, "DISPLAY": DISPLAY}
        subprocess.run(
            ["xdotool", "mousemove", "--screen", "0", str(cx), str(cy), "click", "1"],
            env=env, check=True,
        )
        time.sleep(0.3)

        return json.dumps({
            "status": "clicked",
            "target": target,
            "pixel": [cx, cy],
            "confidence": confidence,
            "description": coords.get("description", ""),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def game_key(args: dict, **kwargs) -> str:
    """Press a keyboard key in the game window.

    Uses xdotool key names (e.g. "b" for build city, "Return" for enter,
    "space" for wait, "x" for auto-explore).
    """
    import time
    key = _normalize_key(args.get("key", ""))
    if not key:
        return json.dumps({"status": "error", "error": "key is required"})

    try:
        _focus_game_window()
        env = {**os.environ, "DISPLAY": DISPLAY}
        subprocess.run(
            ["xdotool", "key", key],
            env=env, check=True,
        )
        time.sleep(0.3)
        return json.dumps({"status": "pressed", "key": key})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

GAME_SCREENSHOT_SCHEMA = {
    "name": "game_screenshot",
    "description": (
        "Capture a screenshot of the game and get an AI analysis of the current "
        "game state including units, cities, terrain, resources, and available actions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": "Optional context to help analyze the screenshot (e.g. 'I just founded a city')",
            },
        },
        "required": [],
    },
}

GAME_CLICK_SCHEMA = {
    "name": "game_click",
    "description": (
        "Click on a game UI element by visual description. Takes a screenshot, "
        "uses vision to locate the element, and clicks it. "
        "Examples: 'the Start New Game button', 'my settler unit', "
        "'the production menu', 'the End Turn button'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Description of what to click (e.g. 'the End Turn button')",
            },
        },
        "required": ["target"],
    },
}

GAME_KEY_SCHEMA = {
    "name": "game_key",
    "description": (
        "Press a keyboard key in the game window. Uses xdotool key names. "
        "Common game keys: 'b' = build/found city (FreeCiv), 'space' = wait, "
        "'x' = auto-explore, 'Return' = end turn/confirm, 'Escape' = cancel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key to press (xdotool name, e.g. 'b', 'Return', 'space')",
            },
        },
        "required": ["key"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="game_screenshot",
    toolset="gaming",
    schema=GAME_SCREENSHOT_SCHEMA,
    handler=game_screenshot,
    check_fn=_check_game_requirements,
    emoji="📸",
    description="Capture and analyze a game screenshot",
)

registry.register(
    name="game_click",
    toolset="gaming",
    schema=GAME_CLICK_SCHEMA,
    handler=game_click,
    check_fn=_check_game_requirements,
    emoji="🖱️",
    description="Click a game UI element by visual description",
)

registry.register(
    name="game_key",
    toolset="gaming",
    schema=GAME_KEY_SCHEMA,
    handler=game_key,
    check_fn=_check_game_requirements,
    emoji="⌨️",
    description="Press a keyboard key in the game",
)

GAME_TURN_SCHEMA = {
    "name": "game_turn",
    "description": (
        "Execute a complete game turn using an OODA loop. Handles all "
        "screenshot capture, game state analysis, and action execution "
        "internally. Returns a compact summary. Pass your strategic "
        "goals and the tool handles mechanical execution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "description": (
                    "High-level strategic goals. Include both grand strategy "
                    "(e.g. 'pursuing science victory, prioritize campus districts') "
                    "and immediate turn goals (e.g. 'send settler to (15,8), "
                    "build warrior in Beijing'). Also include any observation "
                    "tasks (e.g. 'scout NW coast for second city location')."
                ),
            },
            "turn_number": {
                "type": "integer",
                "description": "Current turn number (triggers periodic strategic review)",
            },
            "force_strategic_review": {
                "type": "boolean",
                "description": "Force a Pro model strategic assessment this turn",
            },
            "max_actions": {
                "type": "integer",
                "description": "Max actions this turn (default 15, cap 30)",
            },
            "previous_summary": {
                "type": "string",
                "description": "Summary from previous game_turn call, for continuity",
            },
            "game_knowledge": {
                "type": "string",
                "description": (
                    "Game-specific UI knowledge to help Flash make correct actions. "
                    "Paste relevant skill content here — e.g. keybinds, how to select "
                    "units, how menus work, multi-step action sequences. This is "
                    "injected directly into Flash's decision prompt."
                ),
            },
        },
        "required": ["strategy"],
    },
}

registry.register(
    name="game_turn",
    toolset="gaming",
    schema=GAME_TURN_SCHEMA,
    handler=game_turn,
    check_fn=_check_game_requirements,
    emoji="🎮",
    description="Execute a complete game turn via OODA loop",
)

MENU_NAVIGATE_SCHEMA = {
    "name": "menu_navigate",
    "description": (
        "Navigate application menus and dialogs, or explore an unfamiliar interface. "
        "Two modes: 'exploit' (default) navigates to a goal; 'explore' maps the "
        "interface for learning. Works with any application via vision + clicking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["exploit", "explore"],
                "description": (
                    "'exploit' (default): navigate to a specific goal. "
                    "'explore': systematically map an unfamiliar interface. "
                    "Use explore when you have no memory/skill for this application."
                ),
            },
            "goal": {
                "type": "string",
                "description": "Target screen/state to reach (required for exploit mode)",
            },
            "preferences": {
                "type": "string",
                "description": "Preferences for choices encountered (exploit mode)",
            },
            "context": {
                "type": "string",
                "description": (
                    "What app and why you're exploring (explore mode). "
                    "E.g. 'just launched FreeCiv for the first time'"
                ),
            },
            "max_steps": {
                "type": "integer",
                "description": "Max screenshot-action cycles (default 20 exploit, 25 explore; cap 40)",
            },
            "max_depth": {
                "type": "integer",
                "description": "How many screens deep to explore (explore mode, default 3)",
            },
            "window_name": {
                "type": "string",
                "description": "Window name for xdotool focus (pipe-separated alternatives)",
            },
        },
        "required": [],
    },
}

registry.register(
    name="menu_navigate",
    toolset="gaming",
    schema=MENU_NAVIGATE_SCHEMA,
    handler=menu_navigate,
    check_fn=_check_game_requirements,
    emoji="🧭",
    description="Navigate or explore application menus via vision",
)
