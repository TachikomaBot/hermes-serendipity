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


def _focus_game_window():
    """Focus the active game window before sending input."""
    import time
    env = {**os.environ, "DISPLAY": DISPLAY}
    # Try known game windows
    for name in ["Freeciv", "Unciv"]:
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


def _capture_and_downscale(target_w=None, target_h=None):
    """Capture screenshot at full res, downscale for vision, return (b64, full_w, full_h)."""
    tw, th = target_w or DEFAULT_DOWNSCALE[0], target_h or DEFAULT_DOWNSCALE[1]
    path_full = tempfile.mktemp(suffix=".png")
    path_small = tempfile.mktemp(suffix="_small.png")
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

        subprocess.run(
            ["convert", path_full, "-resize", f"{tw}x{th}", path_small],
            check=True,
        )
        with open(path_small, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        return img_b64, full_w, full_h
    finally:
        for p in (path_full, path_small):
            try:
                os.unlink(p)
            except OSError:
                pass


def _gemini_call(model, prompt, image_b64, system=None):
    """Call Gemini with an image + prompt. Returns response text. Retries once."""
    import time as _time
    from google import genai
    client = genai.Client()
    config = {
        "temperature": 0.0,
        "http_options": {"api_version": "v1alpha"},
    }
    if system:
        config["system_instruction"] = system
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
    return False


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
        key = action.get("key", "")
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
            # Capture fresh screenshot for coordinate lookup
            img_b64, _, _ = _capture_and_downscale()
            system = (
                "You identify UI elements in screenshots and return bounding box "
                "coordinates on a NORMALIZED 0-1000 scale. Respond with ONLY a JSON "
                "object: {\"y0\": N, \"x0\": N, \"y1\": N, \"x1\": N, \"confidence\": "
                "F, \"description\": \"...\"}\n"
                "If target not visible: {\"y0\": null, \"x0\": null, \"y1\": null, "
                "\"x1\": null, \"confidence\": 0, \"description\": \"...\"}"
            )
            resp = _gemini_call(FLASH_MODEL, f"Locate: {target}", img_b64, system=system)
            coords = _parse_json_response(resp)

            if coords.get("x0") is None:
                return {"status": "not_found", "target": target,
                        "description": coords.get("description", "")}
            confidence = coords.get("confidence", 0)
            if confidence < 0.5:
                return {"status": "low_confidence", "target": target,
                        "confidence": confidence}

            cx = int((coords["x0"] + coords["x1"]) / 2 / 1000 * full_w)
            cy = int((coords["y0"] + coords["y1"]) / 2 / 1000 * full_h)

            _focus_game_window()
            env = {**os.environ, "DISPLAY": DISPLAY}
            subprocess.run(
                ["xdotool", "mousemove", "--screen", "0", str(cx), str(cy), "click", "1"],
                env=env, check=True,
            )
            import time as _time
            _time.sleep(0.3)
            return {"status": "clicked", "target": target, "pixel": [cx, cy],
                    "confidence": confidence}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    return {"status": "error", "error": f"unknown action type: {act_type}"}


# ---------------------------------------------------------------------------
# game_turn prompts
# ---------------------------------------------------------------------------

_STATE_ANALYSIS_PROMPT = """\
Analyze this game screenshot. Return ONLY a JSON object:
{{
  "screen_type": "main_map|city_view|dialog|menu|other",
  "turn_info": "Turn N, Year",
  "units_needing_orders": ["unit at (x,y)"],
  "cities": [{{"name": "...", "production": "...", "turns_left": 0}}],
  "threats": [],
  "resources": {{"gold": 0, "researching": "Tech (N turns)"}},
  "notifications": [],
  "observations": [],
  "surprise_level": "none|low|high"
}}

Previous context: {prev_summary}
Recent actions: {recent_actions}"""

_ACTION_DECISION_PROMPT = """\
You are executing a turn in a strategy game. Pick 3-5 actions.

STRATEGY: {strategy}
{guidance_line}
GAME STATE: {game_state}
RECENT ACTIONS: {recent_actions}
ACTION BUDGET: {remaining} remaining

Return JSON:
{{
  "actions": [
    {{"action": "click", "target": "description of element"}},
    {{"action": "key", "key": "xdotool key name"}},
    {{"action": "end_turn"}}
  ],
  "turn_complete": false
}}

Rules:
- Max 5 actions per batch
- Handle dialogs/popups FIRST (Escape or click Close)
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
3. RECOMMENDATIONS: Enumerate priorities for the next 5-10 turns.

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
                prev_summary=prev_summary or "(none)",
                recent_actions=json.dumps(
                    [{"action": a.get("action"), "target": a.get("target"),
                      "key": a.get("key"), "result": a.get("result", {}).get("status")}
                     for a in action_log[-5:]]),
            )
            final_state_text = _gemini_call(FLASH_MODEL, state_prompt, img_b64)
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
            decide_prompt = _ACTION_DECISION_PROMPT.format(
                strategy=strategy,
                guidance_line=guidance_line,
                game_state=final_state_text[:2000],
                recent_actions=json.dumps(
                    [{"action": a.get("action"), "target": a.get("target"),
                      "key": a.get("key"), "status": a.get("result", {}).get("status")}
                     for a in action_log[-3:]]),
                remaining=max_actions - total_actions,
            )
            decide_text = _gemini_call(FLASH_MODEL, decide_prompt, img_b64)
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
    key = args.get("key", "")
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
