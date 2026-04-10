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
