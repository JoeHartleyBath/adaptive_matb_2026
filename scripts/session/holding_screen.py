"""Participant holding screen.

Displays a 'Please wait' message fullscreen on the specified screen index
(default: 1, the participant monitor) between MATB task phases.

Launched as a background subprocess by run_full_study_session.py at the
start of the session and terminated at the end.  OpenMATB's own fullscreen
window covers it during active task phases; it reappears automatically
when MATB exits because a scheduled callback calls window.activate() every
two seconds.

Usage:
    python scripts/session/holding_screen.py [screen_index]

screen_index defaults to 1 (participant monitor).  Must match the
screen_index value set in src/vendor/openmatb/config.ini.
"""
from __future__ import annotations

import sys


def main() -> None:
    screen_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    try:
        import pyglet
    except ImportError:
        print("[holding_screen] pyglet not available — exiting silently.")
        return

    display = pyglet.canvas.get_display()
    screens = display.get_screens()

    if screen_idx >= len(screens):
        print(
            f"[holding_screen] WARNING: screen_index={screen_idx} not found "
            f"({len(screens)} screen(s) detected). Falling back to screen 0."
        )
        screen_idx = 0

    screen = screens[screen_idx]
    window = pyglet.window.Window(
        fullscreen=True,
        screen=screen,
        caption="MATB Session",
    )
    window.set_mouse_visible(False)

    pyglet.gl.glClearColor(0.08, 0.08, 0.08, 1.0)

    label_main = pyglet.text.Label(
        "Please wait",
        font_name="Arial",
        font_size=42,
        bold=True,
        color=(210, 210, 210, 255),
        x=window.width // 2,
        y=window.height // 2 + 30,
        anchor_x="center",
        anchor_y="center",
    )
    label_sub = pyglet.text.Label(
        "Your next task will begin shortly.",
        font_name="Arial",
        font_size=22,
        color=(150, 150, 150, 255),
        x=window.width // 2,
        y=window.height // 2 - 30,
        anchor_x="center",
        anchor_y="center",
    )

    @window.event
    def on_draw():
        window.clear()
        label_main.draw()
        label_sub.draw()

    try:
        pyglet.app.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
