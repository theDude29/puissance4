"""Puissance 4 — entry point.

This is the "boss": it wires together the two modules and starts a game.
    - minmax.py : the game engine + AI (moved out of this file)
    - cli.py    : the interactive terminal front-end

The AI's thinking time, in seconds per move, can be passed as the first
command-line argument. It deepens its search until that budget runs out, so
the depth it reaches depends on the position rather than being fixed.
"""

import sys

import minmax  # the engine (imported so the boss owns the wiring explicitly)
import cli      # the interactive front-end


def main():
    seconds = 2.0  # long enough to reach ~10 plies from the opening
    if len(sys.argv) > 1:
        try:
            seconds = float(sys.argv[1])
            if seconds <= 0:
                raise ValueError
        except ValueError:
            seconds = 2.0
            print(f"Ignoring invalid time budget '{sys.argv[1]}', using {seconds}s.")

    cli.play(time_limit=seconds)


if __name__ == "__main__":
    main()
