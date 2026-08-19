"""Puissance 4 — entry point.

This is the "boss": it wires together the two modules and starts a game.
    - minmax.py : the game engine + AI (moved out of this file)
    - cli.py    : the interactive terminal front-end

The AI's search depth can be passed as the first command-line argument.
"""

import sys

import minmax  # the engine (imported so the boss owns the wiring explicitly)
import cli      # the interactive front-end


def main():
    depth = 9  # a sensible default look-ahead: about half a second per move
    if len(sys.argv) > 1:
        try:
            depth = int(sys.argv[1])
        except ValueError:
            print(f"Ignoring invalid depth '{sys.argv[1]}', using {depth}.")

    cli.play(depth=depth)


if __name__ == "__main__":
    main()
