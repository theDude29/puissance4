"""Interactive command-line front-end for Puissance 4.

The human plays first as X (+1); the minmax AI answers as O (-1). All game
logic lives in `minmax.py` — this module only handles rendering the board and
reading/validating the human's input.
"""

from minmax import WIDTH, HEIGHT, winning_player, is_final, get_next_move, minmax

HUMAN = 1
AI = -1

SYMBOLS = {0: ".", HUMAN: "X", AI: "O"}


def render(board):
    """Print the board with a 1..WIDTH column header (row 0 at the top)."""
    print()
    print(" " + " ".join(str(c + 1) for c in range(WIDTH)))
    for row in range(HEIGHT):
        cells = (SYMBOLS[board[row * WIDTH + col]] for col in range(WIDTH))
        print(" " + " ".join(cells))
    print()


def winner(board):
    """Return HUMAN, AI, or None depending on who (if anyone) has 4-in-a-row."""
    w = winning_player(board)
    return w if w != 0 else None


def ask_human_move(board):
    """Prompt until the human enters a legal, non-full column; return the
    resulting board."""
    legal = {col: child for col, child in get_next_move(board, HUMAN)}

    while True:
        raw = input(f"Your move — pick a column (1-{WIDTH}, q to quit): ").strip()

        if raw.lower() in ("q", "quit", "exit"):
            print("Bye!")
            raise SystemExit(0)

        if not raw.isdigit():
            print("  Please enter a number.")
            continue

        col = int(raw) - 1  # display is 1-based, internal is 0-based
        if col not in legal:
            if 0 <= col < WIDTH:
                print("  That column is full — choose another.")
            else:
                print(f"  Out of range — pick between 1 and {WIDTH}.")
            continue

        return legal[col]


def announce(board):
    """Print the final result. Returns True if the game is over."""
    who = winner(board)
    if who == HUMAN:
        render(board)
        print("You win! 🎉")
        return True
    if who == AI:
        render(board)
        print("The AI wins. Better luck next time!")
        return True
    if is_final(board):
        render(board)
        print("It's a draw.")
        return True
    return False


def play(depth=4):
    """Run one interactive game. The human (X) moves first."""
    board = [0] * (HEIGHT * WIDTH)

    print("Puissance 4 — you are X, the AI is O. You go first.")
    render(board)

    while True:
        # --- human turn ---
        board = ask_human_move(board)
        if announce(board):
            return

        # --- AI turn ---
        print("AI is thinking...")
        col, _ = minmax(AI, board, depth)
        # apply the AI's chosen column
        for c, child in get_next_move(board, AI):
            if c == col:
                board = child
                break
        print(f"AI plays column {col + 1}.")
        render(board)
        if announce(board):
            return
