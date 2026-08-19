"""Puissance 4 (Connect-4) game engine and minmax AI.

Board model: a single flat list of HEIGHT*WIDTH ints.
    - value 0  -> empty
    - value 1  -> player 1 (the human, "X")
    - value -1 -> player -1 (the AI, "O")
Cell (row, col) lives at index `row*WIDTH + col`, with row 0 at the top and
row HEIGHT-1 at the bottom (where tokens pile up).
"""

K = 4
HEIGHT = 6
WIDTH = 7

# Line weights form a geometric series: a line holding n of your tokens is
# worth BASE**(n-1). WIN is simply the next term, so a completed line outranks
# any single open line by the same factor the series already uses.
#
# BASE is 20 rather than 10 to keep WIN above the heuristic. The heuristic is
# dominated by its 3-token term, c * BASE**(K-2), so it reaches WIN once a side
# holds BASE open 3-in-a-rows: the safety margin is exactly BASE, and doubling
# the base doubles it. It is a margin, not a guarantee — nothing caps c at BASE
# — but it is enough in practice: the largest heuristic reachable on a legal
# undecided position measures 6377 against WIN = 8000, whereas BASE = 10 gave
# 1587 against WIN = 1000 and so let an undecided position outrank a won one.
BASE = 20
WIN = BASE ** (K - 1)

LINES = []

# Precompute every K-in-a-row line as a list of flat board indices, generated
# only when the line fully fits on its axis (no wrap-around across rows).
for i in range(HEIGHT):
    for j in range(WIDTH):
        base = i * WIDTH + j

        # horizontal (step +1) — needs K columns to the right
        if j + K <= WIDTH:
            LINES.append([base + l for l in range(K)])

        # vertical (step +WIDTH) — needs K rows below
        if i + K <= HEIGHT:
            LINES.append([base + l * WIDTH for l in range(K)])

        # diagonal down-right (step +WIDTH+1) — needs room right and down
        if j + K <= WIDTH and i + K <= HEIGHT:
            LINES.append([base + l * (WIDTH + 1) for l in range(K)])

        # diagonal down-left (step +WIDTH-1) — needs room left and down
        if j - (K - 1) >= 0 and i + K <= HEIGHT:
            LINES.append([base + l * (WIDTH - 1) for l in range(K)])


def score(board, player, depth=0):
    """Heuristic evaluation of `board` from `player`'s perspective.

    Positive favours `player`, negative favours the opponent; a decided
    position returns `±(WIN + depth)`. The value is naturally antisymmetric —
    `score(b, -p, d) == -score(b, p, d)` — because swapping the player swaps
    the two count buckets, which makes it the right quantity for the negamax
    search (see `minmax`), and it never saturates, so distinct positions stay
    distinguishable.

    `depth` is the search depth still remaining when the position was reached,
    so a win found early (much depth left) scores above the same win found
    later. Left at its default of 0 the function is a pure static evaluation.
    """
    counts_player = [0] * K
    counts_opponent = [0] * K

    for line in LINES:

        curr_player = 0
        curr_oppo = 0

        for n in line:
            if board[n] == player:
                curr_player += 1
            elif board[n] == -player:
                curr_oppo += 1

        # only "open" lines (owned by a single side) contribute
        if curr_player != 0 and curr_oppo == 0:
            counts_player[curr_player - 1] += 1
        if curr_player == 0 and curr_oppo != 0:
            counts_opponent[curr_oppo - 1] += 1

    # A completed line ends the game, so it replaces the open-line terms rather
    # than adding to them: the heuristic describes a position still being
    # played out and has nothing to say about a decided one. Adding `depth`
    # ranks a quick win above a slow one — and, mirrored, makes a loss score
    # higher the longer it is postponed, so the AI puts up a fight.
    if counts_player[K - 1] > 0:
        return WIN + depth
    if counts_opponent[K - 1] > 0:
        return -(WIN + depth)

    score = 0
    mult = 1
    for i in range(K - 1):
        score += mult * counts_player[i]
        score -= mult * counts_opponent[i]
        mult *= BASE

    return score


def winning_player(board):
    """Return 1 or -1 if that player owns a completed K-in-a-row, else 0."""
    for line in LINES:
        first = board[line[0]]
        if first != 0 and all(board[n] == first for n in line):
            return first
    return 0


def is_final(board):
    """True if the position is terminal: someone has 4-in-a-row, or the board
    is full (a draw)."""
    if winning_player(board) != 0:
        return True

    # the top cell of column c is index c (row 0); an empty one means not full
    for c in range(WIDTH):
        if board[c] == 0:
            return False

    return True


def get_next_move(board, player):
    """Return the list of legal moves as (column, resulting_board) pairs.

    A token dropped in a column falls to the lowest empty cell. Full columns
    are skipped entirely.
    """
    moves = []

    for j in range(WIDTH):

        # skip full columns
        if board[j] != 0:
            continue

        # find the lowest empty row in column j (tokens settle at the bottom)
        row = 0
        while row + 1 < HEIGHT and board[(row + 1) * WIDTH + j] == 0:
            row += 1

        # a flat list of ints: a slice copy is enough, and far cheaper than
        # deepcopy, which would walk the list element by element
        new_board = board[:]
        new_board[row * WIDTH + j] = player

        moves.append((j, new_board))

    return moves


def minmax(player, board, depth, a, b):
    """Return (best_column, best_score) for `player` to move.

    best_score is `player`'s heuristic value (higher is better; `±(WIN + depth)`
    for a decided position). best_column is None only when there are no moves
    (terminal / full board).
    """
    # evaluate the leaf from the mover's own perspective, passing the depth
    # left so that a win reached sooner outranks the same win reached later
    if depth == 0 or is_final(board):
        return None, score(board, player, depth)

    next_moves = get_next_move(board, player)

    # seed with the first legal column: every branch can evaluate to -inf (a
    # lost position), and `val > best_score` would then never fire and leave
    # best_move as None. Keep `>` so ties go to the first column found.
    best_move = next_moves[0][0]
    best_score = -float('inf')

    # negamax: each child is evaluated from the opponent's perspective, and
    # because `score` is antisymmetric the value to `player` is its negation.
    for col, child in next_moves:
        _, child_score = minmax(-player, child, depth - 1, -b, -a)
        val = -child_score
        if val > best_score:
            best_score = val
            best_move = col

        a = max(a, val)

        if a >= b:
            break

    return best_move, best_score
