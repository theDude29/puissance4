"""Puissance 4 (Connect-4) game engine and minmax AI.

Board model: a single flat list of HEIGHT*WIDTH ints.
    - value 0  -> empty
    - value 1  -> player 1 (the human, "X")
    - value -1 -> player -1 (the AI, "O")
Cell (row, col) lives at index `row*WIDTH + col`, with row 0 at the top and
row HEIGHT-1 at the bottom (where tokens pile up).

The AI is built in layers, each one useless without the one below it:

    score()   static evaluation — how good a position *looks*, judged on the
              spot, without playing anything out. Fast and shallow.
    minmax()  negamax + alpha-beta + move ordering + transposition table —
              how good a position *plays out*, by looking ahead and assuming
              the opponent plays as well as we do.
    search()  iterative deepening — the entry point the front-end calls. Runs
              minmax at depth 1, 2, 3 ... and feeds each pass to the next.

Read them in that order; each section explains why it exists. The README
covers the same ground with measurements.
"""

from time import perf_counter

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

# Transposition-table bound kinds. Under alpha-beta a stored value is not
# always the node's exact value: a cutoff only proves a lower bound, and a node
# where nothing raised alpha only proves an upper one. Storing the number
# without saying which it is would be wrong.
EXACT, LOWER, UPPER = 0, 1, 2

# Move-ordering key: central columns take part in more lines, so they are more
# often best, and trying them first makes alpha-beta cut sooner.
CENTRE = {c: abs(c - WIDTH // 2) for c in range(WIDTH)}

# How much of the tree must be left below a node for it to be worth reading the
# clock. Most nodes sit at the very bottom, so checking everywhere would spend
# more time on `perf_counter` than on searching; a subtree with two plies left
# finishes in microseconds, which is fine as abort granularity.
CHECK_DEPTH = 2


class TimeUp(Exception):
    """Raised inside `minmax` when the move's time budget is spent.

    Unwinding by exception rather than by returning a sentinel is what keeps
    the transposition table sound: entries are written after a node's move loop
    finishes, so an abort skips every store on the way out and no half-searched
    value is ever recorded. Subtrees that did complete before the abort stay in
    the table and remain valid.
    """


LINES = []

# Precompute every K-in-a-row line as a list of flat board indices. The
# evaluation below walks all of them at every node of the search — millions of
# times per move — so the geometry is worked out once, here, and never again:
# checking a position afterwards is just reading cells.
#
# The four guards are what keep the flat layout honest. Indices 6 and 7 are
# neighbours in the list but sit on different rows (end of row 0, start of
# row 1), so a horizontal run started near the right edge would silently wrap
# around and "win" across two rows. Each direction is emitted only from squares
# where it fully fits on the board.
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
    # The idea: a position is good for you if you have many ways left to make
    # four in a row, and few for the opponent. So count, for each side, the
    # lines it could still complete, bucketed by how far along it already is —
    # counts[0] is lines holding one of its tokens, counts[1] two, and so on.
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

        # A line holding tokens from both sides is dead: neither player can
        # ever fill it, so it is worth nothing to anyone and simply drops out.
        # Only lines owned by a single side — "open" lines — are counted, which
        # is also why the two conditions below are exclusive rather than an
        # if/else over one count.
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

    # Weigh the buckets against each other. One line already holding three
    # tokens is worth far more than three separate lines holding one each — it
    # is one move from winning — so bucket values grow geometrically in BASE
    # rather than linearly. Subtracting the opponent's buckets from ours is
    # what makes the whole function antisymmetric, which the negamax search
    # below relies on.
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

    # No winner, so the only other way to end is a full board — a draw. A
    # column is full exactly when its top cell is taken, and the top cell of
    # column c is index c (row 0), so one pass over the first row settles it.
    for c in range(WIDTH):
        if board[c] == 0:
            return False

    return True


def get_next_move(board, player):
    """Return the list of legal moves as (column, resulting_board) pairs.

    A token dropped in a column falls to the lowest empty cell. Full columns
    are skipped entirely.

    Each move comes with the entire board it leads to, rather than being
    applied to a shared board and undone afterwards. That costs one copy per
    move, but it keeps the search free of any make/unmake bookkeeping: a node
    never has to restore anything, because it never modified anything. For a
    board this small it is the better trade.
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


def minmax(player, board, depth, a, b, tt, deadline=None):
    """Return (best_column, best_score) for `player` to move.

    best_score is `player`'s heuristic value (higher is better; `±(WIN + depth)`
    for a decided position). best_column is None only when there are no moves
    (terminal / full board).

    `a` and `b` are the alpha-beta window: the range of values still worth
    knowing precisely. `a` is the best the side to move has already secured
    somewhere else on the path, `b` the ceiling above which the parent will
    stop caring, because it already holds something at least that good. As
    soon as those two meet, whatever is left in this node cannot change any
    decision above it, and the loop stops — that is the whole of the pruning.
    A first call therefore passes the widest possible window.

    `tt` is the transposition table for this search — see `search`, which owns
    it. Callers outside the recursion should go through `search` rather than
    calling this directly.

    `deadline` is an absolute `perf_counter` reading past which the search
    gives up by raising `TimeUp`. None means no limit.
    """
    # evaluate the leaf from the mover's own perspective, passing the depth
    # left so that a win reached sooner outranks the same win reached later.
    # Done before probing: leaves are never stored, so a probe would only ever
    # miss, and building the key costs more than the test.
    if depth == 0 or is_final(board):
        return None, score(board, player, depth)

    # Out of time: abandon the whole pass. `search` keeps the previous, fully
    # completed pass instead — see there for why a half-finished one is not
    # merely less accurate but actively unsafe to use.
    if deadline is not None and depth >= CHECK_DEPTH and perf_counter() > deadline:
        raise TimeUp

    # `player` belongs in the key: the same board is worth different things
    # depending on who has to move.
    key = (tuple(board), player)
    a0 = a
    tt_move = None

    entry = tt.get(key)
    if entry is not None:
        entry_depth, entry_value, entry_flag, tt_move = entry

        # the stored move is a useful ordering hint whatever depth produced it,
        # but the value may only be trusted if it came from a search at least
        # as deep as the one being asked for
        if entry_depth >= depth:
            if entry_flag == EXACT:
                return tt_move, entry_value
            if entry_flag == LOWER:
                a = max(a, entry_value)
            else:
                b = min(b, entry_value)
            if a >= b:
                return tt_move, entry_value

    next_moves = get_next_move(board, player)

    # best move first, then centre outwards. `tt_move` is None on a miss, which
    # makes the first key constant and leaves the centre ordering intact.
    next_moves.sort(key=lambda m: (m[0] != tt_move, CENTRE[m[0]]))

    # seed with the first column tried: every branch can evaluate to -inf (a
    # lost position), and `val > best_score` would then never fire and leave
    # best_move as None. Keep `>` so ties go to the first column tried.
    best_move = next_moves[0][0]
    best_score = -float('inf')

    # Negamax, rather than the textbook pair of maximising and minimising
    # branches: every node is read from the point of view of whoever is to
    # move, so both players "maximise" and one code path serves both. The
    # child is searched for the opponent, and because `score` is antisymmetric
    # its value to us is exactly its negation. The window is negated and
    # swapped for the same reason — our floor is the opponent's ceiling.
    for col, child in next_moves:
        _, child_score = minmax(-player, child, depth - 1, -b, -a, tt, deadline)
        val = -child_score
        if val > best_score:
            best_score = val
            best_move = col

        a = max(a, val)

        # Cutoff: this node is already worth at least `b` to us, so the parent
        # — which can hold us to `b` by choosing something else — will never
        # pick the move that leads here. The siblings left cannot change that,
        # so they are never generated. Note `>=` and not `>`: a value merely
        # equal to `b` is already enough for the parent to prefer what it has.
        if a >= b:
            break

    # classify what the search actually proved: a value that never beat the
    # incoming alpha is only an upper bound, one that reached beta is only a
    # lower bound, and anything in between is exact
    if best_score <= a0:
        flag = UPPER
    elif best_score >= b:
        flag = LOWER
    else:
        flag = EXACT

    if entry is None or entry[0] <= depth:
        tt[key] = (depth, best_score, flag, best_move)

    return best_move, best_score


def search(player, board, max_depth=HEIGHT * WIDTH, time_limit=None):
    """Return (best_column, best_score, depth) for `player`, by iterative
    deepening under a time budget.

    Searches depth 1, then 2, ... all passes sharing one transposition table,
    and stops at `max_depth` or when `time_limit` seconds are up, whichever
    comes first. `time_limit=None` means no clock at all, which makes this a
    plain fixed-depth search — how the tests pin the values down. The returned
    `depth` is the depth actually reached, which varies with the position: a
    near-empty board is cheap to search deeply, a crowded one is not.

    The re-search is far cheaper than it looks: the tree grows geometrically,
    so every pass before the last costs a fraction of the total, and each one
    leaves its best move in the table for the next to try first — which is what
    makes alpha-beta cut early. Deepening is also exactly what makes a time
    budget usable: there is always a complete, shallower answer in hand.

    Only complete passes are ever returned. An interrupted pass has looked at
    some root moves and not others, so its best-so-far is not the best move —
    it is merely the best of an arbitrary prefix, and can be worse than what
    the previous pass already knew. `TimeUp` therefore discards the pass whole.

    The table is created here and dropped on return, which keeps a useful
    invariant. `score` returns `±(WIN + depth)`, so a mate score depends on the
    depth that found it — normally a reason to renormalise before storing. It
    is not needed here: in Connect-4 the number of tokens fixes the ply, so a
    position always occurs at the same ply and therefore at the same remaining
    depth within a pass, while entries left by shallower passes are refused by
    the `entry_depth >= depth` guard. Every value ever reused was produced at
    exactly the depth it is reused at, so the result matches a plain fixed-depth
    negamax.
    """
    tt = {}
    deadline = None if time_limit is None else perf_counter() + time_limit
    best_move, best_score, reached = None, None, 0

    for depth in range(1, max(1, max_depth) + 1):
        try:
            # Depth 1 runs without a deadline so that there is always a legal
            # move to return, however small the budget. It costs one ply.
            move, value = minmax(player, board, depth, -float('inf'),
                                 float('inf'), tt,
                                 deadline if depth > 1 else None)
        except TimeUp:
            break

        best_move, best_score, reached = move, value, depth

        # A decided position cannot be improved on by looking further. Because
        # deepening goes shallow first, the earliest pass to see a mate sees
        # the shortest one, so the remaining budget would only confirm it.
        # Measured over 250 forced wins, stopping here never lengthens the mate;
        # it can pick a different move among equally fast wins, which is the
        # same tie-break arbitrariness the move ordering already has.
        if abs(value) >= WIN:
            break

    return best_move, best_score, reached
