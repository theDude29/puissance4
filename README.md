# puissance4

A Connect-4 (Puissance 4) engine and terminal front-end in plain Python, no
dependencies. The interesting part is the search: a negamax with alpha-beta
pruning, a transposition table and time-controlled iterative deepening,
documented below.

```bash
python3 main.py        # AI thinks 2 s per move
python3 main.py 0.5    # faster, weaker
```

The argument is the AI's thinking time in seconds, not a depth: it deepens
until the budget runs out and reports how far it got, so it searches further in
simple positions than in crowded ones.

You play `X` and move first; the AI answers as `O`. Columns are 1-indexed at
the prompt, `q` quits.

---

## The board

A single flat list of `HEIGHT * WIDTH` ints — `0` empty, `1` human, `-1` AI.
Cell `(row, col)` lives at `row * WIDTH + col`, row 0 at the top, so tokens
settle towards `HEIGHT - 1`.

Every possible 4-in-a-row is precomputed once into `LINES` (69 of them) as a
list of flat indices. Each candidate line is emitted only when it fully fits on
its axis, which is what keeps a horizontal run from wrapping around the right
edge into the next row.

Because a board is a flat list of ints, `get_next_move` copies it with a slice,
`board[:]`, not `deepcopy`. The two are equivalent here — there is no nested
structure to share — and the slice is roughly a third faster over a full
search.

## Evaluating a position

`score(board, player, depth=0)` walks `LINES` and counts, for each side, how
many lines it *could still* complete — a line contributes only if the other
side has no token in it. Those counts are weighted as a geometric series in
`BASE`: a line holding `n` of your tokens is worth `BASE ** (n - 1)`.

Two properties of this function matter for the search:

- **It is antisymmetric**: `score(b, -p, d) == -score(b, p, d)`. Swapping the
  player swaps the two count buckets, which flips the sign of the whole sum.
  This is exactly what lets the search be written as negamax.
- **It does not saturate.** Distinct positions keep distinct values, so the
  search has something to compare when no side is winning yet.

### Decided positions, and why they carry depth

A completed line returns `±(WIN + depth)` and *replaces* the open-line terms
rather than adding to them: the heuristic describes a position still being
played out and has nothing to say about a decided one.

`WIN` is `BASE ** (K - 1)` — simply the next term in the same series. `depth`
is the search depth still remaining when the position was reached, so a win
found early, with much depth left, scores above the same win found later. The
effect is worth stating in both directions: the AI prefers the quick kill, and
mirrored, it prefers the *slowest* loss, so it stops conceding early and makes
you finish the job.

Using a finite `WIN` is what makes this possible at all. With `±inf` every win
scored the same, so among several winning moves the search kept whichever it
happened to try first. Measured over 400 random positions where an immediate
win was available:

| terminal score | immediate win available but not played |
| --- | ---: |
| `±inf` | 100 / 400 |
| `±(WIN + depth)` | 0 / 400 |

### Why `BASE` is 20

`WIN` has to outrank any undecided position, and it is not guaranteed to. The
heuristic is dominated by its 3-token term, `c * BASE ** (K - 2)`, which
reaches `WIN` once a side holds `BASE` open 3-in-a-rows — so the margin is
exactly `BASE`, and doubling the base doubles it.

It is a margin, not a proof: nothing caps `c` at `BASE`. But it is enough in
practice. Hill-climbing for the largest heuristic on a legal, *undecided*
position finds 6377 against `WIN = 8000`, whereas `BASE = 10` reached 1587
against `WIN = 1000` — a position with no winner outranking a won one, which
the search must never see.

## Negamax

Rather than writing separate `max` and `min` branches, every node is evaluated
from the perspective of the side to move. A child is searched for the
*opponent*, and because `score` is antisymmetric the child's value to us is
simply its negation:

```python
_, child_score = minmax(-player, child, depth - 1, -b, -a, tt)
val = -child_score
```

One code path handles both players. The leaf case, `score(board, player,
depth)`, follows the same convention: it is read from the mover's own point of
view.

## Alpha-beta

The search carries a window `(a, b)` — read as: *we already know this node's
value lies above `a`, and the parent will discard anything at or above `b`.*

- **`a` (alpha)** is the best value the side to move has secured so far
  anywhere on the path. It only ever rises.
- **`b` (beta)** is the ceiling imposed by the parent. If this node turns out
  to be worth `b` or more, the parent will never choose the move that leads
  here, because it already has an alternative at least that good.

Recursing negates *and swaps* the bounds:

```python
minmax(-player, child, depth - 1, -b, -a)
```

The swap is the whole trick. Our lower bound is the opponent's upper bound and
vice versa, and negation converts between the two perspectives — the same
identity that makes negamax work in the first place.

The cutoff:

```python
    a = max(a, val)
    if a >= b:
        break
```

Once `a` reaches `b`, the remaining siblings cannot change the parent's
decision, so the loop stops and the rest of that subtree is never generated.

### A cutoff, concretely

Root, our move, full window `(-inf, +inf)`:

1. Child **A** is searched and comes back worth `3`. We set `a = 3`: we can
   guarantee at least 3.
2. Child **B** is searched with the negated, swapped window `(-inf, -3)`. Inside
   B the opponent is to move and maximizes their own value, cutting as soon as
   they find something worth `>= -3` to them.
3. B's first grandchild is worth `-1` to us, i.e. `+1` to the opponent. Since
   `1 >= -3`, B cuts immediately.

The reasoning: the opponent can hold B to at most `1` from their side, meaning
at most `-1` for us — worse than the `3` that A already guarantees. Whatever B's
remaining grandchildren contain, B is not getting picked. They are never
searched.

### Why `>=` and not `>`

The comparison has to fire on equality. A value *equal* to `b` is already
enough for the parent to prefer the alternative it holds, so searching on gains
nothing and only costs nodes. With integer scores drawn from a coarse series,
exact ties between siblings are common, so this is not a rare edge case.

### Fail-soft

On a cutoff the function returns the value it has actually accumulated, which
may sit outside the window. That value is a *bound*, not the node's exact
score — which is fine, because a non-root node's value is only ever read by its
parent as a bound. The root is searched with `(-inf, +inf)` and every score is
finite, so `a >= b` can never hold there: the root never cuts, and the value it
returns is exact.

The cutoff uses `break` rather than an early `return` so that the function
keeps its `(best_column, best_score)` contract on every path — returning a
bare score from the cutoff branch makes the caller's tuple unpacking raise
`TypeError`.

### Seeding `best_move`

`best_score` starts at `-inf` as a sentinel and is updated under
`val > best_score`, so `best_move` is only assigned when some child beats it.
It is nevertheless seeded up front with the first column to be tried:

```python
best_move = next_moves[0][0]
```

With finite terminal scores this is belt-and-braces — any real value beats the
sentinel, so the first child always assigns. It mattered when a loss scored
`-inf`: in a position lost however you play, every branch tied the sentinel,
the guard never fired, and `best_move` stayed `None`, leaving the caller with
no column to play and crashing the game exactly when the human was about to
win. Seeding keeps that guarantee independent of what `score` returns.

Seeding rather than relaxing the guard to `>=` is deliberate: `>=` would also
remove the `None`, but it would shift tie-breaking to the *last* equal-scoring
column and change the AI's play in ordinary positions, where ties are common.

## Transposition table

The same position is reachable by many move orders, and the plain search
re-explores each arrival from scratch. `search` keeps a dict of what it has
already worked out, keyed on `(tuple(board), player)` — the board alone is not
enough, since a position is worth different things depending on who has to
move.

An entry holds four things: the depth it was searched to, the value, a bound
kind, and the best move found. The bound kind is not optional. Under alpha-beta
a value is often not exact: a cutoff proves only that the node is worth *at
least* that much (`LOWER`), and a node where nothing raised alpha proves only
*at most* (`UPPER`). Storing the number without saying which it is would be
wrong. On a probe, `EXACT` returns immediately, `LOWER` raises alpha, `UPPER`
lowers beta, and if the window collapses the node is done.

The stored move is read back whatever depth produced it — as an ordering hint
it costs nothing to be wrong — while the *value* is used only if the entry came
from a search at least as deep as the one being asked for.

### Why the key is a tuple and not a Zobrist hash

Zobrist is the textbook answer: XOR a random word per occupied cell, update it
in O(1) as moves are made. Measured here, it came out slightly *slower* — 2.18×
against 2.26× for the tuple key. `tuple()` and its hash are C-level, the
incremental XOR is Python bytecode, and the key is not the bottleneck anyway:
`score` walks 69 lines per node and dominates everything. Zobrist would also
mean threading the hash back out of `get_next_move`, changing its contract, and
it reintroduces a collision risk that a tuple key does not have — dicts compare
keys by equality, so an exact key cannot alias. Memory does not argue for it
either: a depth-9 search stores about 7,600 entries.

It would earn its keep in a C engine with bitboards and incremental make/unmake,
where the eval is cheap enough for hashing to matter. Not in this one.

### Mate scores, and the invariant that saves them

`score` returns `±(WIN + depth)`, so a mate score depends on the depth that
found it. That normally makes a transposition table unsound — the same mate,
stored from one depth and reused at another, comes back wrong — and engines
renormalise mate values on the way in and out.

That is not needed here, because of a property of Connect-4 specifically: every
move adds exactly one token, so the number of tokens fixes the ply. Two move
orders reaching the same position always arrive at the same ply, hence at the
same remaining depth. Entries left behind by shallower passes are refused by
the `entry_depth >= depth` guard. Instrumenting the probe over a full search
finds **0 reads at a depth other than the one requested**, iterative deepening
included, so every value ever reused was produced at exactly the depth it is
reused at — and the search still returns what a plain fixed-depth negamax
returns.

This is load-bearing. A variant where a move did not monotonically fill the
board would break it, and the mate scores would then need renormalising.

## Iterative deepening

`search` is the public entry point. It searches depth 1, then 2, and so on, all
passes sharing one table.

Re-searching from scratch each time sounds wasteful and is not: the tree grows
geometrically, so every pass before the last costs a fraction of the total, and
each one leaves its best move in the table for the next to try first. Good move
ordering is what makes alpha-beta cut early, and the cheapest source of a good
guess is a shallower search of the same position.

Moves are otherwise ordered centre-outwards, since central columns take part in
more lines. That ordering alone is worth more than the table: it is what turns
a 2.2× speedup into 5.6×.

## Time control

Deepening is also what makes a clock usable: the search always has a complete,
shallower answer in hand, so it can be stopped at any moment and still have
something to play. `search(player, board, time_limit=2.0)` deepens until the
budget is spent and returns the depth it reached alongside the move.

From the opening, on this machine:

| budget | depth reached |
| ---: | ---: |
| 0.1 s | 6 |
| 0.5 s | 8 |
| 2 s | 10 |

Overshoot is small — 2.7 ms on a 1 s budget at worst — because the clock is
read inside the search rather than only between passes.

### Only complete passes count

The rule that matters: an interrupted pass is thrown away whole. It has
examined some root moves and not others, so its best-so-far is not a best move
at all, only the best of an arbitrary prefix, and it can be worse than what the
previous pass already established. `search` keeps the last pass that finished.

### Aborting without corrupting the table

Running out of time raises `TimeUp` from inside the recursion rather than
returning a sentinel, and that choice is what keeps the transposition table
sound. Entries are written *after* a node's move loop completes, so unwinding
by exception skips every pending store on the way out — no half-searched value
is ever recorded. Subtrees that did finish before the abort keep their entries,
which are valid, and the next pass reuses them.

Checked directly: after aborting mid-pass on a tight budget, every surviving
entry still agrees with a clean search at its own depth — exact values exact,
`LOWER` and `UPPER` bounding the true value on the right side. 0 bad entries
out of 1,600.

Reading the clock costs nothing measurable on the fixed-depth path, since the
check is guarded on there being a budget at all, and even with one it only
fires at nodes with at least two plies left below them — most nodes are at the
very bottom, and a two-ply subtree finishes in microseconds.

### Stopping on a decided position

A forced win needs no further search. Because deepening goes shallow first, the
first pass to see a mate sees the shortest one, so `search` returns as soon as
a mate score appears rather than spending the rest of the budget confirming it.
Measured over 250 forced wins, this never lengthens the mate; it can pick a
different move among equally fast wins.

## What it buys

Nodes visited on 10 random midgame positions, at each stage of the search:

| depth | negamax | + alpha-beta | + table, ordering, deepening |
| ----: | ------: | -----------: | ---------------------------: |
| 4 | 26,274 | 7,615 | 3,469 |
| 5 | 176,209 | 36,134 | 10,113 |
| 6 | 1,184,570 | 132,141 | 25,403 |
| 7 | 7,830,509 | 579,869 | 62,282 |

In wall-clock terms the last column is 5.2× faster than alpha-beta alone at
depth 6 and 9.4× at depth 7. The gain compounds with depth, which is the whole
point: none of this changes the answer, it buys plies — which the time control
then spends, reaching depth 10 from the opening on a 2 s budget.

Every variant returns identical values on all 10 positions at every depth
tested, and matches a plain unpruned negamax. That equivalence is the
correctness test worth keeping — and it is not a given for a transposition
table, see above.

## Known limits

- **The table lives for one search.** It is dropped between moves, so work is
  redone from one turn to the next. Keeping it would need the mate-score
  renormalisation described above, since entries would then be probed at a
  different remaining depth than they were stored at.
- **The budget is per move, flat.** Every move gets the same seconds,
  regardless of whether the position is critical or already decided. Spending
  more on sharp positions and less on quiet ones is the usual next step.
- **`WIN` outranks the heuristic by a margin, not by construction.** See above:
  a side holding `BASE` open 3-in-a-rows would close the gap. Deriving `WIN`
  from `len(LINES) * BASE ** (K - 2)` would make it airtight.
- **The evaluation ignores reachability.** A line is counted as open even when
  the cells that would complete it are floating in mid-air, unreachable for
  many moves yet.
