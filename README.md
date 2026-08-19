# puissance4

A Connect-4 (Puissance 4) engine and terminal front-end in plain Python, no
dependencies. The interesting part is the search: a negamax with alpha-beta
pruning, documented below.

```bash
python3 main.py        # AI searches 5 plies ahead
python3 main.py 7      # deeper, slower
```

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
_, child_score = minmax(-player, child, depth - 1, -b, -a)
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
It is nevertheless seeded up front with the first legal column:

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

## What it buys

Nodes visited on 10 random midgame positions, plain negamax versus the same
search with pruning:

| depth | negamax | alpha-beta | ratio | time |
| ----: | ------: | ---------: | ----: | ---: |
| 4 | 26,274 | 7,615 | 3.5× | 0.08 s |
| 5 | 176,209 | 36,134 | 4.9× | 0.36 s |
| 6 | 1,184,570 | 132,141 | 9.0× | 1.33 s |

The gain compounds with depth, which is the point: pruning does not change the
answer, it buys plies. Both searches return identical values on all 10
positions at each of the three depths — that equivalence is the correctness
test worth keeping.

In theory perfect move ordering reduces the effective branching factor from
`w` to `sqrt(w)`, i.e. roughly double the reachable depth for the same work.
The measured ratios are below that ceiling because moves here are searched
left to right, in no particular order.

## Known limits

- **No move ordering.** Columns are searched 0..6. Trying the centre first, or
  the previous iteration's best move, would cut far more — and it is the single
  change that would most improve the numbers above.
- **No transposition table.** Connect-4 transposes heavily; identical positions
  are re-searched many times over.
- **`WIN` outranks the heuristic by a margin, not by construction.** See above:
  a side holding `BASE` open 3-in-a-rows would close the gap. Deriving `WIN`
  from `len(LINES) * BASE ** (K - 2)` would make it airtight.
- **The evaluation ignores reachability.** A line is counted as open even when
  the cells that would complete it are floating in mid-air, unreachable for
  many moves yet.
