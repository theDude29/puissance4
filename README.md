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

Every possible 4-in-a-row is precomputed once into `LINES` as a list of flat
indices. Each candidate line is emitted only when it fully fits on its axis,
which is what keeps a horizontal run from wrapping around the right edge into
the next row.

## Evaluating a position

`score(board, player)` walks `LINES` and counts, for each side, how many lines
it *could still* complete — a line contributes only if the other side has no
token in it. Those counts are weighted by powers of ten (a line holding 3 of
your tokens is worth 100× one holding a single token), and a completed line
returns `±inf`.

Two properties of this function matter for the search:

- **It is antisymmetric**: `score(b, -p) == -score(b, p)`. Swapping the player
  swaps the two count buckets, which flips the sign of the whole sum. This is
  exactly what lets the search be written as negamax.
- **It does not saturate.** Distinct positions keep distinct values, so the
  search has something to compare when no side is winning yet.

## Negamax

Rather than writing separate `max` and `min` branches, every node is evaluated
from the perspective of the side to move. A child is searched for the
*opponent*, and because `score` is antisymmetric the child's value to us is
simply its negation:

```python
_, child_score = minmax(-player, child, depth - 1, -b, -a)
val = -child_score
```

One code path handles both players. The leaf case, `score(board, player)`,
follows the same convention: it is read from the mover's own point of view.

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
enough for the parent to prefer its existing alternative, so searching on gains
nothing. It matters concretely here because `score` returns `±inf` for a
decided position: with a strict `>`, a forced win searched against `b = +inf`
would never prune — precisely the case where pruning pays most.

### Fail-soft

On a cutoff the function returns the value it has actually accumulated, which
may sit outside the window. That value is a *bound*, not the node's exact
score. This is fine because only the root's value is consumed, and the root is
searched with a full `(-inf, +inf)` window: cutting there requires `val >= +inf`,
an outright win, in which case the accumulated value is already the true
maximum. Every other node's value is only ever read by its parent as a bound.

The cutoff uses `break` rather than an early `return` so that the function
keeps its `(best_column, best_score)` contract on every path — returning a
bare score from the cutoff branch makes the caller's tuple unpacking raise
`TypeError`.

### One subtlety about `best_move`

`best_score` starts at `-inf` and is updated under `val > best_score`. In a
position that is lost however you play, every branch evaluates to `-inf`, the
guard never fires, and `best_move` would stay `None` — the caller then finds no
matching column and crashes, exactly when the human is about to win. So
`best_move` is seeded with the first legal column:

```python
best_move = next_moves[0][0]
```

Seeding rather than relaxing the guard to `>=` is deliberate: `>=` would also
remove the `None`, but it would shift tie-breaking to the *last* equal-scoring
column and change the AI's play in ordinary positions, where ties are common.

## What it buys

Nodes visited on 10 random midgame positions, plain negamax versus the same
search with pruning:

| depth | negamax | alpha-beta | ratio |
| ----: | ------: | ---------: | ----: |
| 4 | 26,274 | 7,547 | 3.5× |
| 5 | 176,209 | 34,853 | 5.1× |
| 6 | 1,184,570 | 128,198 | 9.2× |

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
  the previous iteration's best move, would cut far more.
- **No transposition table.** Connect-4 transposes heavily; identical positions
  are re-searched many times over.
- **Mate scores carry no distance.** A win is `+inf` whether it is one move
  away or five, so the AI has no reason to prefer the quick kill and may
  dawdle in a won position. Folding depth into the terminal score would fix it.
- **`deepcopy` per move.** Boards are flat lists of ints; a slice copy would do
  the same job for much less.
