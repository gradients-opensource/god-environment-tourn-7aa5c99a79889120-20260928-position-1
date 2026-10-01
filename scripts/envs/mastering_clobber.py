"""Clobber teacher — iterative-deepening negamax with a mobility-gap leaf score.

pick_move(state, player_id, time_budget, max_depth) returns the move label: leaf score =
own legal-move count minus the opponent's (in Clobber the last player able to move wins, so
mobility, not material, is the signal). The side to move is threaded down the recursion since
state.current_player() is undefined at a terminal.

ClobberSearchTeacher adapts pick_move to the step()/last_reason interface. Its per-game
search budget (time_budget, max_depth) is sampled once per game from fixed ranges, so games
explore slightly different depths. `last_reason` carries a ranking of the top candidate ids
(describe_ranking), read straight off the ROOT of the deepest search that finished inside the
budget — the same tree that produced the label, not a second re-search. There used to be a
second, shallower full-window pass just for the ranking; that construction could in principle
land its own argmax on a different move than the label, so the code that forced the label to
the front of the sentence was patching over a real disagreement rather than describing an
impossible one. Reading the root scores off the search that already ran removes the second
opinion instead of overruling it.
"""

import os
import random
import time

import pyspiel

_DEFAULT_BUDGET = 0.5
_DEFAULT_DEPTH = 16

# Poll the clock once per this many node expansions (a per-node syscall would be a drag).
_CLOCK_EVERY = 256

_INF = float("inf")

# Per-game search ranges — sampled once per game so depth/time vary across games.
#
# ⚠ 2026-08-04 — RECORD CORRECTION. sft_env_config.py's TOURNAMENT_TEACHERS comment justifies
# this teacher as producing a "smooth DETERMINISTIC mobility label → learnable". The LEAF EVAL
# is deterministic; the LABEL THIS FILE EMITS IS NOT. Three separate sources of cross-game
# variance for one and the same position:
#   1. _max_depth is resampled per game from {14,15,16}  (rng.randint below)
#   2. _time_budget is resampled per game from U(0.3, 1.0) (rng.uniform below)
#   3. pick_move() is ITERATIVE-DEEPENING and returns the best move of the deepest depth that
#      FINISHED inside the budget — so the achieved depth, and hence the move, depends on
#      WALL-CLOCK SPEED. Identical position + identical sampled budget can still yield
#      different moves when the box is busy — and generation runs many workers in parallel,
#      which is exactly when steps slow down.
# It is a RELATED failure mode to the two the repo already paid for (othello MCTS-800 →
# student 0.179; the earlier clobber solver's action-id artifact + stochastic MCTS fallback →
# student 0.45) — but do not over-claim the analogy: the repo's own diagnosis there was that a
# LOOK-AHEAD target is unlearnable (its deterministic othello mobility heuristic still gave
# student 0.11 at 3x data), and sft_env_config.py itself records "DETERMINISM ≠ LEARNABILITY".
# Label noise is an ADDITIONAL problem here, not the same one.
#
# WHICH MECHANISM ACTUALLY DOMINATES (corrected 2026-08-04 after review — the first draft of
# this note had it backwards, so read this before picking a lever):
#   * The opening SATURATES the budget, so (3) is the DOMINANT mechanism, not a dormant one.
#     A full board makes every orthogonally-adjacent pair legal: 4x5 → 31 moves, 5x5 → 40,
#     5x6 → 49. Alpha-beta at b≈31-49 needs ~b^(d/2) ≈ 10^10 nodes for depth 14 — and
#     _order_moves clones the state once per child at every node just to rank. Depth 14 is
#     simply unreachable in an opening within 0.3-1.0 s, on ANY of the three boards,
#     regardless of load. The achieved depth there is whatever the clock allows.
#   * The ~134 ms/step figure quoted below does NOT mean "the clock rarely fires". It is a
#     MEAN, and a mean does not bound a rate: if a fraction p of steps saturate they cost
#     >=300 ms each, so p·650 <= 134 permits p up to ~20%. Roughly one label in five being
#     clock-determined is a live mechanism, and it is STRUCTURAL (opening positions), with
#     load only widening it.
#   * (1) is close to INERT, the opposite of what it looks like. Clobber lasts AT MOST
#     rows*cols-1 plies (4x5 → 19, 5x5 → 24, 5x6 → 29) because the board starts full and every
#     move removes exactly one piece. With a 2-6 ply random opening, a 4x5 game has <=17 plies
#     left, so depth 14-16 reaches terminal, the search is EXACT and 14/15/16 agree. Elsewhere
#     the clock stops the search long before depth 14, so the sampled cap never binds either.
#   * (2) therefore matters mainly THROUGH (3): the budget is what selects the achieved depth.
#   * Unlike goofspiel/othello the state space is huge, so exact duplicate positions carrying
#     contradictory labels are rare; the harm is the weaker "features → action becomes a
#     noisier function", not a flat contradiction.
# So: real defect, uncertain size. NOT fixed here on purpose — it trades against generation
# throughput and must be MEASURED, not guessed.
#
# ⤷ 2026-08-05 — THE FIX IS FAR CHEAPER THAN THIS NOTE ASSUMES, because the mechanism is already
#   half-built. `_search` already threads a NODE COUNTER (`nodes: list[int]`, incremented on entry)
#   and already consults it — `if nodes[0] % _CLOCK_EVERY == 0 and time.monotonic() > stop_at`.
#   Swapping that predicate for `nodes[0] >= node_budget` makes the search DETERMINISTIC BY
#   CONSTRUCTION: same position + same budget ⇒ same tree ⇒ same move, on any box, at any load.
#   Mechanism (3) — the dominant one — disappears entirely rather than being reduced.
#   It also mostly dissolves the throughput objection above: a node budget does not do LESS work,
#   it bounds the SAME work deterministically, so it can be calibrated to today's mean wall time.
#   And the determinism does not need measuring to be believed — it is an argument, not a result.
#
#   WHAT DOES NEED A MEASUREMENT is exactly one number: the node count pick_move actually reaches
#   at the current 0.3-1.0 s budget (report mean and, more usefully, the p10/p50/p90 — the tail is
#   what the clock is currently truncating). Set node_budget from that and the mean search depth is
#   preserved. Picking it blind is the one thing that could hurt: too low silently shallows every
#   label, too high blows the generation window.
#   Cheap instrumentation, no GPU: accumulate nodes[0] per pick_move call during a normal clobber
#   generation run and print the percentiles at the end.
#   Sequencing note: clobber's field spread is narrow (IQR 0.039 over 1176 games, 12 miners), so
#   this buys label quality in an env that is close to a wash — worth doing, not worth doing first.
#
# THE LEVERS, when there is a GPU to gate on — in the corrected order:
#   * PRIMARY: _time_budget. It has NO env override, and it is the knob that actually picks
#     the achieved depth. Pinning it to the TOP of its range (1.0) is the safe direction:
#     more time can only let MORE depths complete, never fewer. Note the guarantee is
#     monotonicity in ACHIEVED DEPTH — "deeper is better" is a separate empirical claim
#     (search pathology with a heuristic leaf eval is rare but real), so still gate on eval.
#   * SECONDARY: CLOBBER_MAX_DEPTH=<n> pins (1) with no code change, but per the above it is
#     the LEAST useful of the three. Gotchas: .isdigit() accepts "0", and that is worse than it
#     looks — range(1, 0+1) is empty, so pick_move never calls _order_moves and returns
#     legal_actions(player)[0], the LOWEST ACTION ID, unsearched and unordered. That is exactly
#     the action-id artifact sft_env_config records as having made the previous clobber solver
#     unlearnable (student 0.45). (An earlier version of this line called it "the unsearched
#     move-ordering favourite", which wrongly implies the ordering ran.) It also rejects
#     negatives, and a non-numeric value silently falls back to the sample instead of erroring.
#   * Measure BOTH label consistency (same position, repeated runs → same move?) AND the
#     student head-to-head. Teacher strength is the wrong metric — that is the standing rule.
#   * ⚠ Treat the "~2.9k games/window vs a ~4.5k target" bottleneck framing below as
#     UNVERIFIED: GAME_TARGETS['clobber'] is a SAMPLE target, and _samples_to_games divides by
#     WINDOW_RATIO/WIN_RATE_ESTIMATE/STRATIFIED_RETENTION, so ~2.9k GAMES is roughly 5.4k
#     samples — i.e. it may already MEET the target and the "bottleneck" may be a units
#     conflation. Do not spend throughput budget defending it until that is checked.
#
# One precision note on (1)/(2): the per-game seed is base_seed + g + (g % 2), which collides
# on adjacent pairs (g=1,2 share a draw; g=3,4 share one), so it is ~3 distinct draws per 4
# games rather than 4 — "resampled per game" in substance, not literally every game.
_TIME_BUDGET_RANGE = (0.3, 1.0)
_MAX_DEPTH_RANGE = (14, 16)


class _OutOfTime(Exception):
    pass


def _flip(side: int) -> int:
    return side ^ 1


def _leaf_score(state: pyspiel.State, side: int) -> float:
    mine = len(state.legal_actions(side))
    theirs = len(state.legal_actions(_flip(side)))
    return float(mine - theirs)


def _order_moves(state: pyspiel.State, side: int, moves: "list[int]") -> "list[int]":
    """Order moves by a one-ply peek (winning moves first) so alpha-beta prunes sooner."""
    foe = _flip(side)

    def rank(move: int) -> float:
        nxt = state.clone()
        nxt.apply_action(move)
        if nxt.is_terminal():
            payoff = nxt.returns()[side]
            if payoff > 0:
                return -_INF
            if payoff < 0:
                return _INF
            return 0.0
        return float(len(nxt.legal_actions(foe)) - len(nxt.legal_actions(side)))

    # (rank, index) keeps ties in original order — a stable sort by rank.
    decorated = sorted((rank(m), i, m) for i, m in enumerate(moves))
    return [m for _, _, m in decorated]


def _search(
    state: pyspiel.State, side: int, depth: int, alpha: float, beta: float,
    stop_at: float, nodes: "list[int]",
) -> float:
    nodes[0] += 1
    if nodes[0] % _CLOCK_EVERY == 0 and time.monotonic() > stop_at:
        raise _OutOfTime()

    if state.is_terminal():
        return state.returns()[side]
    if depth == 0:
        return _leaf_score(state, side)

    best = -_INF
    for move in _order_moves(state, side, state.legal_actions(side)):
        nxt = state.clone()
        nxt.apply_action(move)
        best = max(best, -_search(nxt, _flip(side), depth - 1, -beta, -alpha, stop_at, nodes))
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def _best_at_depth(
    state: pyspiel.State, side: int, depth: int, stop_at: float, nodes: "list[int]",
) -> "tuple[int, list[tuple[float, int]]]":
    """One root pass at a fixed depth: returns the argmax move AND every root move's score,
    in the same walk that picks the argmax. The scores after the first are read from a window
    narrowed by the running best-so-far (standard alpha-beta root ordering), so they can be
    bounds rather than exact values for the trailing candidates — that is the price of getting
    them for free out of the search that already has to run for the label, instead of paying
    for a second independent pass that has its own argmax and could name a different move."""
    ordered = _order_moves(state, side, state.legal_actions(side))
    chosen = ordered[0]
    top = -_INF
    alpha = -_INF
    scored = []
    for move in ordered:
        nxt = state.clone()
        nxt.apply_action(move)
        score = -_search(nxt, _flip(side), depth - 1, -_INF, -alpha, stop_at, nodes)
        scored.append((score, move))
        if score > top:
            top = score
            chosen = move
        alpha = max(alpha, top)
    return chosen, scored


def _search_root(
    state: pyspiel.State,
    side: int,
    time_budget: float = _DEFAULT_BUDGET,
    max_depth: int = _DEFAULT_DEPTH,
) -> "tuple[int, list[tuple[float, int]]]":
    """Iterative deepening 1, 2, ... up to max_depth, returning the move AND the root scores
    from the SAME depth that produced it (a depth cut short by the clock is discarded whole,
    scores and all — never mixed with a shallower depth's numbers). This is the one search
    both the label and the ranking sentence read from; there is no second tree to disagree
    with it."""
    legal = list(state.legal_actions(side))
    if len(legal) == 1:
        return legal[0], [(0.0, legal[0])]

    stop_at = time.monotonic() + time_budget
    chosen = legal[0]
    scored = [(0.0, m) for m in legal]  # fallback if even depth 1 is clock-cut
    for depth in range(1, max_depth + 1):
        nodes = [0]
        try:
            depth_chosen, depth_scored = _best_at_depth(state, side, depth, stop_at, nodes)
        except _OutOfTime:
            break
        chosen, scored = depth_chosen, depth_scored
        if time.monotonic() > stop_at:
            break
    return chosen, scored


def pick_move(
    state: pyspiel.State,
    player_id: int,
    time_budget: float = _DEFAULT_BUDGET,
    max_depth: int = _DEFAULT_DEPTH,
) -> int:
    """Search depth 1, 2, ... up to max_depth, returning the best move from the deepest
    search that finished inside time_budget (a depth cut short by the clock is discarded)."""
    chosen, _ = _search_root(state, player_id, time_budget, max_depth)
    return chosen


_RANK_WIDTH = 3


def describe_ranking(chosen: int, scored: "list[tuple[float, int]]") -> str:
    """Name the chosen move and the strongest alternatives, ordered by the root scores the
    label's own search already computed (see _search_root / _best_at_depth) — nothing here
    re-searches the position.

    Only action ids are written: clobber labels are coordinates and the values are floats, and
    neither is something the student should be tempted to copy into the tool argument. Every
    integer in the sentence is a legal action id, because `scored` always covers exactly the
    legal moves at the root. The chosen move is sorted to the front by construction (it is the
    argmax of `scored`, ties broken by search order), and is pinned there explicitly below as
    a second, cheap guarantee — not because the two could disagree, but because a guarantee
    that costs three lines is worth keeping even after the disagreement is structurally
    impossible.
    """
    if len(scored) <= 1:
        return 'Only one move is legal here.'
    ranked = sorted(scored, key=lambda t: -t[0])
    rest = [m for _, m in ranked if m != chosen][:_RANK_WIDTH - 1]
    listed = ', '.join([f'{chosen} (best)'] + [str(m) for m in rest])
    return f'Ranking the legal moves by the mobility they leave me: {listed}.'


def _selftest() -> int:
    """Structural invariants for the clobber search.

    UNLIKE the liars / othello / gin self-tests this one NEEDS pyspiel: this module imports it at
    module scope (see the top of the file) and every function below takes a pyspiel.State. So it
    runs on the VPS, not on a bare laptop — that is a property of clobber, not an oversight.

    Deliberately board-AGNOSTIC: it asserts things that must hold on ANY position — antisymmetry,
    permutation, legality — rather than pinning particular moves. Pinning a move would be wrong
    here on purpose: pick_move is CLOCK-bounded, so the same position can legitimately return a
    different move on a busy box (the long note at the top of this file explains why, and why the
    node-budget fix is the real answer). A test that pinned moves would flake exactly when the
    machine is loaded, which is when generation actually runs.
    """
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    check("_flip(0) = 1", _flip(0), 1)
    check("_flip(1) = 0", _flip(1), 0)
    check("_flip is an involution", [_flip(_flip(s)) for s in (0, 1)], [0, 1])

    # The three board shapes the validator's ClobberAgent actually uses.
    for rows, cols in ((4, 5), (5, 5), (5, 6)):
        game = pyspiel.load_game("clobber", {"rows": rows, "columns": cols})
        state = game.new_initial_state()
        side = state.current_player()
        tag = f"{rows}x{cols}"

        # THE PREMISE THE WHOLE EVALUATION RESTS ON, settled from OpenSpiel's source (spiel.h,
        # State::LegalActions(Player)): "if (!IsTerminal() && player == CurrentPlayer()) {...}
        # else { return {}; }" — the OFF-TURN player gets an EMPTY list. clobber.h overrides only
        # the no-arg LegalActions(), never the Player overload, so it inherits that.
        # Why this matters, and why the docs elsewhere are wrong: in Clobber a legal move is one
        # ordered pair (my piece, orthogonally adjacent enemy piece), and that set is in BIJECTION
        # read from either end — so both players have the SAME true move count at every position.
        # A genuine "mobility difference" would therefore be identically ZERO and useless. What
        # _leaf_score actually computes, given the empty off-turn list, is the SIGNED MOBILITY OF
        # THE SIDE TO MOVE (+m when side is the mover, -m when it is not). That is a real and
        # sensible heuristic — the implementation is right and the "mobility-difference" wording
        # at the top of this file and in sft_env_config is what is wrong.
        # Asserted, not printed: if a future OpenSpiel ever returns the real list off-turn, the
        # eval silently collapses to 0.0 everywhere, and this is the line that says so.
        check(f"{tag}: off-turn legal list is empty (spiel.h base behaviour)",
              len(state.legal_actions(_flip(side))), 0)
        check(f"{tag}: leaf = mobility of the side to move",
              _leaf_score(state, side), float(len(state.legal_actions(side))))

        # This one is an algebraic identity (a-b == -(b-a)) and holds for ANY legal_actions, so it
        # proves only that _leaf_score stays a symmetric difference under future edits.
        check(f"{tag}: leaf score antisymmetric", _leaf_score(state, 0), -_leaf_score(state, 1))

        moves = list(state.legal_actions(side))
        ordered = _order_moves(state, side, list(moves))
        # A reordering must never DROP or INVENT a move. Dropping one makes the search silently
        # ignore a legal reply — invisible in play, visible only as quietly worse labels.
        # Compared as a bool, not as two 31-49 element lists, to keep the log readable.
        check(f"{tag}: order_moves is a permutation", sorted(ordered) == sorted(moves), True)

        # The invariant that decides forfeits: an action outside the legal list is an instant loss
        # at eval, no retry, no fallback. Tiny budget/depth so the self-test stays quick.
        chosen = pick_move(state, side, time_budget=0.05, max_depth=2)
        check(f"{tag}: pick_move returns a legal action", chosen in moves, True)

    print("PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


def _mobility_cot(state, player: int, move: int) -> "str | None":
    """Value-trace distillation: narrate the mobility lead the chosen move creates — the exact
    negamax leaf eval the teacher optimises (Clobber is won by the last player able to move, so
    mobility, not material, is the signal). A learnable, deterministic rationale (Stream-of-Search
    2404.03683: distilling the value/trace, not just the move, helps + generalises). Never raises."""
    try:
        nxt = state.clone()
        nxt.apply_action(move)
        mine = len(nxt.legal_actions(player))
        theirs = len(nxt.legal_actions(player ^ 1))
        # State the principle + the grounded fact; do NOT claim this move maximises the 1-ply lead
        # (the negamax decides over several plies, so the 1-ply gap can be negative on a good move).
        return (f"Clobber is won by the last player able to move, so mobility, not material, decides. "
                f"After this move I have {mine} legal moves and the opponent {theirs}.")
    except Exception:
        return None


class ClobberSearchTeacher:
    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        # Sampled once per game (the teacher is constructed per game) for depth/time variety.
        self._time_budget = rng.uniform(*_TIME_BUDGET_RANGE)
        # Gen search depth is a THROUGHPUT knob: measured teacher-step is ~134 ms at depth 14-16,
        # the #1 clobber-generation bottleneck (only ~2.9k games/window vs a ~4.5k target). The move
        # is driven by the mobility leaf-eval, so a shallower search may pick a similar move far
        # faster -> more games -> better state coverage. CLOBBER_MAX_DEPTH overrides (default = the
        # 14-16 sample, unchanged). GATE ON EVAL: a shallower teacher can change move quality.
        _d = os.environ.get('CLOBBER_MAX_DEPTH')
        self._max_depth = int(_d) if (_d and _d.strip().isdigit()) else rng.randint(*_MAX_DEPTH_RANGE)
        # Value-trace CoT is OPT-IN (CLOBBER_COT=1, default OFF → content=None as before).
        # It distils the mobility value the negamax already computes; GATE ON EVAL before enabling.
        self._emit_cot = os.environ.get('CLOBBER_COT', '0').strip() == '1'
        self.last_reason = None  # bare move, content=None (no CoT) unless _emit_cot

    def step(self, state) -> int:
        self.last_reason = None
        try:
            player = state.current_player()
            legal = list(state.legal_actions(player))
        except Exception:
            try:
                legal = list(state.legal_actions())
            except Exception:
                legal = []
        if not legal:
            return 0  # defensive; step is only called on a live turn
        if len(legal) == 1:
            move, scored = legal[0], [(0.0, legal[0])]
        else:
            try:
                move, scored = _search_root(state, player, self._time_budget, self._max_depth)
            except Exception:
                return legal[0]  # never crash generation; the search is robust but stay safe
        # The sentence is built from the scores this same search already produced — see
        # describe_ranking / _search_root docstrings for why there is no second pass to disagree.
        try:
            self.last_reason = describe_ranking(move, scored)
        except Exception:
            self.last_reason = None
        if self._emit_cot and self.last_reason is None:
            self.last_reason = _mobility_cot(state, player, move)
        return move


if __name__ == "__main__":
    raise SystemExit(_selftest())
