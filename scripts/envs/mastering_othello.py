"""Othello teacher — a static piece-square evaluation with a one-line explanation.

The design goes back to the earliest strong Othello programs. Rosenbloom's IAGO (1982)
already leaned on a fixed positional weighting of the board — corners dominant, the
squares touching a corner heavily negative — and Buro's later work on LOGISTELLO showed
that the real strength lives in features such as mobility, stability and parity. We
deliberately keep only the piece-square part of that lineage, because our purpose is not
to play well by ourselves but to produce labels a small policy model can copy.

The policy is three ordered rules over the legal action ids (no search, no board read):
  1. an open corner is taken outright;
  2. otherwise the X-squares, then the C-squares, are set aside while a safer move exists;
  3. among what remains, the cell with the highest static weight is played.

Because the answer depends only on (which squares are legal), a student that sees the board
and the legal set can learn the mapping outright — the label is a function of exactly the
information in its prompt. `.last_reason` holds the sentence for the rule that fired and is
emitted as the assistant turn. Encoding: action = row * 8 + col (a1 = 0 … h8 = 63; 64 is the
forced pass).

Interface: OthelloPositionalTeacher(seed).step(state) -> int. Deterministic; never raises.
That class, which reads nothing but the legal set, remains the default we ship. Alongside it
the file carries an opt-in variant, OthelloCornerAwareTeacher (OTHELLO_TEACHER=positional_corner),
which does consult the board for one fact — whether each corner is occupied — so the X/C
penalty can be lifted once the corner it guards has been decided. Still no search.
"""

_SIDE = 8
_PASS_ACTION = 64  # OpenSpiel's forced-pass action id

# Chain-of-thought length. A long rationale costs tokens on every turn; under a fixed
# per-evaluation time budget that means fewer games played, and in our experience the
# long-winded runs were also the ones that slipped on a weaker base model. Othello and
# goofspiel are the only two of our teachers that carry a rationale at all, so this
# knob lets us compare three settings: "verbose" = the multi-clause explanation that
# names skipped squares (our established setting), "terse" = just the move, "none" =
# no rationale (an assistant turn with no content). We keep "verbose" by default and
# treat the other two as an A/B to be run on more than one base model before switching.
OTHELLO_COT_MODE = "verbose"  # "verbose" | "terse" | "none"


def _row_col(action_id: int) -> "tuple[int, int]":
    return divmod(action_id, _SIDE)


def _cells_to_ids(cells) -> "frozenset[int]":
    return frozenset(r * _SIDE + c for r, c in cells)


def _label(action_id: int) -> str:
    """Square name in algebraic style ('e6': file letter then rank digit) for an action id.

    We pinned this to the OpenSpiel source rather than to our own convention, because the
    rationale text has to name squares the same way the game's own action_to_string does.
    From open_spiel/games/othello/othello.{h,cc} on master:
        Move(int move) : row_(move / kNumCols), col_(move % kNumCols)   -> divmod(a, 8), row-major
        GetAction() { return row_ * kNumCols + col_; }                  -> what _cells_to_ids builds
        kPassMove = kNumCells                                           -> 64, our _PASS_ACTION
        Move::ToString() = ColumnString(col_) + RowString(row_)
          where ColumnString(c) = "abcdefgh"[c] and RowString(r) = 1 + r
    Hence the labels here are the same bytes pyspiel would print. Should upstream ever change
    its convention, the _label anchors in _selftest are the tripwire.
    """
    row, col = _row_col(action_id)
    return chr(ord("a") + col) + str(row + 1)


# Corner theory, encoded as geometry. The corners are listed once and every "danger"
# square is derived from them, so no action id is written out by hand.
_CORNER_CELLS = ((0, 0), (0, 7), (7, 0), (7, 7))
_CORNERS = _cells_to_ids(_CORNER_CELLS)
# X-squares: the diagonal neighbour of each corner (b2, g2, b7, g7).
_X_SQUARES = _cells_to_ids(
    (r + dr, c + dc)
    for (r, c) in _CORNER_CELLS
    for dr in (1, -1)
    for dc in (1, -1)
    if 0 <= r + dr < _SIDE and 0 <= c + dc < _SIDE
)
# C-squares: the two edge squares orthogonally next to each corner.
_C_SQUARES = _cells_to_ids(
    (r + dr, c + dc)
    for (r, c) in _CORNER_CELLS
    for (dr, dc) in ((0, 1), (0, -1), (1, 0), (-1, 0))
    if 0 <= r + dr < _SIDE and 0 <= c + dc < _SIDE
)

# Danger square -> the corner it endangers. Only the opt-in corner-aware variant uses this.
# It walks the same king-step offsets with the same bounds filter as _X_SQUARES/_C_SQUARES,
# but it is its own comprehension, so nothing structurally stops it drifting from them.
# The invariant we rely on is a bijection: set(_DANGER_CORNER) == _X_SQUARES | _C_SQUARES,
# each key exactly one king-step from its corner. _selftest asserts it; keep that in step
# with any edit to the three definitions.
_DANGER_CORNER = {
    (r + dr) * _SIDE + (c + dc): r * _SIDE + c
    for (r, c) in _CORNER_CELLS
    for (dr, dc) in ((1, 1), (1, -1), (-1, 1), (-1, -1),
                     (0, 1), (0, -1), (1, 0), (-1, 0))
    if 0 <= r + dr < _SIDE and 0 <= c + dc < _SIDE
}

# The piece-square table in the IAGO tradition. It is symmetric under the board's four
# reflections, so one 4x4 quadrant suffices and a lookup folds (row, col) into it.
# Weights: corner 100, edge 10, one-in-from-edge 5, C-square -20, X-square -50,
# interior 1..5. Only the ordering matters to the policy, not the magnitudes.
_QUADRANT = (
    (100, -20, 10, 5),
    (-20, -50, -2, -2),
    ( 10,  -2,  5, 1),
    (  5,  -2,  1, 1),
)


def _weight(action_id: int) -> int:
    row, col = _row_col(action_id)
    return _QUADRANT[min(row, _SIDE - 1 - row)][min(col, _SIDE - 1 - col)]


def _heuristic_choose(
    legal_action_ids: "list[int]", relaxed: "frozenset[int]" = frozenset()
) -> "tuple[int, str]":
    """Run the three-rule static evaluation -> (action_id, rationale).

    `relaxed` lists danger squares whose corner has already been occupied, so their
    penalty no longer applies. The default (empty) is the board-blind teacher exactly.
    """
    if legal_action_ids == [_PASS_ACTION]:
        return _PASS_ACTION, "No board moves available — forced pass."

    board_actions = [a for a in legal_action_ids if a != _PASS_ACTION]

    # Rule 1: a corner can never be flipped, so an available one is always taken.
    for a in board_actions:
        if a in _CORNERS:
            return a, f"Corner {_label(a)} is available — taking it."

    # Rule 2: veto the X-squares, then the C-squares, as long as something else is legal.
    # With `relaxed` empty (the default path) active_* are just the full X/C sets, so this
    # block is the unmodified board-blind rule; only the corner-aware variant passes more.
    active_x = _X_SQUARES - relaxed
    active_c = _C_SQUARES - relaxed
    x_free = [a for a in board_actions if a not in active_x]
    candidates = [a for a in x_free if a not in active_c] or x_free or list(board_actions)
    skipped_x = [a for a in board_actions if a in active_x]
    skipped_c = [a for a in board_actions if a in active_c and a not in active_x]

    # Rule 3: best static weight among whatever survived.
    chosen = max(candidates, key=_weight)
    chosen_lbl = _label(chosen)

    # Nothing but danger squares was legal: say so rather than pretend it was a choice.
    if chosen in active_x:
        return chosen, f"No safe moves — forced to play X-square {chosen_lbl}."
    if chosen in active_c:
        return chosen, f"No X-free safe moves — forced to play C-square {chosen_lbl}."

    # Ordinary turn: list what we declined, then the move.
    parts = []
    if skipped_x:
        parts.append(f"Avoid X-squares ({', '.join(_label(a) for a in skipped_x)}).")
    if skipped_c:
        parts.append(f"Avoid C-squares ({', '.join(_label(a) for a in skipped_c)}).")
    parts.append(f"Play {chosen_lbl}.")
    return chosen, " ".join(parts)


class OthelloPositionalTeacher:
    def __init__(self, seed: int = 0):
        self._seed = seed  # unused: the policy is deterministic; kept so all teachers share a signature
        self.last_reason = None

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
            return _PASS_ACTION  # should not happen on a live turn; pass is the harmless answer
        action_id, reasoning = _heuristic_choose(legal)
        if OTHELLO_COT_MODE == "none":
            self.last_reason = None
        elif OTHELLO_COT_MODE == "terse":
            self.last_reason = f"Play {_label(action_id)}."
        else:
            self.last_reason = reasoning
        return action_id


def _parse_board(obs: str) -> "list[str] | None":
    """Pull the eight board rows out of OpenSpiel's othello ObservationString.

    We took the layout from the C++ rather than guessing it. OthelloState::ObservationString
    is just `return ToString();`, and ToString emits, in order:
        header    : IsTerminal() ? "Terminal State:\\n"
                                 : StrCat(PlayerToString(CurrentPlayer()), " to play:\\n")
        col labels: "  a b c d e f g h  " + "\\n"
        per row r : RowString(r) + " " + (StateToString(BoardAt(r,c)) + " ") x8
                    + RowString(r) + "\\n"
        col labels again
    with RowString(r) = str(1+r) and StateToString giving "-" empty / "x" black / "o" white.
    So board[0] is rank 1 and board[r][c] is (row, col) — the same row-major layout that
    _cells_to_ids uses (r*_SIDE+c), which is what lets _settled_danger index a corner
    directly.

    The eight row lines are the only lines that both begin with a digit and contain exactly
    eight cell glyphs: the header begins with a letter and the column-label line has none of
    "-xo". That is enough to avoid a regex (and this module has no imports at all). Returns
    None unless exactly eight rows were found; callers treat None as "behave board-blind",
    and step() also checks orientation before believing the result.
    """
    rows = []
    for line in obs.splitlines():
        s = line.strip()
        if not s or not s[0].isdigit():
            continue
        cells = [ch for ch in s if ch in "-xo"]
        if len(cells) == _SIDE:
            rows.append("".join(cells))
    return rows if len(rows) == _SIDE else None


def _settled_danger(board: "list[str]") -> "frozenset[int]":
    """Danger squares whose corner has already been taken by somebody.

    The whole reason an X- or C-square is bad is that occupying it can hand the adjacent
    corner to the opponent. A corner disc is permanently stable — flipping needs the disc
    bracketed on a line, and no line through a corner continues past it — so once a corner
    is occupied by either side that risk is gone for good. That is why the test is `!= "-"`
    and not "is ours".

    What this does not say: b2 beside the opponent's a1 is still a poor square (it feeds
    their diagonal). It is merely no longer catastrophic. The relaxation cannot promote it
    into a good square, because _QUADRANT is untouched and a relaxed square keeps its
    -50/-20, the two lowest weights on the board. Relaxation only lifts a veto; a relaxed
    square wins `max(candidates, key=_weight)` only when every competing candidate is also
    a danger square. Where a safe move exists the choice is unchanged; where only bad squares
    remain we now prefer the one whose corner is already decided over one that still risks a
    live corner. The "not empty" test matters for exactly that case: with a1 gone to the
    opponent and only {b2, g7} legal, an "is ours" test would keep vetoing b2 and leave the
    tie to list order, which can give up h8.

    Coarse in one direction on purpose: when the corner is ours, the X-square is genuinely
    good (it extends a stable anchor), yet it still carries -50 and will rarely be chosen.
    Fixing that needs a colour-dependent table, which would cost the colour-independence
    below and widen the change well beyond an opt-in experiment.

    Colour-independent: we never need to know which seat we occupy, which removes a whole
    class of colour-mapping mistakes.
    """
    return frozenset(
        danger
        for danger, corner in _DANGER_CORNER.items()
        if board[corner // _SIDE][corner % _SIDE] != "-"
    )


class OthelloCornerAwareTeacher(OthelloPositionalTeacher):
    """Opt-in variant (OTHELLO_TEACHER=positional_corner). Not the default.

    Motivation. The default teacher touches `state` only for current_player() and
    legal_actions(); it never reads the board. The student, however, is shown the whole
    board in its prompt. That mismatch runs the wrong way: the label was chosen from the
    legal set alone, so the board is noise as far as the label is concerned and the
    student is taught to ignore it. In a game where the field of opponents varies widely
    in strength, we would rather the student learn something the board actually says.

    The change. One rule only, and it makes the evaluation more correct rather than
    deeper: the X/C-square penalty applies only while the neighbouring corner is still
    empty. That is standard corner theory — the X-square is bad because it concedes the
    corner, so once the corner is settled the penalty is simply false. Corner grab, the
    static table, tie-breaks and the wording of the rationale are all untouched.

    Why it should stay learnable. The static table exists because search-based teachers
    (an MCTS player, and a one-ply mobility look-ahead) both failed to distil into the
    student; their labels were not a simple function of what the student sees. This adds
    no search: the label remains a function of (board, legal set), and the single board
    fact consulted — is this corner occupied — is one visible cell. If anything the
    conditional rule is easier to fit, because the unconditional one is visibly wrong in
    positions the student can inspect.

    Gate. Opt-in because it is unmeasured, and our rule is to gate on local student
    head-to-head, never on how strong the teacher looks. Any trouble reading the board
    degrades silently to the default behaviour (relaxed = empty set).
    """

    def step(self, state) -> int:
        self.last_reason = None
        player = 0
        try:
            player = state.current_player()
            legal = list(state.legal_actions(player))
        except Exception:
            try:
                legal = list(state.legal_actions())
            except Exception:
                legal = []
        if not legal:
            return _PASS_ACTION  # should not happen on a live turn; pass is the harmless answer
        relaxed = frozenset()
        try:
            board = _parse_board(state.observation_string(player))
            # Orientation check. _parse_board only validates shape (8 x 8); a row-flipped or
            # transposed rendering still yields eight tidy rows, and we would then read a1's
            # occupancy from a8's cell and relax the wrong corners without any error. Every
            # legal move must land on an empty cell, so one pass over `legal` catches a flip,
            # a transposition and a changed empty glyph. Filtering _PASS_ACTION is essential:
            # 64 // 8 == 8 would index past the board.
            if board is not None and all(
                board[a // _SIDE][a % _SIDE] == "-"
                for a in legal
                if a != _PASS_ACTION
            ):
                relaxed = _settled_danger(board)
        except Exception:
            relaxed = frozenset()  # cannot read the board -> behave exactly like the default
        action_id, reasoning = _heuristic_choose(legal, relaxed)
        if OTHELLO_COT_MODE == "none":
            self.last_reason = None
        elif OTHELLO_COT_MODE == "terse":
            self.last_reason = f"Play {_label(action_id)}."
        else:
            self.last_reason = reasoning
        return action_id


def _selftest() -> int:
    """Deterministic checks on the geometry and the three-rule chooser. No pyspiel required.

    Othello punishes a broken teacher harder than most games — a single systematic blunder
    (a wrong square label, a mis-indexed corner) loses whole matches, not single moves — so
    we want a boring, reliable teacher and a test that proves it stays that way.
    """
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    # Naming: file letter from the column, rank digit from row + 1.
    for aid, want in ((0, "a1"), (7, "h1"), (56, "a8"), (63, "h8")):
        check(f"_label({aid})", _label(aid), want)

    # The square sets are computed, not written out, so pin where they land.
    check("corners", set(_CORNERS), {0, 7, 56, 63})
    check("X-squares (one diagonal step inward)", set(_X_SQUARES), {9, 14, 49, 54})
    check("C-squares (two orthogonal per corner)", set(_C_SQUARES), {1, 6, 8, 15, 48, 55, 57, 62})
    check("corners/X disjoint", _CORNERS & _X_SQUARES, frozenset())
    check("corners/C disjoint", _CORNERS & _C_SQUARES, frozenset())
    check("X/C disjoint", _X_SQUARES & _C_SQUARES, frozenset())

    # The bijection the _DANGER_CORNER comment promises. It is a separate comprehension and
    # could drift from _X_SQUARES/_C_SQUARES without anything complaining, so enforce it.
    check("_DANGER_CORNER keys == X | C", set(_DANGER_CORNER), set(_X_SQUARES | _C_SQUARES))
    king_step = all(
        max(abs(d // _SIDE - c // _SIDE), abs(d % _SIDE - c % _SIDE)) == 1
        for d, c in _DANGER_CORNER.items())
    check("every danger square is ONE king-step from its corner", king_step, True)
    check("_DANGER_CORNER values are all corners", set(_DANGER_CORNER.values()) <= set(_CORNERS), True)

    # Piece-square table: corners on top, X-squares at the bottom, symmetric.
    check("weight(corner a1) = 100", _weight(0), 100)
    check("weight(X-square b2) = -50", _weight(9), -50)
    check("weight(C-square b1) = -20", _weight(1), -20)
    check("all four corners weigh the same", len({_weight(a) for a in (0, 7, 56, 63)}), 1)
    check("every corner outranks every X-square",
          min(_weight(a) for a in _CORNERS) > max(_weight(a) for a in _X_SQUARES), True)

    # Pass lies off the board and must never reach the geometry helpers. _row_col(64) is
    # (8, 0); _weight would fold min(8, 7 - 8) = -1 and _QUADRANT[-1] wraps to the last row,
    # returning a plausible number instead of raising. That quiet wrap is why the pass filter
    # in _heuristic_choose matters, so assert the premise it rests on.
    check("_row_col(PASS) is off-board", _row_col(_PASS_ACTION)[0] >= _SIDE, True)
    check("PASS belongs to no board set", _PASS_ACTION in (_CORNERS | _X_SQUARES | _C_SQUARES), False)

    # Chooser invariants. The legality loop guards structure, not a live bug: every return
    # path already yields a member of board_actions (a filtered copy of the input) or the
    # literal pass on the branch guarded by legal == [_PASS_ACTION]. It is here for whoever
    # rewrites this next — an illegal action is an immediate forfeit at play time, with no
    # retry.
    for legal in ([_PASS_ACTION], [9, 0], [9, 20], [1, 9, 20, 63], [20], [9]):
        check(f"choice from {legal} is legal", _heuristic_choose(list(legal))[0] in legal, True)
    check("takes the corner when offered", _heuristic_choose([9, 0])[0], 0)
    check("prefers a quiet square over an X-square", _heuristic_choose([9, 20])[0], 20)
    check("pass-only stays pass", _heuristic_choose([_PASS_ACTION])[0], _PASS_ACTION)
    check("X-square taken when it is the ONLY move", _heuristic_choose([9])[0], 9)

    # Everything above covers the stable geometry and the board-blind path. The corner-aware
    # additions — _parse_board, _settled_danger and the `relaxed` branch — are pure functions
    # too, so they get their own cases.
    empty = ["-" * _SIDE] * _SIDE
    check("_parse_board rejects a non-board", _parse_board("garbage\nno rows here\n"), None)
    _obs = ("Black to play:\n  a b c d e f g h  \n"
            + "".join(f"{r + 1} " + " ".join(empty[r]) + f" {r + 1}\n" for r in range(_SIDE))
            + "  a b c d e f g h  \n")
    check("_parse_board reads 8 rows", _parse_board(_obs), empty)
    check("empty board settles nothing", _settled_danger(empty), frozenset())
    check("a1 taken settles {1,8,9}",
          _settled_danger(["x" + "-" * 7] + ["-" * _SIDE] * 7), frozenset({1, 8, 9}))
    # Row-major, not transposed: h1 is id 7, i.e. board[0][7]. A transposed lookup reads
    # board[7][0], sees "-", and settles nothing — the silent failure described in
    # _settled_danger — and the a1 case cannot expose it because that board is symmetric
    # under transposition.
    check("h1 taken settles {6,14,15}",
          _settled_danger(["-" * 7 + "o"] + ["-" * _SIDE] * 7), frozenset({6, 14, 15}))
    # The central claim: relaxation lifts a veto and never promotes.
    check("relaxed X-square is still not promoted over a quiet square",
          _heuristic_choose([9, 20], frozenset({1, 8, 9}))[0], 20)
    check("among bad squares only, the settled one wins",
          _heuristic_choose([9, 54], frozenset({1, 8, 9}))[0], 9)

    print("PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
