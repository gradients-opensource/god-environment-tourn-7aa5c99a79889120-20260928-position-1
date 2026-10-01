"""Leduc hand-win teacher — best-response that maximises P(win the hand), NOT chip-EV.

The validator scores leduc by hand-win (determine_outcome sign-tests the terminal chip
stack; the margin is discarded — core/pvp/scoring.py). The opponent plays a near-Nash
CFR+ style blueprint that optimises chip-EV. Under the hand-win metric these objectives
DIVERGE: a near-Nash is ~unexploitable in chips (chip-BR ±0.086) yet HUGELY exploitable
in hand-win — because folding is dominated (a fold locks in a lost hand), so the best-
response bets to harvest the opponent's folds and calls down to win showdowns.

Offline-measured ceiling (exact infoset-respecting best-response vs a CFR+ near-Nash;
method validated: chip-BR sum ≈ exploitability ≈ 0, Nash-vs-Nash baseline = 0.500):
    hand-win-BR = 0.709 (p0 0.710 / p1 0.708) — well above the 0.60 emission gate.

So a CFR teacher is a BLUNDER here (it teaches Nash → ~0.500 wash vs a near-Nash
opponent). This teacher instead distils the hand-win best-response. The BR policy is
precomputed once (CFR+ Nash → win-utility best-response over the ~small leduc tree) and
cached.

SHIPPED DEFAULT — an earlier version of this line read "GATED OFF by default", which is
false: generate_pvp_toolcalling._make_teacher uses os.environ.get('LEDUC_TEACHER',
'handwin'), and sft_env_config.TOURNAMENT_TEACHERS passes ${LEDUC_TEACHER:-handwin}. Both
paths default to this teacher, so it is what actually trains.

⚠ WHAT 0.709 DOES AND DOES NOT SAY (diagnosis 2026-08-05)
It is the exploitability of a CFR+ Nash BLUEPRINT under hand-win utility — a teacher-vs-
fixed-bot benchmark. PvP never faces that blueprint; it faces another miner's fine-tuned
model. A best response is optimal against exactly ONE opponent and carries no guarantee
against any other, and this one is DETERMINISTIC (pure table lookup, randomised only by the
fallback in step()), which is the most counter-exploitable shape a policy can take in an
imperfect-information game. sft_env_config's own rule — a wash may only be declared when
ALL miners land on the same number, never from a teacher-vs-bot benchmark — applies here
too: 0.709 is exactly such a benchmark.

One mechanism explains BOTH tournament readings. Folding is dominated under hand-win, so
miners converge on the same never-fold/call-down shape; hands then reach showdown and the
result is card-determined -> 0.500 for everyone, which is the Aug-3 R2 observation (0.500
in all four matchups over 296 games). Let ONE player deviate to a counter-strategy and the
separation is large: the Aug-3 R3 boss round went 16-48 over 74 games. So "leduc is a
structural wash" was a POPULATION artifact, not a property of the game — which is why that
label was withdrawn in sft_env_config.

LIMITS, so nobody overreads this: the never-fold characterisation comes from the argument
above and from the CFR+/BR construction, NOT from inspecting the built table; and the R3
loser was another miner's model, so neither datum measures OUR student. The honest status
of leduc is UN-EVALUATED, and the open question is robustness (would a mixed or
Nash-anchored teacher give up ceiling for counter-exploitation resistance?), which only a
student head-to-head can answer.

CACHE: .expert_cache/ is gitignored, so a fresh container has no pickle and build_table()
runs 2000 CFR+ iterations plus a two-player best-response on first use, then caches to
disk. Construction happens inside play_game, which the caller wraps in a per-game
try/except, so a build failure skips one game rather than killing the run.
"""
from __future__ import annotations

import os
import pickle
import random
import re

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".expert_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "leduc_handwin.pkl")
_ITERATIONS = 2000


def _win_util(r: float) -> float:
    return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)


def _win_best_response_cls():
    """Subclass OpenSpiel's (correct, infoset-respecting) best-response, swapping ONLY
    the terminal utility from chip-return to the win-indicator (1/0.5/0)."""
    from open_spiel.python.algorithms import best_response

    class WinBestResponse(best_response.BestResponsePolicy):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._wcache = {}

        def value(self, state):
            key = state.history_str()
            c = self._wcache.get(key)
            if c is not None:
                return c
            if state.is_terminal():
                v = _win_util(state.player_return(self._player_id))
            elif (state.current_player() == self._player_id or
                  state.is_simultaneous_node()):
                action = self.best_response_action(
                    state.information_state_string(self._player_id))
                v = self.q_value(state, action)
            else:
                v = sum(p * self.q_value(state, a)
                        for a, p in self.transitions(state)
                        if p > self._cut_threshold)
            self._wcache[key] = v
            return v

    return WinBestResponse


def build_table(iterations: int = _ITERATIONS) -> dict:
    """infoset_string -> best hand-win action, precomputed once and cached."""
    import pyspiel
    from open_spiel.python.algorithms import cfr

    game = pyspiel.load_game("leduc_poker")
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    nash = solver.average_policy()

    WinBestResponse = _win_best_response_cls()
    table: dict[str, int] = {}
    for p in (0, 1):
        br = WinBestResponse(game, p, nash)
        _before = len(table)
        for infostate in br.infosets:
            table[infostate] = int(br.best_response_action(infostate))
        # Both players write into ONE shared dict, and they CANNOT collide — verified 2026-08-05
        # against upstream rather than argued from parity. open_spiel/games/leduc_poker/
        # leduc_poker.cc, the info-state observer's StringFrom, emits
        #     "[Observer: <player>][Private: <card>][Round r][Player: <cur>][Pot: ...]
        #      [Money: ...][Public: ...][Round1: ...][Round2: ...]"
        # so the OBSERVING PLAYER is the FIRST field of every key. That is a stronger guarantee
        # than the betting-sequence parity argument and holds regardless of it.
        # The same source settles the lookup this whole teacher depends on: open_spiel/python/
        # algorithms/best_response.py builds br.infosets as
        #     infosets[s.information_state_string(self._player_id)]
        # i.e. exactly the call LeducHandWinBot.step makes at runtime. Had those two string
        # formats differed, EVERY table lookup would miss, .get() would return None, and the
        # teacher would silently fall back to rng.choice(legal) — playing random leduc for an
        # entire run with nothing in any log to say so.
        # Kept as an assert anyway: a silent overwrite would HALVE the table while every surviving
        # value stayed individually correct, and no row-count floor can tell that apart from a
        # healthy build. This is the check the __main__ count floor cannot make.
        if len(table) - _before != len(br.infosets):
            raise RuntimeError(
                f'leduc infoset key collision: player {p} contributed {len(br.infosets)} infosets '
                f'but only {len(table) - _before} new keys landed in the shared table')

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "wb") as fh:
        pickle.dump({"version": 1, "iterations": iterations, "table": table}, fh)
    return table


def _load_or_build() -> dict:
    try:
        with open(_CACHE_PATH, "rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict) and data.get("version") == 1:
            return data["table"]
    except (OSError, pickle.UnpicklingError, EOFError):
        pass
    return build_table()


_RANK_NAMES = ('Jack', 'Queen', 'King')
_INFO_PRIVATE = re.compile(r'\[Private: (\d+)\]')
_INFO_ROUND = re.compile(r'\[Round (\d+)\]')
_INFO_PUBLIC = re.compile(r'\[Public: (-?\d+)\]')


def describe_decision(info: str, label: str) -> str:
    """One-sentence account of a table decision, written for the student.

    The table is opaque -- an infoset string maps to an action id and nothing else -- so a row
    built from it shows the student a move with no visible reason. This sentence supplies the
    reason from the two things the policy actually conditions on: the private card's rank and,
    in the second round, whether it pairs the board card. It states the action as "the hand-win
    line", which is exactly what the table is (a best response under the win indicator), and it
    claims nothing the table does not compute. No bare number appears in the text: the pot and
    money figures are integers the student could copy into the tool argument, so they are left
    out and only the rank word is spoken.
    """
    m = _INFO_PRIVATE.search(info)
    rank = _RANK_NAMES[int(m.group(1)) // 2] if m and int(m.group(1)) // 2 < 3 else 'card'
    r = _INFO_ROUND.search(info)
    verb = label.strip().lower() or 'act'
    if r and r.group(1) == '2':
        pub = _INFO_PUBLIC.search(info)
        if pub and int(pub.group(1)) >= 0:
            board = _RANK_NAMES[int(pub.group(1)) // 2] if int(pub.group(1)) // 2 < 3 else 'board card'
            pairs = 'pairs' if board == rank else 'does not pair'
            return f'Second round: my {rank} {pairs} the {board} on the board; the hand-win line is to {verb}.'
        return f'Second round holding the {rank}; the hand-win line is to {verb}.'
    return f'First betting round holding the {rank}; the hand-win line is to {verb}.'


class LeducHandWinBot:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._table = _load_or_build()
        self.last_reason = None

    def step(self, state):
        self.last_reason = None
        player = state.current_player()
        legal = state.legal_actions(player)
        info = state.information_state_string(player)
        a = self._table.get(info)
        if a is None or a not in legal:
            a = self._rng.choice(legal)
        else:
            # Only a table decision gets a rationale; the random fallback has none to give. The
            # sentence is derived after the action is fixed, so it can never change the move.
            try:
                self.last_reason = describe_decision(info, state.action_to_string(player, a))
            except Exception:
                self.last_reason = None
        return a


if __name__ == "__main__":
    # Self-test: rebuild the table and confirm it reproduces the ~0.709 hand-win ceiling.
    import pyspiel
    from open_spiel.python.algorithms import cfr

    game = pyspiel.load_game("leduc_poker")
    solver = cfr.CFRPlusSolver(game)
    for _ in range(_ITERATIONS):
        solver.evaluate_and_update_policy()
    nash = solver.average_policy()
    WBR = _win_best_response_cls()
    vals = [WBR(game, p, nash).value(game.new_initial_state()) for p in (0, 1)]
    print(f"hand-win-BR: p0={vals[0]:.4f} p1={vals[1]:.4f} avg={sum(vals)/2:.4f} "
          f"(expect ~0.709)")
    # GATE — this block is invoked by the first-hour verification queue, so it MUST be able to
    # fail. Until 2026-08-05 it only PRINTED "expect ~0.709" and exited 0 whatever came out, so a
    # broken solve would have been reported as a clean pass while every leduc label was wrong.
    #
    # Checked BEFORE build_table() on purpose. That call writes a COMPLETE, version-1-valid pickle
    # to _CACHE_PATH, and _load_or_build validates nothing but the version tag — so failing after
    # it would print [FAIL] and still leave the rejected table on disk for every later run to load
    # (LeducHandWinBot is constructed once per GAME, not once per run). A complete-but-wrong cache
    # is worse than a missing one.
    avg = sum(vals) / len(vals)
    if not 0.65 <= avg <= 0.75:
        # Read the numbers right: this module measures chip-BR at ~0 (exploitability ~0), so a
        # chip-return utility shows up near 0.0, NOT 0.5. ~0.500 is the Nash-vs-Nash hand-win
        # baseline — i.e. the best response is not best-responding at all.
        print(f"  [FAIL] hand-win-BR avg={avg:.4f} outside [0.65, 0.75]; measured ceiling is 0.709 "
              f"(p0 0.710 / p1 0.708). ~0.500 = not best-responding; ~0.0 = utility is chip-return "
              f"instead of the win indicator. Cache NOT written.")
        raise SystemExit(1)

    tbl = build_table()
    print(f"table entries: {len(tbl)} (cached to {_CACHE_PATH})")
    if len(tbl) < 100:
        print(f"  [FAIL] table has only {len(tbl)} entries; leduc is a ~936 information-set game, "
              f"so the traversal stopped early. Removing the rejected cache so no later run loads it.")
        try:
            os.remove(_CACHE_PATH)
        except OSError:
            pass
        raise SystemExit(1)
    print("  [OK] leduc self-test: BR in range, table populated")
