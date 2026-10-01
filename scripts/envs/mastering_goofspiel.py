"""Goofspiel teachers — deterministic policies with a short chain-of-thought.

The default we ship is GoofspielOneUpBot (bottom of file). GoofspielValueMatchBot, the
policy it replaced, stays available through GOOFSPIEL_TEACHER=valuematch so the two can be
compared. The notes below explain why we moved, and why the move is an exploit rather than
a claim about optimal play.

What the literature says. Goofspiel (the "game of pure strategy") has been studied since
Ross (1971), who worked out the optimal play against a fully random opponent and showed
that the natural "bid the prize" policy is far from equilibrium. Rhoads and Bartholdi (2012)
computed equilibrium strategies for the 13-card game exactly and found them heavily mixed:
at equilibrium a player randomises over several bids in most states. That has an awkward
consequence for us. A mixed strategy is only safe when it is SAMPLED; play its mode
deterministically and it becomes an ordinary exploitable pure policy. And a sampled policy
is useless as a teacher, because the student sees the same information state labelled with
different "correct" bids and learns noise — the same failure we met with a search-based
teacher in othello. So equilibrium play is not the goal. We want a pure, simple rule that
does well against the pure, simple rules other people are likely to field.

Why value-matching looked right. "Bid the card closest to the prize" is the policy every
introduction to the game reaches for. It is deterministic, it is a function of (prize,
hand) only, it self-balances spend, and against a search-based opponent it is respectable.
We shipped it on that basis and it played poorly against a field of other distilled
policies, and the reason is elementary: because the prize card is always still in hand (by
induction), value-matching bids exactly the prize on every round. A rule that bids one
above the prize therefore wins every contested round at one card's overspend and concedes
only the single dearest prize, and 1 + 2 + ... + (N-1) > N for every N >= 4. That counter is
the first thing anyone tries, and it is total: value-match scores zero against it at every
deck size we play (N in 5, 8, 10, 13).

Why one-up. In a matrix of the obvious pure policies (value-match, one-up, rank-based
one-up, two-up) there is no dominant member — they form a cycle, much as Rhoads and
Bartholdi describe — but one-up has the best worst case and, under fictitious play
(Robinson 1951) over that matrix, sits at the equilibrium of the meta-game while
value-match sits near the bottom. The edge also survives noise: with both sides bidding
at random some fraction of the time, one-up keeps the best worst case at every noise
level we tried and still beats value-match clearly. None of this says one-up is optimal in
Goofspiel; it says one-up is the hardest of the simple policies to exploit and the best
exploiter of the most common one.

Learnability is preserved, which is the constraint that produced value-match in the first
place. One-up is a strict function of (prize, hand) and is a simpler sentence than
"closest by absolute value". A sharper lesson came out of the same study: determinism is
necessary but not sufficient. A hash-mixed policy scored well in the matrix and is
perfectly reproducible, yet no student can fit a hash function — that is the sampled-
teacher disease wearing a deterministic mask. We rejected it for that reason, not its score.
A teacher must be a simple function of visible features, not merely repeatable.

Caveat, stated plainly: the policy study used our own re-implementation of the rules, not
pyspiel, which is fine for a combinatorial result that does not depend on observation
formatting. The student's performance with the new teacher is gated on local student
head-to-head, as always; a strong teacher is not by itself the objective.

Contract (shared by all our game teachers): constructed with a seed, exposes
``step(state) -> int`` (pyspiel action id == card value − 1), and sets ``.last_reason``
after each step so generate_pvp_toolcalling.py can place the rationale in the example
content. Deterministic (the seed is kept only so every teacher has the same signature).
Never raises: any parse miss falls through to a no-crash floor over the legal action ids.
Reads the current prize and our hand from observation_string — public information only.
"""
from __future__ import annotations

import random
import re

_RE_PRIZE = re.compile(r"[Pp]oint card:\s*(\d+)")
# A legal-action line reads "<id> -> [Pn]Bid: <value>"; when absent we fall back to value = id + 1.
_RE_ACTION_VALUE = re.compile(r"^\s*(\d+)\s*->.*?Bid:\s*(\d+)", re.IGNORECASE)

# Chain-of-thought length (same knob and same reasoning as mastering_othello).
# "verbose" = the full spending-balance explanation, "terse" = just the bid,
# "none" = no rationale. Compare across more than one base model before changing it.
GOOFSPIEL_COT_MODE = "terse"  # "verbose" | "terse" | "none"


def _cot(verbose: str, terse: str) -> "str | None":
    if GOOFSPIEL_COT_MODE == "none":
        return None
    return terse if GOOFSPIEL_COT_MODE == "terse" else verbose


class GoofspielValueMatchBot:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)  # signature parity only; the policy never samples
        self.last_reason = None

    def _action_values(self, obs: str, legal: list[int]) -> dict:
        """Map each legal action id to the card value it bids. Take the value from the
        legal-action text when present; otherwise value = id + 1, since goofspiel action
        ids are 0-based card indices."""
        found = {}
        for line in obs.splitlines():
            m = _RE_ACTION_VALUE.match(line)
            if m:
                found[int(m.group(1))] = int(m.group(2))
        return {a: found.get(a, a + 1) for a in legal}

    def step(self, state):
        self.last_reason = None
        player = state.current_player()
        legal = list(state.legal_actions(player))
        if not legal:
            return None
        if len(legal) == 1:
            self.last_reason = _cot("Only one card left — I bid it.", "Bid the last card.")
            return legal[0]

        try:
            obs = state.observation_string(player)
        except Exception:
            obs = ""
        m = _RE_PRIZE.search(obs)
        if not m:
            # Prize not readable (should not occur mid-round): match against 0 on id+1,
            # i.e. the lowest card, rather than crash.
            a = min(legal, key=lambda x: (abs((x + 1) - 0), -(x + 1)))
            self.last_reason = None
            return a

        prize = int(m.group(1))
        id_to_val = self._action_values(obs, legal)
        # Bid the card nearest the prize in value; if one card sits just above and one just
        # below, prefer the higher (the -val tie-break) so a big prize meets a big card.
        # Same state, same bid.
        pick = min(legal, key=lambda a: (abs(id_to_val[a] - prize), -id_to_val[a]))
        card = id_to_val[pick]
        self.last_reason = _cot(
            f"The prize this round is worth {prize}. I bid the card closest to it — "
            f"card {card} — spending high cards on high prizes and low cards on low "
            f"prizes so my spending stays balanced across the game.",
            f"Prize {prize}; bid card {card}.",
        )
        return pick


# Instrumentation for the one failure in this file that would make no noise.
#
# Should _RE_PRIZE stop matching (upstream renames the field, changes the spacing, moves it
# out of observation_string), step() falls through to the parent's floor, which bids the
# LOWEST legal card on every turn. Nothing raises, so the per-game try/except in
# generate_pvp_toolcalling never trips, and the log looks like healthy play. We would be
# training on the worst possible goofspiel and learn of it only from results.
#
# It matters more than usual because there is no way to revert after submission: the
# submission record is repo + branch + commit + datasets, with no environment-variable
# field, so GOOFSPIEL_TEACHER cannot be flipped afterwards (it is a container-side default
# only). Counting and printing is the one defence left.
#
# Reported at powers of two: a systematic miss is loud within the first few games
# (1, 2, 4, 8, ...) while a rare one stays quiet. `n & (n - 1) == 0` is the power-of-two
# test; n is at least 1 by then, so n == 0 never reaches it.
_PRIZE_PARSE_STATS = {"total": 0, "fallback": 0}


class GoofspielOneUpBot(GoofspielValueMatchBot):
    """The goofspiel teacher we ship (see the module docstring for the reasoning).

    The whole rule: bid the CHEAPEST card that still beats the prize value; if no card
    beats it, give the round away with the lowest card. That sentence is the entire
    policy — a strict function of (prize, hand), so one information state always maps to
    one bid and a small student can reproduce it at inference.

    Why it dominates a value-matcher (zero for them at every deck size): a value-matcher
    bids exactly the prize, so one-up takes every round it chooses to contest at one card
    of overspend and concedes only the single dearest prize. Winning 1+2+...+(N-1) while
    losing N is a win for every N >= 4, and our decks are N in (5, 8, 10, 13).

    Inherits __init__/_action_values; only step() differs. Two distinct parse misses hit
    two distinct floors — keep them apart:
      * the PRIZE regex misses -> we defer to the parent, which in that situation bids the
        LOWEST legal action id with no rationale. A no-crash floor, not value-matching
        (the parent only value-matches when it can read the prize).
      * a legal-action VALUE line misses -> _action_values uses value = id + 1, which is
        the correct goofspiel identity, so one-up proceeds normally.
    """

    def step(self, state):
        self.last_reason = None
        player = state.current_player()
        legal = list(state.legal_actions(player))
        if not legal:
            return None
        if len(legal) == 1:
            self.last_reason = _cot("Only one card left — I bid it.", "Bid the last card.")
            return legal[0]

        try:
            obs = state.observation_string(player)
        except Exception:
            obs = ""
        _PRIZE_PARSE_STATS["total"] += 1
        if not _RE_PRIZE.search(obs):
            # Prize unreadable: hand over to the parent, which sets last_reason itself.
            # Counted and printed — see _PRIZE_PARSE_STATS above for why this path is the
            # one failure that would otherwise be invisible.
            _PRIZE_PARSE_STATS["fallback"] += 1
            n = _PRIZE_PARSE_STATS["fallback"]
            if n & (n - 1) == 0:
                print(
                    f"[goofspiel] prize-parse fallback {n}/{_PRIZE_PARSE_STATS['total']} "
                    f"— bidding LOWEST card, teacher is DEGRADED",
                    flush=True,
                )
            return super().step(state)

        prize = int(_RE_PRIZE.search(obs).group(1))
        id_to_val = self._action_values(obs, legal)
        above = [a for a in legal if id_to_val[a] > prize]
        if above:
            # Cheapest card that wins: take the prize with the least overspend.
            pick = min(above, key=lambda a: id_to_val[a])
            card = id_to_val[pick]
            self.last_reason = _cot(
                f"The prize this round is worth {prize}. I bid card {card} — the cheapest "
                f"card I hold that still beats it — so I take the prize without wasting a "
                f"high card I will need for a dearer prize later.",
                f"Prize {prize}; bid card {card}.",
            )
        else:
            # No card can win this prize: discard the cheapest and keep the rest of the
            # hand for rounds we can still take.
            pick = min(legal, key=lambda a: id_to_val[a])
            card = id_to_val[pick]
            self.last_reason = _cot(
                f"The prize this round is worth {prize} and no card I hold beats it, so "
                f"this round is lost either way. I throw my lowest card ({card}) to keep "
                f"my stronger cards for the prizes I can still win.",
                f"Prize {prize}; bid card {card}.",
            )
        return pick


# --------------------------------------------------------------------------------------
# Self-test: `python -m envs.mastering_goofspiel` (the same shape as the leduc solver's).
# Runs without pyspiel against a scripted stand-in state, so it works on a laptop as well
# as on the training box. The policy study that justified one-up could only show the
# POLICY is right; it could not show that the shipped bot reads the real observation
# correctly. These cases pin the reading down.
#
# The observation text below follows the imperfect-information goofspiel ObservationString
# layout (open_spiel/games/goofspiel/goofspiel.cc on master: for imp_info + single-player +
# !perfect_recall the observer writes, in order, StringCurrentPointCard,
# StringRemainingPointCards, StringPoints, StringPlayerHand, StringWinSequence).
# Two traps live in that text and both get a case: "Remaining Point Cards:" must not be
# read as the prize, and action ids are 0-based card indices, so card value == id + 1.
# --------------------------------------------------------------------------------------
class _FakeState:
    """Just enough of a pyspiel goofspiel state for the self-test."""

    def __init__(self, obs: str, legal: "list[int]", player: int = 0):
        self._obs, self._legal, self._player = obs, legal, player

    def current_player(self) -> int:
        return self._player

    def legal_actions(self, player=None) -> "list[int]":
        return list(self._legal)

    def observation_string(self, player=None) -> str:
        return self._obs


def _obs(prize, hand, remaining="12345689") -> str:
    return (
        f"Current point card: {prize}\n"
        f"Remaining Point Cards: {remaining}\n"
        "Points: 3 5 \n"
        f"P0 hand: {' '.join(str(c) for c in hand)} \n"
        "Win sequence: 0 1 \n"
    )


def _selftest() -> int:
    HAND = [1, 2, 4, 6, 9]
    # pyspiel: action id == card value - 1. Taken from the upstream source rather than from
    # this module's own docstring — open_spiel/games/goofspiel/goofspiel.cc:
    #   LegalActions(Player) pushes 0-based hand indices  ("for (int bid = 0; ...) movelist.push_back(bid)")
    #   ActionToString renders StrCat("[P", player, "]Bid: ", (action_id + 1))
    # So the value a person (or a rationale) names is one more than the id. That off-by-one
    # is why generate_pvp_toolcalling appends the real action id to this teacher's rationale:
    # the text says "bid card 9" while the correct argument is 8.
    legal = [c - 1 for c in HAND]
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    print("GoofspielOneUpBot (SHIPPED DEFAULT)")
    bot = GoofspielOneUpBot(0)
    # prize 7: cheapest card strictly above 7 is 9 -> action 8
    check("prize 7 -> bid 9", bot.step(_FakeState(_obs(7, HAND), legal)), 8)
    # prize 9: nothing beats it -> concede with the 1 -> action 0
    check("prize 9 -> concede 1", bot.step(_FakeState(_obs(9, HAND), legal)), 0)
    # prize 1: cheapest above 1 is 2 -> action 1
    check("prize 1 -> bid 2", bot.step(_FakeState(_obs(1, HAND), legal)), 1)
    # one legal action comes back untouched
    check("single legal action", bot.step(_FakeState(_obs(5, [4]), [3])), 3)
    # prize unreadable -> parent floor, and no exception
    check("no prize -> no crash", bot.step(_FakeState("garbage\n", legal)), 0)

    print("GoofspielValueMatchBot (previous default, kept for A/B)")
    vm = GoofspielValueMatchBot(0)
    # nearest to 7 is 6 (|6-7|=1), not 9 (|9-7|=2) -> action 5. This is the losing move:
    # one-up meets a 6 with 7-or-better and takes the prize.
    check("prize 7 -> bid 6", vm.step(_FakeState(_obs(7, HAND), legal)), 5)

    print("regression traps")
    # These test _RE_PRIZE directly rather than through step(). Routing the "Remaining Point
    # Cards" trap through the bot would prove nothing: .search() returns the first match and
    # "Current point card:" precedes it in the real observation, so a regex loosened to accept
    # the Remaining line would still return the right prize and the case would pass. Line
    # order would be doing the work, not the pattern. So test the pattern alone.
    check("_RE_PRIZE rejects 'Remaining Point Cards:'",
          _RE_PRIZE.search("Remaining Point Cards: 12345689") is None, True)
    check("_RE_PRIZE reads 'Current point card:'",
          _RE_PRIZE.search("Current point card: 7").group(1), "7")
    check("_RE_PRIZE matches exactly once per observation",
          len(_RE_PRIZE.findall(_obs(7, HAND))), 1)
    # last_reason must be set on an ordinary turn (it becomes the example content).
    bot.step(_FakeState(_obs(7, HAND), legal))
    has_reason = bot.last_reason is not None or GOOFSPIEL_COT_MODE == "none"
    check("last_reason set", has_reason, True)

    print("PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
