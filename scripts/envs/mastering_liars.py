"""Liar's Dice teacher — bidding as Bayesian inference over the hidden dice.

Every bid in Liar's Dice is a claim about dice we cannot see, so the natural policy is the
statistician's one: given what we hold and what has been said, how probable is each claim?
With sixes wild the count of dice showing a given face among the n hidden dice is
Binomial(n, 2/6) (Binomial(n, 1/6) for a bid on sixes), and P(bid is true) is a binomial
tail. The classic analyses of bluffing games — Ferguson and Ferguson's treatment of Liar's
Dice and the poker endgames it descends from — show that a plain likelihood player is
exploitable precisely because it ignores the information carried by the opponent's bids.
Our earlier flat-prior teacher had that weakness: it assumed uniform opponent dice, so its
argmax bids were predictable and a bid-reading opponent beat it comfortably.

This teacher closes that gap. It keeps a posterior over the opponent's hand, updated from
the bids THEY chose (a likelihood model: they were probably bidding what looked good from
their side), and then picks the action that maximises P(claim true | posterior). Calling
"liar" is scored as 1 - P(current bid true), which is the "challenge when the claim is
improbable" rule from the same literature. In our local head-to-head this recovered the
deficit to roughly even, which is about what we expect: the game is close to balanced and
a decisive edge is not on offer.

Sixes are wild (a bid on face f<6 is satisfied by f or 6; a bid on 6 only by 6), matching
pyspiel's liars_dice. This is the liars teacher we ship (LIARS_TEACHER=posterior).
"""
from __future__ import annotations

import math
import random
import re
from functools import lru_cache
from itertools import combinations_with_replacement

_BID_RE = re.compile(r"(\d+)-(\d+)")
_TEMPS = (0.01, 0.05, 0.1, 0.3)  # we do not know how sharply the opponent plays, so average over a few softmax temperatures


def _binom_sf(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def _p_true(q: int, f: int, counts, n_hidden: int) -> float:
    if f == 6:
        known, p_hit = counts[5], 1 / 6
    else:
        known, p_hit = counts[f - 1] + counts[5], 2 / 6
    need = q - known
    if need <= 0:
        return 1.0
    if n_hidden <= 0:
        return 0.0
    return _binom_sf(need, n_hidden, p_hit)


@lru_cache(maxsize=None)
def _hands(n: int):
    out = []
    for combo in combinations_with_replacement(range(6), n):
        c = [0] * 6
        for x in combo:
            c[x] += 1
        mult = math.factorial(n)
        for cnt in c:
            mult //= math.factorial(cnt)
        out.append((tuple(c), mult / 6**n))
    return out


def _decode(bid_id: int):
    return (bid_id // 6 + 1, bid_id % 6 + 1)


def _bid_id(q: int, f: int) -> int:
    return (q - 1) * 6 + (f - 1)


def _softmax_pick(scores, temp: float, idx: int) -> float:
    if temp <= 0:
        best = max(range(len(scores)), key=lambda i: scores[i])
        return 1.0 if idx == best else 1e-9
    m = max(s / temp for s in scores)
    exps = [math.exp(s / temp - m) for s in scores]
    return exps[idx] / sum(exps)


def _posterior(obs, n_opp: int, n_us: int, total_dice: int):
    """Bayes update over the opponent's count-vector. obs = [(context_bid_id|None, opp_chosen_bid_id)].

    Prior: the multinomial over hands. Likelihood of each observed bid: a softmax over the
    opponent's legal actions scored by how true each looks from THEIR hand, marginalised
    over _TEMPS because we do not know their sharpness.
    """
    liar_id = total_dice * 6
    out = []
    for hand, prior in _hands(n_opp):
        if not obs:
            out.append((hand, prior))
            continue
        per_obs = []
        for ctx, chosen in obs:
            cbid = _decode(ctx) if ctx is not None else None
            legal = (list(range(ctx + 1, liar_id)) + [liar_id]) if ctx is not None else list(range(liar_id))
            scores = []
            for a in legal:
                if a == liar_id:
                    scores.append(1.0 - _p_true(cbid[0], cbid[1], hand, n_us))
                else:
                    q, f = _decode(a)
                    scores.append(_p_true(q, f, hand, n_us))
            per_obs.append((scores, legal.index(chosen)))
        mix = 0.0
        for t in _TEMPS:
            lk = 1.0
            for scores, i in per_obs:
                lk *= _softmax_pick(scores, t, i)
            mix += lk
        out.append((hand, prior * mix / len(_TEMPS)))
    tot = sum(w for _, w in out)
    return [(h, w / tot) for h, w in out] if tot > 0 else _hands(n_opp)


class LiarsPosteriorBot:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def step(self, state):
        p = state.current_player()
        info = state.information_state_string(p)
        toks = info.split()
        if not toks:
            return self._rng.choice(state.legal_actions(p))
        own = [int(d) for d in toks[0] if d.isdigit()]
        my_counts = [0] * 6
        for d in own:
            my_counts[d - 1] += 1
        n_own = len(own)
        total_dice = n_own * state.num_players()
        n_opp = total_dice - n_own
        seq = []
        for t in toks[1:]:
            m = _BID_RE.fullmatch(t)
            if m:
                seq.append(_bid_id(int(m.group(1)), int(m.group(2))))
        # Bids alternate starting with player 0, so the i-th bid belongs to player i % 2;
        # keep the opponent's, each paired with the bid it was answering.
        obs = [(seq[i - 1] if i > 0 else None, b) for i, b in enumerate(seq) if i % 2 != p]
        post = _posterior(tuple(obs), n_opp, n_own, total_dice)
        cur = _decode(seq[-1]) if seq else None
        legal = state.legal_actions(p)
        best_a, best = legal[0], -1.0
        for a in legal:
            lbl = state.action_to_string(p, a).strip()
            m = _BID_RE.fullmatch(lbl)
            if m:
                q, f = int(m.group(1)), int(m.group(2))
                prob = sum(w * _p_true(q, f, [c + h for c, h in zip(my_counts, hand)], 0) for hand, w in post)
            elif cur is not None:
                prob = sum(w * (1 - _p_true(cur[0], cur[1], [c + h for c, h in zip(my_counts, hand)], 0)) for hand, w in post)
            else:
                prob = 0.0
            if prob > best:
                best, best_a = prob, a
        return best_a


def _selftest() -> int:
    """Deterministic checks on the numeric core. No pyspiel needed, so it runs anywhere.

    Liar's Dice is the game where a numerical slip hurts us most quietly: a wrong tail
    probability or a broken wild-six rule does not crash, it just bids badly forever. Every
    expectation below follows from the definitions in this file, not from a recorded game.
    """
    fails = []

    def check(name, got, want, tol=0.0):
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    # The bid encoding has to round-trip, or every parsed bid maps to the wrong action id.
    rt = all(_decode(_bid_id(q, f)) == (q, f) for q in range(1, 11) for f in range(1, 7))
    check("bid_id/decode round-trip over q=1..10, f=1..6", rt, True)
    # Round-tripping only proves self-consistency: a face-major encoding round-trips just as
    # well, and pyspiel actually ships one. From open_spiel/games/liars_dice/liars_dice.cc,
    # UnrankBid:
    #   BiddingRule::kResetFace     (the default; kDefaultBiddingRule = "reset-face"):
    #       quantity = bidnum / dice_sides() + 1 ; face = 1 + bidnum % dice_sides()
    #     -> id = (q-1)*6 + (f-1), exactly _bid_id/_decode.
    #   BiddingRule::kResetQuantity (opt-in through the `bidding_rule` game parameter):
    #       face-major — the same ids would name different bids.
    # We construct the game with {'players': 2, 'numdice': 5} and never pass `bidding_rule`,
    # so the default applies and our encoding is right. The same source confirms the liar
    # action id is total_num_dice * dice_sides() and the legal set is current_bid+1 ..
    # liar_id-1 plus liar, which is what _posterior assumes. Pin the convention itself, so a
    # face-major regression fails here rather than as a ValueError from legal.index(chosen)
    # in the middle of a generation run.
    check("_bid_id(1,1) == 0", _bid_id(1, 1), 0)
    check("_bid_id(1,6) == 5  (face varies fastest)", _bid_id(1, 6), 5)
    check("_bid_id(2,1) == 6  (quantity strides by 6)", _bid_id(2, 1), 6)

    # _binom_sf is the upper tail P(X >= k); the boundary cases are exact.
    check("binom_sf(k<=0) = 1", _binom_sf(0, 5, 0.5), 1.0)
    check("binom_sf(k>n) = 0", _binom_sf(6, 5, 0.5), 0.0)
    check("binom_sf(1,1,p=1) = 1", _binom_sf(1, 1, 1.0), 1.0, 1e-12)
    check("binom_sf(1,1,p=0) = 0", _binom_sf(1, 1, 0.0), 0.0, 1e-12)
    mono = all(_binom_sf(k, 5, 0.4) >= _binom_sf(k + 1, 5, 0.4) for k in range(0, 6))
    check("binom_sf non-increasing in k", mono, True)

    # The prior over hands must be a proper distribution, otherwise the posterior is quietly
    # rescaled and every bid score moves with it.
    for n in (1, 2, 3):
        check(f"_hands({n}) priors sum to 1", sum(w for _, w in _hands(n)), 1.0, 1e-9)

    # The rule most likely to be "tidied away" by accident: sixes count for every face except
    # six itself. Lose that line in _p_true and the teacher misjudges every non-six bid.
    three_sixes = [0, 0, 0, 0, 0, 3]
    check("three 6s satisfy a bid of 3 THREES (wild)", _p_true(3, 3, three_sixes, 0), 1.0)
    check("three 6s satisfy a bid of 3 SIXES", _p_true(3, 6, three_sixes, 0), 1.0)
    three_threes = [0, 0, 3, 0, 0, 0]
    check("three 3s do NOT satisfy a bid of 3 SIXES", _p_true(3, 6, three_threes, 0), 0.0)
    check("already-satisfied bid = 1.0", _p_true(1, 3, three_threes, 0), 1.0)
    check("unsatisfiable with no hidden dice = 0.0", _p_true(9, 1, three_threes, 0), 0.0)

    # Per-die success is 1/6 for sixes and 2/6 for any other face (itself plus the wild six),
    # so with one hidden die a non-six bid is worth exactly twice a six bid.
    p6 = _p_true(1, 6, [0] * 6, 1)
    p3 = _p_true(1, 3, [0] * 6, 1)
    check("one hidden die, bid 1 six = 1/6", p6, 1 / 6, 1e-12)
    check("one hidden die, bid 1 three = 2/6", p3, 2 / 6, 1e-12)
    # Known dice and hidden dice together: subtract what we hold, then take the tail on the rest.
    check("wild+hidden: hold one 3, bid 2 threes, 1 hidden", _p_true(2, 3, [0, 0, 1, 0, 0, 0], 1), 2 / 6, 1e-12)

    # _softmax_pick and _posterior do the actual reweighting, so test them, not only _hands.
    sc = [0.1, 0.9, 0.5]
    check("softmax weights sum to 1", sum(_softmax_pick(sc, 0.3, i) for i in range(3)), 1.0, 1e-12)
    check("temp<=0 picks the argmax", _softmax_pick(sc, 0.0, 1), 1.0)
    check("temp<=0 suppresses non-argmax", _softmax_pick(sc, 0.0, 0) < 1e-6, True)
    # obs = [(context_bid_id|None, opp_chosen_bid_id)]. total_dice=2 -> liar_id=12; with no
    # context the legal set is range(12) and the chosen id 5 lies inside it.
    post = _posterior(((None, _bid_id(1, 6)),), 1, 1, 2)
    check("_posterior renormalises to 1", sum(w for _, w in post), 1.0, 1e-9)
    check("_posterior keeps every hand", len(post), len(_hands(1)))

    print("PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
