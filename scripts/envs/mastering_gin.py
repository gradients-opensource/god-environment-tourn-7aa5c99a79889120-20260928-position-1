"""Gin Rummy expert teacher for tool-calling trajectory generation.

Produces the demonstration moves that the student distils from. Every decision is exact where the
game allows it and heuristic only where it must be, which is what makes the trajectories learnable:
a student can reproduce a deterministic policy but not a search.

The engine has four decision paths, one per phase of the game.

Draw
    Take the upcard when it completes or extends a meld, or when the free-roll test says the swap
    cannot lose tempo; otherwise draw from stock. An optional expected-value swap check refines the
    marginal cases.

Discard
    Score every candidate against exact deadwood after the discard, meld membership, adjacency to
    cards already held, and how dangerous the card is to hand over. Deadwood is solved by dynamic
    programming over meld partitions rather than approximated, so the discard is optimal with
    respect to the hand rather than to a rule of thumb.

Knock
    Fire on a win-probability estimate rather than on the bare legality of knocking. Knocking the
    moment it becomes legal is a common way to convert a winning hand into a narrow one.

Meld and layoff
    Choose the partition that minimises remaining deadwood, with a delta search over layoff
    candidates. A legacy path is retained as a never-raising fallback so a parse failure degrades
    into a legal move instead of aborting a game.

Several refinements are gated behind environment variables and default OFF: perfect-information
Monte Carlo discard reranking, dynamic-programming meld declaration, and two-stage knock evaluation.
They are off because they have not earned their cost in measurement, not because they are unfinished.
The EV swap, draw free-roll and discard safety bonus default ON.

The observation parser reads the rendered game state rather than the engine state, which keeps the
teacher honest: it decides from exactly the information the student will see at inference, so a field
the prompt omits cannot leak into a demonstration.
"""
import math
import os
import random
import re
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def set_meta(*_a, **_k):
    """No-op sink for the per-row metadata side-channel.

    The engine reports a couple of diagnostics per decision (the discard quality gap, and whether the
    turn had exactly one legal action and so carried no strategic signal). Nothing downstream
    consumes them here, so they are discarded rather than threaded through the generator.
    """
    return None


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}


RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']


def get_rank(card: str) -> str:
    return card[0]


def get_suit(card: str) -> str:
    return card[1]


def get_value(card: str) -> int:
    return CARD_VALUES[get_rank(card)]


def find_potential_runs(hand: list[str], additional_card: Optional[str] = None) -> list[list[str]]:
    test_hand = hand.copy()
    if additional_card:
        test_hand.append(additional_card)
    suit_groups: dict[str, list[str]] = {}
    for card in test_hand:
        suit_groups.setdefault(get_suit(card), []).append(card)
    runs = []
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if RANK_ORDER.index(get_rank(sorted_cards[j])) == RANK_ORDER.index(get_rank(run[-1])) + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            if len(run) >= 2:
                runs.append(run)
            i = j if len(run) > 1 else i + 1
    return runs


def count_complete_runs(hand: list[str]) -> int:
    return sum(1 for r in find_potential_runs(hand) if len(r) >= 3)


def would_complete_run(hand: list[str], card: str) -> bool:
    current = sum(1 for r in find_potential_runs(hand) if len(r) >= 3)
    return sum(1 for r in find_potential_runs(hand, card) if len(r) >= 3) > current


def would_improve_run(hand: list[str], card: str) -> bool:
    rank_idx = RANK_ORDER.index(get_rank(card))
    suit = get_suit(card)
    for existing in hand:
        if get_suit(existing) != suit:
            continue
        if abs(rank_idx - RANK_ORDER.index(get_rank(existing))) == 1:
            for run in find_potential_runs(hand + [card]):
                if card in run and len(run) >= 3:
                    return True
    return False


def would_complete_set(hand: list[str], card: str) -> bool:
    return sum(1 for c in hand if get_rank(c) == get_rank(card)) >= 2


def would_improve_set(hand: list[str], card: str) -> bool:
    return sum(1 for c in hand if get_rank(c) == get_rank(card)) == 1


@dataclass
class GameState:
    hand:         list[str]
    deadwood:     int
    phase:        str
    knock_card:   int
    upcard:       str
    stock_size:   int
    discard_pile: list[str]
    player_id:    int

    def total_hand_value(self) -> int:
        return sum(get_value(c) for c in self.hand)

    def num_high_cards(self) -> int:
        return sum(1 for c in self.hand if get_value(c) == 10)

    def can_knock(self) -> bool:
        return self.deadwood <= self.knock_card

    def count_pairs(self) -> int:
        return sum(1 for cnt in Counter(get_rank(c) for c in self.hand).values() if cnt >= 2)

    def count_sets(self) -> int:
        return sum(1 for cnt in Counter(get_rank(c) for c in self.hand).values() if cnt >= 3)

    def count_runs(self) -> int:
        return count_complete_runs(self.hand)

    def count_potential_runs(self) -> int:
        return sum(1 for r in find_potential_runs(self.hand) if len(r) == 2)


def extract_and_format_observation(obs_text: str) -> str:
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text
    state_match = re.search(r'Current State:\n', obs_text)
    if not state_match:
        return obs_text
    state_text   = obs_text[state_match.start():]
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id    = int(player_match.group(1)) if player_match else 0
    if 'Legal Actions:' in state_text:
        before_actions, after_actions = state_text.split('Legal Actions:', 1)
        return before_actions + f"You are Player {player_id}.\nLegal Actions:" + after_actions
    return state_text


def parse_hand_from_observation(observation: str) -> list[str]:
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id    = int(player_match.group(1)) if player_match else 0
    # In the Knock/Layoff phases the observation inserts a "Layed melds: ..." line
    # BETWEEN "Deadwood=N" and the "+---+" hand border, which broke the old regex
    # (it required the border immediately after Deadwood) -> ~4-5% of states parsed
    # hand=[] -> the expert's DP move fell back to the smallest legal id for those
    # Knock/Layoff decisions. Allow the optional melds line so the hand parses there.
    section      = re.search(
        rf'Player{player_id}: Deadwood=\d+\n(?:Layed melds:[^\n]*\n)?\+-+\+\n(.*?)\n\+-+\+', observation, re.DOTALL
    )
    hand = []
    if section:
        for row in section.group(1).strip().split('\n'):
            hand.extend(re.findall(r'([A2-9TJQK][shdc])', row))
    return hand


def parse_discard_pile(observation: str) -> list[str]:
    m = re.search(r'Discard pile: (.*?)\n', observation)
    if not m:
        return []
    pile_str = m.group(1).strip()
    if not pile_str:
        return []
    if ' ' in pile_str:
        return pile_str.split()
    return [pile_str[i:i + 2] for i in range(0, len(pile_str), 2)]


def parse_game_state(observation: str) -> GameState:
    if 'Invalid' in observation and 'Legal Actions:' not in observation:
        raise ValueError("Invalid action response — not a game state")
    parse_warnings = []
    player_match   = re.search(r'You are Player (\d+)', observation)
    player_id      = int(player_match.group(1)) if player_match else 0
    hand           = parse_hand_from_observation(observation)
    if not hand:
        parse_warnings.append("hand=[] (empty — shaping disabled)")
    dw_match       = re.search(r'Deadwood=(\d+)', observation)
    deadwood       = int(dw_match.group(1)) if dw_match else 0
    if not dw_match:
        parse_warnings.append("deadwood=0 (fallback — shaping will be 0)")
    phase_match    = re.search(r'Phase: (\w+)', observation)
    phase          = phase_match.group(1) if phase_match else 'Draw'
    if not phase_match:
        parse_warnings.append("phase='Draw' (fallback)")
    knock_match    = re.search(r'Knock card: (\d+)', observation)
    knock_card     = int(knock_match.group(1)) if knock_match else 10
    upcard_match   = re.search(r'Stock size: \d+\s+Upcard: (\w+)', observation)
    upcard         = upcard_match.group(1) if upcard_match else 'XX'
    stock_match    = re.search(r'Stock size: (\d+)', observation)
    stock_size     = int(stock_match.group(1)) if stock_match else 0
    if parse_warnings:
        print(f"[PARSE_WARN] parse_game_state fallbacks: {', '.join(parse_warnings)}")
    return GameState(
        hand=hand, deadwood=deadwood, phase=phase, knock_card=knock_card,
        upcard=upcard, stock_size=stock_size,
        discard_pile=parse_discard_pile(observation), player_id=player_id,
    )


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

_CARD_RE = re.compile(r"\b([A2-9TJQK][shdc])\b")


_CARD_EXACT_RE = re.compile(r"^([A2-9TJQK][shdc])$")


_MELD_GROUP_RE = re.compile(r"^([A2-9TJQK][shdc]){2,}$")


_RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']


_RANK_IDX = {r: i for i, r in enumerate(_RANK_ORDER)}


_SUITS = ('s', 'h', 'd', 'c')


_FULL_DECK = tuple(r + s for r in _RANK_ORDER for s in _SUITS)


_GR_PIMC_ENABLED = os.environ.get("GR_PIMC_ENABLED", "0").strip().lower() in ("1", "true", "yes")


_PIMC_N_WORLDS = int(os.environ.get("GR_PIMC_WORLDS", "12") or "12")


_PIMC_ROLLOUT_DEPTH = int(os.environ.get("GR_PIMC_ROLLOUT", "3") or "3")


_PIMC_DISCARD_TOPK = 4               # PIMC-rerank only the top-K heuristic discards


_PIMC_TIME_BUDGET_S = float(os.environ.get("GR_PIMC_BUDGET_S", "0.03") or "0.03")


_PIMC_GIN_BONUS = 25.0


_PIMC_KNOCK_CARD = 10                # rollout knock-bonus threshold (eligibility uses gs.can_knock())


_GR_EV_SWAP_ENABLED = (os.environ.get("GR_EV_SWAP") or "1").strip().lower() in ("1", "true", "yes", "on")


_GR_SAFETY_BONUS_ENABLED = (os.environ.get("GR_SAFETY_BONUS") or "1").strip().lower() in ("1", "true", "yes", "on")


_GR_DP_MELD_ENABLED = (os.environ.get("GR_DP_MELD") or "0").strip().lower() in ("1", "true", "yes", "on")


_GR_DRAW_FREEROLL = (os.environ.get("GR_DRAW_FREEROLL") or "1").strip().lower() in ("1", "true", "yes", "on")


_GR_KNOCK_TWO_STAGE = (os.environ.get("GR_KNOCK_TWO_STAGE") or "0").strip().lower() in ("1", "true", "yes", "on")


def _flag_on(name: str) -> bool:
    """True iff env var ``name`` is set to a truthy value. Unset / '' / '0' /
    'false' / 'no' / 'off' => False (default OFF). Read at CALL time so an A/B run
    or a unit test can toggle the flag without a reimport. Never raises."""
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _knock_winp_thresh() -> float:
    """GIN_KNOCK_WINP_THRESH (default 0.5), read at call time. Garbage value ->
    0.5 (never raises)."""
    try:
        return float(os.environ.get("GIN_KNOCK_WINP_THRESH") or "0.5")
    except (TypeError, ValueError):
        return 0.5


def _parse_legal_actions(obs: str) -> dict[int, str]:
    """Return {action_id: label_text} parsed from a "Legal Actions:" block.

    Handles both compact labels ("Discard Kh") and Player-prefixed labels
    ("Player: 0 Action: Discard Kh") produced by the env server.
    """
    result: dict[int, str] = {}
    for m in re.finditer(r"^\s*(\d+)\s*->\s*(.+)$", obs, re.MULTILINE):
        label = m.group(2).strip()
        # Strip optional "Player: N Action:" prefix.
        player_strip = re.match(r"^Player:\s*\d+\s*Action:\s*(.+)$", label)
        if player_strip:
            label = player_strip.group(1).strip()
        result[int(m.group(1))] = label
    return result


def _classify_action(label: str) -> dict:
    """Categorize an action label into structured fields.

    Returns dict with keys:
      kind: 'gin' | 'knock' | 'draw_upcard' | 'draw_stock' | 'discard' |
            'meld' | 'pass' | 'other'
      card: card string (e.g. "Kh") for discard, None otherwise.

    'meld' is recognised via _MELD_GROUP_RE so the Knock/Layoff/Wall phases
    can pick declared melds instead of falling through to a generic
    "smallest action id" tiebreak.
    """
    low = label.lower()
    if "gin" in low:
        return {"kind": "gin", "card": None}
    if "knock" in low:
        return {"kind": "knock", "card": None}
    if "draw" in low and "upcard" in low:
        return {"kind": "draw_upcard", "card": None}
    if "draw" in low and ("stock" in low or "deck" in low):
        return {"kind": "draw_stock", "card": None}
    if low == "pass":
        return {"kind": "pass", "card": None}
    if "discard" in low:
        m = _CARD_RE.search(label)
        return {"kind": "discard", "card": m.group(1) if m else None}
    # Bare single card token (e.g. "As", "Kh") => discard/layoff card. OpenSpiel
    # renders these with NO "Discard" word, so the keyword check above misses
    # them; without this case every discard fell through to "other" and the
    # caller emitted min(legal) (the lowest card), making the whole min-deadwood
    # DP dead code. _CARD_EXACT_RE matches a lone single-card label. MUST precede
    # the meld-group check (a meld group is 2+ concatenated tokens).
    stripped = label.strip()
    if _CARD_EXACT_RE.match(stripped):
        return {"kind": "discard", "card": stripped}
    # Meld group declaration (e.g. "AhAsAd" or "4h5h6h"). Check stripped.
    if _MELD_GROUP_RE.match(stripped):
        # The regex guarantees the label is 2-char card tokens back-to-back, so
        # chunking by 2 recovers the declared card set (word-boundary regexes
        # cannot split "AhAsAd" — no \b between letters).
        cards = tuple(stripped[i:i + 2] for i in range(0, len(stripped), 2))
        return {"kind": "meld", "card": None, "cards": cards}
    return {"kind": "other", "card": None}


def _enumerate_meld_masks(hand: list[str]) -> list[int]:
    """Return bitmasks (one per card index in hand) for every valid 3+ meld.

    Sets: enumerate any 3-of-a-rank and 4-of-a-rank subsets.
    Runs: enumerate every same-suit consecutive contiguous run of length >= 3.

    A meld of length L produces C(L, 3) + C(L, 4) + ... masks; we keep all
    of them so the DP can pick the assignment that frees the most cards.
    """
    n = len(hand)
    if n == 0:
        return []
    card_to_idx = {c: i for i, c in enumerate(hand)}
    masks: list[int] = []

    # Sets: same-rank groups of size >= 3.
    rank_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        rank_groups[get_rank(c)].append(c)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            # 3-card subsets.
            from itertools import combinations
            for combo in combinations(cards, 3):
                mask = 0
                for c in combo:
                    mask |= (1 << card_to_idx[c])
                masks.append(mask)
            if len(cards) >= 4:
                # 4-of-a-kind whole group.
                mask = 0
                for c in cards[:4]:
                    mask |= (1 << card_to_idx[c])
                masks.append(mask)

    # Runs: same-suit contiguous sequences of length >= 3.
    suit_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        suit_groups[get_suit(c)].append(c)
    for cards in suit_groups.values():
        # Sort by rank index.
        sorted_cards = sorted(cards, key=lambda c: _RANK_IDX[get_rank(c)])
        # Find maximal consecutive runs first, then enumerate sub-runs.
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if _RANK_IDX[get_rank(sorted_cards[j])] == _RANK_IDX[get_rank(run[-1])] + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            # Every contiguous sub-run of length >= 3 is a valid meld.
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    mask = 0
                    for c in run[start:end]:
                        mask |= (1 << card_to_idx[c])
                    masks.append(mask)
            i = j if len(run) > 1 else i + 1

    return masks


def _optimal_deadwood_and_meld_mask(hand: list[str]) -> tuple[int, int]:
    """Return (min_deadwood, optimal_meld_bitmask) via memoized DP.

    State: bitmask of cards already assigned to melds. Transition: try
    enumerating every remaining meld; pick the assignment minimising the
    sum of card values for unassigned cards (the deadwood).
    """
    n = len(hand)
    if n == 0:
        return 0, 0
    values = [get_value(c) for c in hand]
    masks = _enumerate_meld_masks(hand)
    memo: dict[int, tuple[int, int]] = {}

    def _dp(used: int) -> tuple[int, int]:
        if used in memo:
            return memo[used]
        # Base: no further meld; remaining cards are deadwood.
        best_dw = sum(values[i] for i in range(n) if not (used >> i & 1))
        best_used = used
        for mm in masks:
            if mm & used:
                continue
            child_dw, child_used = _dp(used | mm)
            if child_dw < best_dw:
                best_dw, best_used = child_dw, child_used
        memo[used] = (best_dw, best_used)
        return best_dw, best_used

    return _dp(0)


@lru_cache(maxsize=8192)
def _optimal_deadwood_value_cached(hand_key: tuple) -> int:
    dw, _ = _optimal_deadwood_and_meld_mask(list(hand_key))
    return dw


def _optimal_deadwood(hand: list[str]) -> int:
    """Min deadwood (sum of unmelded card values) achievable on this hand.

    The deadwood VALUE is independent of card ORDER, so we memoize on a canonical
    sorted-tuple key: the discard search calls this many times per turn (the
    resulting hand for each candidate discard, plus the draw look-ahead at
    _upcard_meld_potential) and the same multiset recurs across turns/games. The
    bitmask-returning _optimal_deadwood_and_meld_mask / _optimal_meld_cards stay
    uncached (the mask is index-order-dependent, so it must not be keyed on a
    sorted hand)."""
    return _optimal_deadwood_value_cached(tuple(sorted(hand)))


def _optimal_meld_cards(hand: list[str]) -> set[str]:
    """Set of cards that go into a meld under the optimal assignment."""
    if not hand:
        return set()
    _, used_mask = _optimal_deadwood_and_meld_mask(hand)
    return {hand[i] for i in range(len(hand)) if (used_mask >> i & 1)}


def _optimal_meld_partition(hand: list[str]) -> list[frozenset]:
    """The LIST OF MELDS (as card frozensets) chosen along the DP-optimal path.

    Duplicates the bitmask DP but records the chosen meld masks, so the Knock
    phase can declare EXACTLY the partition melds. Subset tests against the
    union-of-melded-cards helper are WRONG for this purpose: a sub-run of an
    optimal 4-card run passes the subset test but declaring it strands the end
    cards as deadwood — partition-meld EQUALITY is required.
    """
    n = len(hand)
    if n == 0:
        return []
    values = [get_value(c) for c in hand]
    masks = _enumerate_meld_masks(hand)
    memo: dict[int, tuple[int, tuple]] = {}

    def _dp(used: int) -> tuple[int, tuple]:
        if used in memo:
            return memo[used]
        best_dw = sum(values[i] for i in range(n) if not (used >> i & 1))
        best_melds: tuple = ()
        for mm in masks:
            if mm & used:
                continue
            child_dw, child_melds = _dp(used | mm)
            if child_dw < best_dw:
                best_dw, best_melds = child_dw, (mm,) + child_melds
        memo[used] = (best_dw, best_melds)
        return memo[used]

    _, meld_masks = _dp(0)
    return [
        frozenset(hand[i] for i in range(n) if (mm >> i & 1))
        for mm in meld_masks
    ]


def _meld_extension_cards(cards) -> set[str]:
    """Cards that would EXTEND this meld group (opponent layoff candidates).

    Set of 3 -> the missing 4th same-rank card(s); run -> the same-suit
    rank-1/rank+1 cards off both ends. Unknown shapes -> empty set.
    """
    cs = [c for c in cards if isinstance(c, str) and len(c) == 2]
    if len(cs) < 3:
        return set()
    ranks = {get_rank(c) for c in cs}
    if len(ranks) == 1:
        # Set: extension = the missing suits of this rank.
        rank = next(iter(ranks))
        have = {get_suit(c) for c in cs}
        return {rank + s for s in _SUITS if s not in have}
    suits = {get_suit(c) for c in cs}
    if len(suits) == 1:
        # Run: extension = rank-1 below the low end + rank+1 above the high end.
        suit = next(iter(suits))
        idxs = sorted(_RANK_IDX[get_rank(c)] for c in cs if get_rank(c) in _RANK_IDX)
        if not idxs:
            return set()
        exts: set[str] = set()
        if idxs[0] > 0:
            exts.add(_RANK_ORDER[idxs[0] - 1] + suit)
        if idxs[-1] < 12:
            exts.add(_RANK_ORDER[idxs[-1] + 1] + suit)
        return exts
    return set()


def _partial_set_can_complete(card: str, hand: list[str], dead: set[str]) -> bool:
    """True iff card has a chance of forming a set of 3+ given what's gone.

    A pair (card + one same-rank already in hand) can become a set if at
    least one of the two remaining same-rank cards is still live (not in
    `dead`).
    """
    rank = get_rank(card)
    own_suits = {get_suit(c) for c in hand if get_rank(c) == rank}
    if len(own_suits) >= 3:
        return True  # already a triple in hand
    for s in 'shdc':
        if s in own_suits:
            continue
        if (rank + s) not in dead:
            return True
    return False


def _partial_run_can_complete(card: str, hand: list[str], dead: set[str]) -> bool:
    """True iff card has a chance of extending into a run of 3+ given dead cards.

    Checks whether any same-suit neighbour (rank ±1) of the existing
    consecutive segment containing this card is still live.
    """
    target_idx = _RANK_IDX[get_rank(card)]
    suit = get_suit(card)
    # Build sorted same-suit rank indices including `card`.
    indices = sorted({_RANK_IDX[get_rank(c)] for c in hand if get_suit(c) == suit} | {target_idx})
    if not indices:
        return False
    # Find contiguous segment that contains target_idx.
    seg = [target_idx]
    # Extend right.
    cur = target_idx
    for i in indices:
        if i <= target_idx:
            continue
        if i == cur + 1:
            seg.append(i)
            cur = i
        else:
            break
    # Extend left.
    cur = target_idx
    for i in reversed(indices):
        if i >= target_idx:
            continue
        if i == cur - 1:
            seg.insert(0, i)
            cur = i
        else:
            break
    if len(seg) >= 3:
        return True
    lo, hi = seg[0], seg[-1]
    if lo > 0 and (_RANK_ORDER[lo - 1] + suit) not in dead:
        return True
    if hi < 12 and (_RANK_ORDER[hi + 1] + suit) not in dead:
        return True
    return False



def _hand_stats(hand: list[str]) -> tuple[dict[str, int], set[str]]:
    """Pre-compute rank counts + cards adjacent in suit (one rank apart)."""
    rank_counts: dict[str, int] = Counter(get_rank(c) for c in hand)
    adj: set[str] = set()
    by_suit: dict[str, list[int]] = defaultdict(list)
    for c in hand:
        by_suit[get_suit(c)].append(_RANK_IDX[get_rank(c)])
    for s, idxs in by_suit.items():
        idxs.sort()
        for k, idx in enumerate(idxs):
            if (k > 0 and idxs[k - 1] == idx - 1) or (k < len(idxs) - 1 and idxs[k + 1] == idx + 1):
                adj.add(_RANK_ORDER[idx] + s)
    return rank_counts, adj


def _score_discard_candidate(
    card: str,
    hand: list[str],
    meld_cards: set[str],
    rank_counts: dict[str, int],
    adj_cards: set[str],
    dead_set: set[str],
) -> int:
    """Score a discard candidate: HIGHER = better candidate (prefer to discard).

    Factors:
      + card value      (high-value deadwood costs more if we lose)
      - 15 if in meld   (never discard a melded card)
      - 8  if pair      (preserve potential set) — vetoed if dead-cards block
      - 5  if adjacent  (preserve potential run) — vetoed if dead-cards block
      - 6  if opponent likely needs this card (pile-draw signal) — KEEP it

    Vetoes use dead_set to detect "the completing card is gone": pair
    bonus only applies if at least one same-rank card is still live; adj
    bonus only applies if at least one neighbouring rank in the same suit
    is still live.

    _best_discard_card picks the MAX-scoring card, so anything we want to keep
    must be pushed DOWN, not up.
    """
    score = get_value(card)
    if card in meld_cards:
        score -= 15
    has_pair = rank_counts[get_rank(card)] >= 2
    has_adj = card in adj_cards
    if has_pair and _partial_set_can_complete(card, hand, dead_set):
        score -= 8
    if has_adj and _partial_run_can_complete(card, hand, dead_set):
        score -= 5
    return score


_GR_DISCARD_BAND = float(os.environ.get("GR_DISCARD_BAND", "4.0") or "4.0")


_GR_DISCARD_TAU = float(os.environ.get("GR_DISCARD_TAU", "2.0") or "2.0")


_GR_DISCARD_TAU = max(_GR_DISCARD_TAU, 1e-6)


def _pile_adjacency_bonus(card: str, recent: list) -> int:
    """E. Safety-in-band: +2 (bounded) iff `card` is same-rank OR same-suit with
    rank within 1 of any of the opponent's recent discard-pile entries — the
    opponent discarded near it, so it is SAFER to discard (they don't want it).
    Deterministic; the caller applies it inside the existing score so the meld
    penalty (-15) always dominates."""
    try:
        rank = get_rank(card)
        if rank not in _RANK_IDX:
            return 0
        idx = _RANK_IDX[rank]
        suit = get_suit(card)
        for p in recent:
            if not isinstance(p, str) or len(p) != 2:
                continue
            p_rank = get_rank(p)
            if p_rank not in _RANK_IDX:
                continue
            if p_rank == rank:
                return 2
            if get_suit(p) == suit and abs(_RANK_IDX[p_rank] - idx) <= 1:
                return 2
    except Exception:
        return 0
    return 0


def _best_discard_card(
    hand: list[str],
    dead_set: set[str],
    sample: bool = False,
    rng=None,
    recent: "list | None" = None,
) -> "str | None":
    """Pick the card whose discard maximises _score_discard_candidate.

    Tiebreak: prefer the higher raw card value. With
    sample=True (persona B) SAMPLE softmax among near-best discards instead of
    argmax — bounded, mixing diversity for league self-play. `recent` is the
    ordered discard pile (last-3 entries feed the E. safety bonus, gated by
    GR_SAFETY_BONUS default ON).
    """
    if not hand:
        return None
    meld_cards = _optimal_meld_cards(hand)
    rank_counts, adj_cards = _hand_stats(hand)
    recent3 = list(recent or [])[-3:] if _GR_SAFETY_BONUS_ENABLED else []
    scored = [
        (
            c,
            _score_discard_candidate(c, hand, meld_cards, rank_counts, adj_cards, dead_set)
            + (_pile_adjacency_bonus(c, recent3) if recent3 else 0),
        )
        for c in hand
    ]
    if not scored:
        return None
    if not sample:
        best_card = None
        best_score = None
        for c, s in scored:
            if best_score is None or s > best_score or (s == best_score and get_value(c) > get_value(best_card)):
                best_score = s
                best_card = c
        return best_card
    # persona B: softmax-sample among near-best discards (bounded deviation).
    best = max(s for _, s in scored)
    band = [(c, s) for c, s in scored if best - s <= _GR_DISCARD_BAND]
    if len(band) <= 1:
        return band[0][0]
    top = max(s for _, s in band)
    weights = [math.exp((s - top) / _GR_DISCARD_TAU) for _, s in band]
    total = sum(weights)
    if total <= 0.0:
        return band[0][0]
    return (rng or random).choices([c for c, _ in band], weights=weights, k=1)[0]


def _upcard_meld_potential(hand: list[str], upcard: str) -> int:
    """Return reduction in optimal deadwood if we add the upcard to our hand.

    Higher = more useful upcard. 0 means the upcard is pure deadwood
    relative to the current optimal meld assignment.

    This is strictly more accurate than the categorical 0/1/2/3 tier
    used previously (which counted "improves a pair" the same as
    "completes a meld"), because it integrates with the DP that already
    enumerates every legal meld combination.
    """
    if not upcard or upcard == "XX" or len(upcard) != 2:
        return 0
    return max(0, _optimal_deadwood(hand) - _optimal_deadwood(hand + [upcard]))


def _ev_swap_take_upcard(hand: list[str], upcard: str) -> bool:
    """D. EV-positive deadwood-swap: take a ZERO-meld-delta upcard when it is a
    cheap card (value <= 5) and swapping it for one of our top-value junk cards
    cuts deadwood by >= GR_SWAP_MARGIN.

    Cheap gate: the swapped-out candidate ranges over ONLY the top-3
    highest-value cards NOT in the DP-optimal meld set. Deterministic (sorted
    with a card-string tiebreak). Never raises via the caller's wrap.
    """
    if not hand or not isinstance(upcard, str) or upcard == "XX" or not _CARD_EXACT_RE.match(upcard):
        return False
    up_val = get_value(upcard)
    if up_val > 5:
        return False
    dw = _optimal_deadwood(hand)
    if dw < up_val + _GR_SWAP_MARGIN:
        return False
    meld_cards = _optimal_meld_cards(hand)
    non_meld = [c for c in hand if c not in meld_cards]
    top3 = sorted(non_meld, key=lambda c: (-get_value(c), c))[:3]
    if not top3:
        return False
    best_after = None
    for d in top3:
        rem = list(hand)
        rem.remove(d)
        rem.append(upcard)
        after = _optimal_deadwood(rem)
        if best_after is None or after < best_after:
            best_after = after
    return best_after is not None and (dw - best_after) >= _GR_SWAP_MARGIN


def _pimc_seed(hand: list[str]) -> int:
    """Per-hand RNG seed so both self-play seats reproduce identical world samples
    for the same hand (SFT determinism). zlib.crc32 (not the salted built-in
    hash()) so the seed is stable run-to-run, not just within one process."""
    return zlib.crc32("".join(sorted(hand)).encode()) & 0xFFFFFFFF


def _pimc_best_discard(hand: list[str]) -> "tuple[str | None, int]":
    """Greedy discard minimizing resulting optimal deadwood (tie-break: higher card
    value). Returns (card, deadwood). Reuses the cached optimal-deadwood DP."""
    if not hand:
        return None, 0
    best_card = hand[0]
    best_dw = None
    seen = set()
    for c in hand:
        if c in seen:
            continue
        seen.add(c)
        rem = list(hand)
        rem.remove(c)
        dw = _optimal_deadwood(rem)
        if best_dw is None or dw < best_dw or (dw == best_dw and get_value(c) > get_value(best_card)):
            best_dw = dw
            best_card = c
    return best_card, (best_dw if best_dw is not None else 0)


def _pimc_unseen_pool(hand: list[str], seen: set) -> list[str]:
    known = set(hand) | set(seen)
    return [c for c in _FULL_DECK if c not in known]


def _pimc_build_worlds(hand, seen, rng, deadline):
    """Sample determinized (opponent_hand, stock_order) worlds from the unseen pool.
    Collapses to 1 world when the pool barely exceeds the opp hand (endgame) and
    honours the time deadline (stops sampling once exceeded)."""
    pool = _pimc_unseen_pool(hand, seen)
    if not pool:
        return []
    opp_size = min(10, len(pool))
    n = 1 if len(pool) <= opp_size else _PIMC_N_WORLDS
    worlds = []
    for _ in range(n):
        if time.monotonic() > deadline:
            break
        sh = pool[:]
        rng.shuffle(sh)
        worlds.append((sh[:opp_size], sh[opp_size:]))
    return worlds


def _pimc_prospects(hand10: list[str], stock: list[str]) -> float:
    """Value of a 10-card hand: -deadwood + knock/gin bonus + a short greedy
    look-ahead drawing from the determinized stock."""
    dw = _optimal_deadwood(hand10)
    value = -float(dw)
    if dw == 0:
        value += _PIMC_GIN_BONUS
    elif dw <= _PIMC_KNOCK_CARD:
        value += 10.0 + (_PIMC_KNOCK_CARD - dw) * 0.5
    cur = list(hand10)
    cur_dw = dw
    for ply in range(_PIMC_ROLLOUT_DEPTH):
        if ply >= len(stock):
            break
        cand = cur + [stock[ply]]
        d2, ndw = _pimc_best_discard(cand)
        if ndw < cur_dw:
            cur = cand
            cur.remove(d2)
            value += (dw - ndw) * (0.6 ** (ply + 1)) * 0.5
            cur_dw = ndw
    return value


def _pimc_draw_value(kind, hand, upcard, worlds) -> "float | None":
    """Average PIMC value of a draw action (draw_upcard / draw_stock) over worlds."""
    if not worlds:
        return None
    total = 0.0
    for _opp, stock in worlds:
        if kind == "draw_upcard":
            if not upcard or len(upcard) != 2:
                return None
            after = hand + [upcard]
        elif stock:
            after = hand + [stock[0]]
            stock = stock[1:]
        else:
            total += _pimc_prospects(hand, [])
            continue
        disc, _ = _pimc_best_discard(after)
        h10 = list(after)
        if disc in h10:
            h10.remove(disc)
        total += _pimc_prospects(h10, stock)
    return total / len(worlds)


def _pimc_discard_value(card, hand11, worlds) -> "float | None":
    """Average PIMC value of discarding `card` from an 11-card hand over worlds."""
    if not worlds:
        return None
    rem = list(hand11)
    if card in rem:
        rem.remove(card)
    total = 0.0
    for _opp, stock in worlds:
        total += _pimc_prospects(rem, stock)
    return total / len(worlds)


def _pimc_choose_draw(gs, upcard_id, stock_id) -> "int | None":
    """PIMC draw decision: upcard vs stock by averaged world value. None -> defer
    to the heuristic (no real choice / empty worlds / any error)."""
    try:
        if upcard_id is None or stock_id is None:
            return None
        hand = list(gs.hand)
        if not hand:
            return None
        seen = set(gs.discard_pile or [])
        if gs.upcard and len(gs.upcard) == 2:
            seen.add(gs.upcard)
        rng = random.Random(_pimc_seed(hand))
        worlds = _pimc_build_worlds(hand, seen, rng, time.monotonic() + _PIMC_TIME_BUDGET_S)
        if not worlds:
            return None
        v_up = _pimc_draw_value("draw_upcard", hand, gs.upcard, worlds)
        v_st = _pimc_draw_value("draw_stock", hand, gs.upcard, worlds)
        if v_up is None or v_st is None:
            return None
        return upcard_id if v_up >= v_st else stock_id
    except Exception:
        return None


def _pimc_choose_discard(gs, dead_set) -> "str | None":
    """PIMC discard: pre-rank ALL discards by the cheap heuristic, then PIMC-rerank
    only the top-K (the biggest throughput lever). None -> defer to the heuristic."""
    try:
        hand = list(gs.hand)
        if len(hand) < 2:
            return None
        meld_cards = _optimal_meld_cards(hand)
        rank_counts, adj_cards = _hand_stats(hand)
        ranked = sorted(
            set(hand),
            key=lambda c: _score_discard_candidate(
                c, hand, meld_cards, rank_counts, adj_cards, dead_set),
            reverse=True,
        )
        topk = ranked[:_PIMC_DISCARD_TOPK]
        seen = set(gs.discard_pile or [])
        if gs.upcard and len(gs.upcard) == 2:
            seen.add(gs.upcard)
        rng = random.Random(_pimc_seed(hand))
        worlds = _pimc_build_worlds(hand, seen, rng, time.monotonic() + _PIMC_TIME_BUDGET_S)
        if not worlds:
            return None
        best_card = None
        best_val = None
        for c in topk:
            v = _pimc_discard_value(c, hand, worlds)
            if v is None:
                continue
            if best_val is None or v > best_val:
                best_val = v
                best_card = c
        return best_card
    except Exception:
        return None


def _choose_draw(gs, classified: dict[int, dict]) -> "int | None":
    """Pick draw_upcard / draw_stock / pass for the Draw or FirstUpcard phase.

    Draw the upcard only when it materially reduces optimal deadwood
    (i.e. completes or improves a meld). Otherwise draw from stock so we
    gain entropy. Pass when neither is legal (rare; mostly FirstUpcard
    second-decision).
    """
    upcard_id = next((aid for aid, info in classified.items() if info["kind"] == "draw_upcard"), None)
    stock_id = next((aid for aid, info in classified.items() if info["kind"] == "draw_stock"), None)
    pass_id = next((aid for aid, info in classified.items() if info["kind"] == "pass"), None)

    if _GR_PIMC_ENABLED:
        choice = _pimc_choose_draw(gs, upcard_id, stock_id)
        if choice is not None:
            return choice

    if upcard_id is not None:
        # F. Draw free-roll (GR_DRAW_FREEROLL, default ON): take the upcard iff,
        # after adding it and making our best discard, optimal deadwood STRICTLY
        # drops. Catches the meld-EXTENSION free-roll the old delta>0 rule missed
        # (extending a made meld is delta 0 but lets us dump a junk card on the
        # forced discard). Discarding the upcard back is always considered by
        # _pimc_best_discard, so a strict drop guarantees a real gain and rejects
        # the degenerate take-then-re-discard case. Subsumes the EV-swap.
        if _GR_DRAW_FREEROLL:
            try:
                dw_now = _optimal_deadwood(list(gs.hand))
                _, dw_after = _pimc_best_discard(list(gs.hand) + [gs.upcard])
                if dw_after < dw_now:
                    return upcard_id
            except Exception:
                # never-raise: fall back to the meld-potential heuristic below
                if _upcard_meld_potential(gs.hand, gs.upcard) > 0:
                    return upcard_id
        else:
            delta = _upcard_meld_potential(gs.hand, gs.upcard)
            if delta > 0:
                return upcard_id
            # D. EV-positive deadwood-swap (GR_EV_SWAP, default ON): zero meld
            # delta but a cheap upcard that swaps top junk for >= GR_SWAP_MARGIN.
            if _GR_EV_SWAP_ENABLED and delta == 0:
                try:
                    if _ev_swap_take_upcard(list(gs.hand), gs.upcard):
                        return upcard_id
                except Exception:
                    pass  # never-raise: fall through to stock draw
    if stock_id is not None:
        return stock_id
    if pass_id is not None:
        return pass_id
    return upcard_id  # last resort


def _knock_win_prob(gs) -> float:
    """GIN_KNOCK_WINP: deterministic estimate of P(win | knock now), in [0, 1].

    Gin (deadwood 0) cannot be undercut -> 1.0. Otherwise a thicker knock (higher
    deadwood) is easier for the opponent to undercut, so the estimate is a MONOTONE
    NON-INCREASING function of our deadwood: at deadwood == knock_card the knock is
    maximally thin (0.5), falling linearly toward the deadwood ceiling. Because the
    estimate does NOT depend on the threshold, ``winp > thresh`` is monotone in the
    threshold — a HIGHER GIN_KNOCK_WINP_THRESH can only ever demand a LOWER deadwood
    to knock, so it never makes us knock EARLIER. Unknown/garbage deadwood -> 1.0
    (defer to the legal knock rather than suppress it). Never raises."""
    try:
        dw = int(gs.deadwood)
    except (TypeError, ValueError, AttributeError):
        return 1.0
    if dw <= 0:
        return 1.0  # gin: un-undercuttable
    try:
        knock_card = int(gs.knock_card)
    except (TypeError, ValueError, AttributeError):
        knock_card = 10
    if knock_card <= 0:
        knock_card = 10
    winp = 1.0 - dw / (2.0 * knock_card)
    return max(0.0, min(1.0, winp))


def _gin_discard_qgap(gs) -> "float | None":
    """Row-meta qgap for the Discard phase = the deadwood gap (teacher's OWN units)
    between the two best discards: 0.0 when two discards tie for the minimum
    resulting deadwood, larger when the single best discard is clearly ahead. None
    when it can't be computed (< 2 distinct cards / any error). Deterministic; never
    raises; side-channel only (does NOT affect the returned action)."""
    try:
        hand = list(gs.hand)
        if len(hand) < 2:
            return None
        dws = []
        seen = set()
        for c in hand:
            if c in seen:
                continue
            seen.add(c)
            rem = list(hand)
            rem.remove(c)
            dws.append(_optimal_deadwood(rem))
        if len(dws) < 2:
            return None
        dws.sort()
        return float(dws[1] - dws[0])
    except Exception:
        return None


def _rank_discard_ids(gs, classified: dict[int, dict], chosen: "int | None" = None) -> list[int]:
    """Rank the LEGAL discard action ids by the deadwood they leave behind,
    lowest (best) first, then pin `chosen` to the front.

    The pin matters: `_choose_discard` does not always play the pure
    minimum-deadwood card (dead-card vetoes, the safety bonus and PIMC/near-
    best sampling can all pick something else), so a plain deadwood sort can
    leave the id that was actually played outside the printed slice. Scoring
    the WHOLE legal set first (not just the top few) before pinning guarantees
    `chosen` is always found and always leads, so the sentence never shows a
    ranking that omits the id the trajectory builder is about to emit. Ties
    broken by action id for a stable order. Trimmed to 3 entries afterward —
    that is what the assistant sentence surfaces."""
    try:
        hand = list(gs.hand)
        scored = []
        for aid, info in classified.items():
            if info.get("kind") != "discard" or not info.get("card"):
                continue
            card = info["card"]
            if card not in hand:
                continue
            rem = list(hand)
            rem.remove(card)
            scored.append((_optimal_deadwood(rem), aid))
        scored.sort(key=lambda t: (t[0], t[1]))
        ids = [aid for _dw, aid in scored]
        if chosen is not None and chosen in ids:
            ids = [chosen] + [a for a in ids if a != chosen]
        return ids[:3]
    except Exception:
        return []


def _rank_knock_ids(gs, classified: dict[int, dict], chosen: "int | None" = None) -> list[int]:
    """Rank the knock id against the best continue-and-discard id, for the
    Knock-vs-continue decision. Knock is only ever legal here when
    `_should_knock` already approved it, so when both ids exist the knock id
    leads; a caller that only wants the alternative can read index 1. `chosen`
    is accepted for the same reason as `_rank_discard_ids`: if it disagrees
    with the single legal knock id (should not happen, but never-raise means
    we do not assume it can't), `chosen` still wins the front slot."""
    try:
        knock_id = next((aid for aid, info in classified.items() if info.get("kind") == "knock"), None)
        if knock_id is None:
            return []
        lead = chosen if (chosen is not None and chosen == knock_id) else knock_id
        alt = _rank_discard_ids(gs, classified)
        alt_best = next((a for a in alt if a != lead), None)
        return [lead, alt_best] if alt_best is not None else [lead]
    except Exception:
        return []


def _should_knock(gs) -> bool:
    """Knock timing. Default (GR_KNOCK_TWO_STAGE off) = knock the instant eligible
    (deadwood <= knock_card), gin included — so the ~6-ply knock-completion
    sequence finishes within the turn cap.

    GR_KNOCK_TWO_STAGE on = undercut-aware stock-keyed gate: while the stock is
    deep, hold for a bigger deadwood margin so a thin knock can't be undercut; as
    the stock drains, RELAX to knock-always well before the turn cap so the
    completion sequence never stalls. Gin (deadwood 0) always knocks."""
    if not gs.can_knock():
        return False
    # GIN_KNOCK_WINP (default OFF): replace the fixed deadwood rule with a
    # win-probability gate — knock only when the deterministic P(win | knock now)
    # estimate EXCEEDS GIN_KNOCK_WINP_THRESH (default 0.5). Deterministic and
    # monotone in the threshold (a higher threshold never knocks earlier). The
    # rest of the engine (free-roll draw, Wall-knock fix, exact-DP layoff, opp
    # model) is untouched — only the KNOCK TIMING is gated. OFF => the two rules
    # below are byte-identical to the shipped behaviour.
    if _flag_on("GIN_KNOCK_WINP"):
        return _knock_win_prob(gs) > _knock_winp_thresh()
    if not _GR_KNOCK_TWO_STAGE:
        return True
    try:
        dw = int(gs.deadwood)
        if dw == 0:
            return True  # gin: un-undercuttable, always knock
        stock = int(gs.stock_size)
        knock_card = int(gs.knock_card)
        # endgame / low-knock-card variant: terminate decisively (anti-stall)
        if stock <= 0 or stock < 20 or knock_card < 10:
            return True
        if stock >= 30:
            return dw <= max(3, knock_card - 6)
        return dw <= knock_card - 3  # 20 <= stock < 30
    except (TypeError, ValueError, AttributeError):
        return True  # never-raise: fall back to knock-ASAP


def _choose_discard(
    gs,
    classified: dict[int, dict],
    dead_set: set[str],
    sample: bool = False,
    rng=None,
) -> "int | None":
    """Pick the best discard for the Discard phase, knock-eligible.

    GR-knock-gate: knock only when `_should_knock(gs)` (Gin / comfortably-low
    deadwood / endgame). Taking the knock the instant deadwood <= knock_card
    threw away the deadwood margin that decides peer matchups. Otherwise emit the
    discard that maximises the multi-factor score (chase a lower deadwood / Gin).
    With sample=True (persona B) the discard CARD is sampled among near-best;
    the knock gate is unchanged so completion still fits the turn cap.
    """
    if gs.can_knock() and _should_knock(gs):
        for aid, info in classified.items():
            if info["kind"] == "knock":
                return aid
    target = None
    if _GR_PIMC_ENABLED:
        target = _pimc_choose_discard(gs, dead_set)
    if target is None:
        target = _best_discard_card(
            gs.hand, dead_set, sample=sample, rng=rng,
            recent=gs.discard_pile,
        )
    if target is not None:
        for aid, info in classified.items():
            if info["kind"] == "discard" and info["card"] == target:
                return aid
        # Exact-card label not present (rare label-format edge case):
        # fall through to any discard action.
        for aid, info in classified.items():
            if info["kind"] == "discard":
                return aid
    return None


def _partition_meld_group_id(gs, classified: dict[int, dict], dead_set: set[str]) -> "int | None":
    """B. DP-consistent meld declaration for the Knock/Layoff phases.

    Prefer the legal meld-group whose card set EQUALS one of the DP-optimal
    partition melds (partition equality, NOT union-subset — a sub-run of an
    optimal 4-run passes the subset test but strands the end cards). Among the
    matching groups prefer the one with the FEWEST live extension cards (dead
    extensions = the opponent cannot lay off onto it); deterministic tiebreak =
    smallest action id. Fallback = first meld-group action (old behavior).
    Returns None iff no meld-group action is legal.
    """
    melds = [(aid, info) for aid, info in classified.items() if info["kind"] == "meld"]
    if not melds:
        return None
    # Conservative default: FIRST legal meld-group action (lowest-entropy, still
    # declares a valid meld). Only the GR_DP_MELD=1 A/B path runs the DP-optimal-
    # partition-equality selection that spreads distinct high meld ids.
    if not _GR_DP_MELD_ENABLED:
        return melds[0][0]
    try:
        partition = set(_optimal_meld_partition(list(gs.hand)))
        matching = [
            (aid, info) for aid, info in melds
            if info.get("cards") and frozenset(info["cards"]) in partition
        ]
        if matching:
            def _live_ext_then_id(item):
                aid, info = item
                exts = _meld_extension_cards(info["cards"])
                return (sum(1 for e in exts if e not in dead_set), aid)
            return min(matching, key=_live_ext_then_id)[0]
    except Exception:
        pass
    return melds[0][0]  # old behavior: first meld-group action


def _layoff_delta_choice(gs, classified: dict[int, dict]) -> "int | None":
    """A. Exact-DP layoff / forced discard for the Knock/Layoff/Wall phases.

    For each LEGAL bare-card action c: delta = dw(hand) - dw(hand \\ {c}) via
    the cached optimal-deadwood DP. Pick the legal card with MAX delta
    (tiebreak: higher card value). If Pass is legal and NO legal card has
    delta > 0 -> Pass. If Pass is NOT legal (Knock-phase forced discard /
    Wall) -> play the max-delta card anyway (least-bad). Returns None only
    when nothing is evaluable AND Pass is illegal (caller falls back).
    """
    pass_id = next((aid for aid, info in classified.items() if info["kind"] == "pass"), None)
    hand = list(gs.hand)
    if not hand:
        return pass_id
    base_dw = _optimal_deadwood(hand)
    best = None  # (delta, card_value, action_id)
    for aid, info in classified.items():
        if info["kind"] != "discard" or not info["card"]:
            continue
        card = info["card"]
        if card not in hand:
            continue
        rem = list(hand)
        rem.remove(card)
        delta = base_dw - _optimal_deadwood(rem)
        val = get_value(card)
        if best is None or delta > best[0] or (delta == best[0] and val > best[1]):
            best = (delta, val, aid)
    if best is None:
        return pass_id
    if pass_id is not None and best[0] <= 0:
        return pass_id
    return best[2]


def _choose_meld_or_layoff(
    gs,
    classified: dict[int, dict],
    dead_set: set[str],
    sample: bool = False,
    rng=None,
) -> "int | None":
    """Pick the right action during Knock / Layoff / Wall phases.

    Order of preference:
      1. DP-consistent meld-group declaration (partition-equality match, B;
         fallback inside = first meld-group action).
      2. Exact-DP layoff / forced discard by MAX deadwood-delta (A). The old
         code computed _best_discard_card over the WHOLE hand and required an
         exact label match; in the Layoff phase only knocker-meld-extending
         cards are legal, so the match failed and we PASSED, keeping ALL our
         deadwood. Now every LEGAL card is evaluated directly.
      3. Pass (only when no legal card strictly reduces deadwood — handled
         inside step 2 — or nothing else is legal).
      4. Legacy path (old scoring + smallest legal id) as never-raise fallback.
    """
    try:
        # PHASE ROUTING (review fix): the exact deadwood-delta rule is provably
        # optimal only at TERMINAL-SCORED plies (Layoff, and the post-knock
        # forced discard) where future-potential/safety scoring is irrelevant.
        #   * Wall: the hand CONTINUES — keep the multi-factor scorer (dead-card
        #     vetoes + safety bonus) and persona-B's near-best sampling.
        #   * Knock + sample=True (persona-B driver): keep the legacy sampled
        #     scorer so league diversity survives (all 11 cards are legal there,
        #     so the legacy whole-hand argmax matches a legal action fine).
        #   * Layoff (both personas) + Knock label path: partition-meld
        #     declaration + exact-delta layoff/forced-discard below.
        phase = (getattr(gs, "phase", "") or "").strip()
        # Wall bug fix: the legacy chooser below has NO knock branch, so a Knock
        # action that is legal in the Wall phase gets Passed into a draw (win 1.0
        # -> draw 0.5). OpenSpiel only offers Knock in Wall when it is legal, so
        # taking it unconditionally is safe. Scan before the legacy routing.
        if phase == "Wall" and gs.can_knock():
            for aid, info in classified.items():
                if info.get("kind") == "knock":
                    return aid
        if phase == "Wall" or (phase == "Knock" and sample):
            return _choose_meld_or_layoff_legacy(gs, classified, dead_set, sample=sample, rng=rng)

        # 1. Meld-group declaration (B).
        meld_choice = _partition_meld_group_id(gs, classified, dead_set)
        if meld_choice is not None:
            return meld_choice

        # 2. Exact-DP layoff / forced discard (A).
        if any(info["kind"] == "discard" for info in classified.values()):
            choice = _layoff_delta_choice(gs, classified)
            if choice is not None:
                return choice

        # 3. Pass.
        for aid, info in classified.items():
            if info["kind"] == "pass":
                return aid
    except Exception:
        pass  # never-raise: fall back to the old code path below

    return _choose_meld_or_layoff_legacy(gs, classified, dead_set, sample=sample, rng=rng)


def _choose_meld_or_layoff_legacy(
    gs,
    classified: dict[int, dict],
    dead_set: set[str],
    sample: bool = False,
    rng=None,
) -> "int | None":
    """Earlier Knock/Layoff/Wall path, retained as the never-raising fallback."""
    # 1. Pick the first meld-group action.
    for aid, info in classified.items():
        if info["kind"] == "meld":
            return aid

    # 2. Discards: same scoring as Discard phase.
    discard_present = any(info["kind"] == "discard" for info in classified.values())
    if discard_present:
        target = _best_discard_card(gs.hand, dead_set, sample=sample, rng=rng)
        if target is not None:
            for aid, info in classified.items():
                if info["kind"] == "discard" and info["card"] == target:
                    return aid

    # 3. Pass.
    for aid, info in classified.items():
        if info["kind"] == "pass":
            return aid

    # 4. Last resort.
    if classified:
        return min(classified.keys())
    return None


def _dead_set(gs) -> set[str]:
    """C. Dead cards = discard pile PLUS the visible upcard (when valid and not
    the 'XX' placeholder). The upcard is public information both seats can see,
    so the veto set is only complete when it is included."""
    dead = set(gs.discard_pile or [])
    up = getattr(gs, "upcard", None)
    if isinstance(up, str) and up != "XX" and _CARD_EXACT_RE.match(up):
        dead.add(up)
    return dead


def _select_action_id(
    gs,
    obs: str,
    sample: bool = False,
    rng=None,
) -> "int | None":
    """Phase-aware dispatch. Always prefer Gin if legal. sample=True (persona B)
    routes the discard through the bounded near-best sampler (knock + draw stay
    deterministic, so the turn-cap completion is unaffected)."""
    legal = _parse_legal_actions(obs)
    if not legal:
        return None
    classified = {aid: _classify_action(label) for aid, label in legal.items()}

    # Always take Gin (deadwood = 0 win).
    for aid, info in classified.items():
        if info["kind"] == "gin":
            return aid

    phase = (gs.phase or "").strip()
    if phase in ("Draw", "FirstUpcard"):
        chosen = _choose_draw(gs, classified)
    elif phase == "Discard":
        chosen = _choose_discard(gs, classified, _dead_set(gs), sample=sample, rng=rng)
    elif phase in ("Knock", "Layoff", "Wall"):
        chosen = _choose_meld_or_layoff(gs, classified, _dead_set(gs), sample=sample, rng=rng)
    else:
        # Unknown phase: try draw → discard → first action.
        chosen = (
            _choose_draw(gs, classified)
            or _choose_discard(gs, classified, _dead_set(gs), sample=sample, rng=rng)
            or min(legal.keys())
        )

    if chosen is None:
        chosen = min(legal.keys())
    return chosen


def get_expert_action_b(
    messages: list[dict],
) -> str:
    """Persona B (second teacher): same DP-optimal engine + immediate-knock, but
    the discard is SAMPLED among near-best cards (bounded mixing diversity for
    league self-play). Falls back to smallest legal id. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
    forced = len(raw) == 1
    try:
        gs = parse_game_state(last_user)
    except Exception:
        set_meta(qgap=None, forced=forced)
        return min(raw, key=int) if raw else "0"
    chosen = _select_action_id(gs, last_user, sample=True)
    if chosen is None:
        set_meta(qgap=None, forced=forced)
        return min(raw, key=int) if raw else "0"
    # persona B SAMPLES the discard -> no deterministic best-vs-alt gap.
    set_meta(qgap=None, forced=forced)
    return str(chosen)


def get_expert_action(
    messages: list[dict],
) -> str:
    """Pick action via DP-optimal heuristic. Falls back to smallest legal id."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
    forced = len(raw) == 1  # exactly one legal action -> no strategic signal
    try:
        gs = parse_game_state(last_user)
    except Exception:
        # State parser failed: return smallest legal id from the raw label list.
        set_meta(qgap=None, forced=forced)
        return min(raw, key=int) if raw else "0"

    chosen = _select_action_id(gs, last_user)
    if chosen is None:
        set_meta(qgap=None, forced=forced)
        return min(raw, key=int) if raw else "0"
    phase = (getattr(gs, "phase", "") or "").strip()
    qgap = _gin_discard_qgap(gs) if phase == "Discard" else None
    set_meta(qgap=qgap, forced=forced)
    return str(chosen)


# ---------------------------------------------------------------------------
# Harness adapter
# ---------------------------------------------------------------------------

class GinStrongBot:
    """Teacher policy over a live pyspiel state.

    Constructed once per generation worker and called for every teacher turn. `step` returns a legal
    action id, or None when the state offers no legal action at all.

    The state is handed to the decision engine as rendered text rather than as the pyspiel object.
    That is deliberate: the engine then reads the same board the student will read at inference, so a
    detail the prompt does not carry cannot influence a demonstration the student is expected to
    reproduce. The rendering carries the observation string, which supplies hand, deadwood, phase,
    knock card, stock size, upcard and discard pile, followed by the legal-action block in the
    `id -> label` form the parser expects.

    Failure is contained rather than propagated. The engine never raises, and anything that does
    escape falls back to the lowest legal id, so a malformed state costs one weak move instead of an
    aborted game. The returned id is re-checked against the live legal set before it is handed back.
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.last_reason = None

    def step(self, state):
        self.last_reason = None
        player = state.current_player()
        ids = state.legal_actions(player)
        if not ids:
            return None
        lines = '\n'.join(f'{a} -> {state.action_to_string(player, a)}' for a in ids)
        text = extract_and_format_observation(
            f'You are Player {player}.\n{state.observation_string(player)}\nLegal Actions:\n{lines}'
        )
        try:
            chosen = int(get_expert_action([{'role': 'user', 'content': text}]))
        except Exception:
            return min(ids)
        chosen = chosen if chosen in ids else min(ids)
        # The decision is final at this point; the sentence below only describes it. It is
        # computed after the legality re-check so it can never disagree with the id returned.
        try:
            legal_labels = _parse_legal_actions(text)  # same parser as the engine; seat prefix stripped
            kind = _classify_action(legal_labels[chosen])['kind']
            gs = parse_game_state(text)
            ranking: list[int] = []
            if kind in ('discard', 'knock'):
                classified = {aid: _classify_action(lbl) for aid, lbl in legal_labels.items()}
                ranking = (
                    _rank_discard_ids(gs, classified, chosen=chosen) if kind == 'discard'
                    else _rank_knock_ids(gs, classified, chosen=chosen)
                )
                # Both rankers pin `chosen` to the front themselves; only emit the
                # sentence when it actually leads (never-raise: silently drop it
                # rather than risk a mismatch between the text and the played id).
                if not ranking or ranking[0] != chosen:
                    ranking = []
            self.last_reason = describe_decision(gs, kind, ranking)
        except Exception:
            self.last_reason = None
        return chosen


def describe_decision(gs, kind: str, ranking: "list[int] | None" = None) -> str:
    """One-sentence rationale for the move the engine just chose, written for the student.

    The sentence is built from the phase and the action kind, and from how the hand's deadwood
    stands relative to the knock card. Each wording mirrors the rule that actually fires in the
    chooser for that branch (`_choose_draw` takes the upcard only when optimal deadwood strictly
    drops; the meld/layoff chooser declares partition melds first, then the exact-delta layoff or
    forced discard; the knock decision is deadwood-vs-knock-card), so the text is a faithful
    account of the engine, not a story told after the fact.

    It never names a card, because a card token in the rationale is the most likely thing a small
    model copies into the action field. For the two decisions where the engine already scores
    several legal candidates against each other — Discard-phase discard, and the Knock-vs-continue
    choice — `ranking` (built by `_rank_discard_ids`/`_rank_knock_ids` in `GinStrongBot.step`) is
    appended as an explicit best-first list. Every integer in that list is one of the row's OWN
    legal action ids (never a card label, never a decoy), and the chosen id — the one the trajectory
    builder emits as `action_id` — always leads the list, so the ranking can only ever restate what
    was played plus the runner-up(s), never introduce something the student could copy into an
    illegal action. This is the same "expose the ranking, not just the argmax" target-format used to
    strengthen behavioural cloning elsewhere in the pipeline. Every other kind stays argmax-only.
    """
    phase = (getattr(gs, 'phase', '') or '').strip()
    deadwood = getattr(gs, 'deadwood', None)
    knock_card = getattr(gs, 'knock_card', None)
    can_knock = deadwood is not None and knock_card is not None and deadwood <= knock_card
    if kind == 'gin':
        return 'Every card in my hand is melded, so I declare gin.'
    if kind == 'knock':
        base = 'My deadwood is at or below the knock card, so I knock.'
        if ranking and len(ranking) >= 2:
            ranked = ', '.join(str(a) for a in ranking)
            base += f' Ranked knock-vs-continue, this id first: {ranked}.'
        return base
    if kind == 'draw_upcard':
        return 'Taking the upcard lowers my best deadwood, so I take it.'
    if kind == 'draw_stock':
        return 'The upcard does not improve my hand, so I draw from the stock.'
    if kind == 'meld':
        return 'Declaring this meld so its cards no longer count as deadwood.'
    if kind == 'pass':
        if phase in ('Draw', 'FirstUpcard'):
            return 'The first upcard does not help my hand, so I pass.'
        return 'I have nothing left to declare, so I pass.'
    if kind == 'discard':
        if phase == 'Layoff':
            return 'Laying this card onto the knocker\'s meld so it stops counting as deadwood.'
        if phase == 'Knock':
            return 'Discarding the card that leaves the lowest deadwood before the melds are declared.'
        if phase == 'Wall':
            return 'The stock is nearly gone, so I discard the card that lowers my deadwood most.'
        base = (
            'My deadwood is already within the knock card; discarding the card that keeps it lowest.'
            if can_knock else
            'My deadwood is still above the knock card, so I discard the card that lowers it most.'
        )
        if ranking and len(ranking) >= 2:
            ranked = ', '.join(str(a) for a in ranking)
            base += f' Ranked by resulting deadwood, this id first: {ranked}.'
        return base
    return 'Choosing the strongest legal action for this phase.'
