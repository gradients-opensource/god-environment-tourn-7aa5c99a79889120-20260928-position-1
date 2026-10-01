from __future__ import annotations
import argparse
import re
from envs import pvp_toolcalling as tc
VALIDATION_RATIO = 0.01
_OPPONENT_SIMS = {'liars_dice': 225, 'leduc_poker': 50, 'gin_rummy': 50, 'othello': 50, 'goofspiel': 50, 'clobber': 50}
_TEACHER_SIMS = {'liars_dice': 400, 'leduc_poker': 400, 'gin_rummy': 200, 'othello': 400, 'goofspiel': 200, 'clobber': 400}
_MCTS_UCT_C = 2.0
# Goofspiel deck sizes — mirror the validator's GoofspielAgent (_GOOFSPIEL_NUM_CARDS);
# selected per game from the config id so each game varies board size (and length).
_GOOFSPIEL_NUM_CARDS = (5, 8, 10, 13)
# Clobber board variants — mirror the validator's ClobberAgent (_CLOBBER_BOARD_SIZES),
# selected per game from the config id like goofspiel's deck size.
_CLOBBER_BOARD_SIZES = ((4, 5), (5, 5), (5, 6))
# Games whose game_action rows end with an explicit "action_id N" sentence. Membership follows
# the action-label format, not the game's strength: a game is listed when its legal-action
# labels (or the teacher's rationale) expose integers other than the id that a model could
# copy into the tool argument. See the anchor block in _play_matchup for the per-game reading.
_ANCHORED_GAMES = ('gin_rummy', 'liars_dice', 'leduc_poker', 'othello', 'goofspiel')

def _game_params(game_name: str, config_id: int) -> tuple[str, dict]:
    if game_name == 'liars_dice':
        return ('liars_dice', {'players': 2, 'numdice': 5})
    if game_name == 'leduc_poker':
        return ('leduc_poker', {'players': 2})
    if game_name == 'gin_rummy':
        hand_var = config_id // 3 % 3
        knock_var = config_id % 3
        return ('gin_rummy', {'hand_size': 7 + hand_var, 'knock_card': 10 - knock_var})
    if game_name == 'othello':
        return ('othello', {})
    if game_name == 'clobber':
        (rows, cols) = _CLOBBER_BOARD_SIZES[config_id % len(_CLOBBER_BOARD_SIZES)]
        return ('clobber', {'rows': rows, 'columns': cols})
    if game_name == 'goofspiel':
        num_cards = _GOOFSPIEL_NUM_CARDS[config_id % len(_GOOFSPIEL_NUM_CARDS)]
        return ('goofspiel', {'players': 2, 'num_cards': num_cards, 'imp_info': True, 'points_order': 'random', 'returns_type': 'win_loss'})
    raise ValueError(f'unsupported game: {game_name!r}')

def _load_pyspiel_game(game_name: str, name: str, params: dict):
    """Load the pyspiel game; wrap goofspiel's simultaneous moves into sequential
    turns exactly as the validator's GoofspielAgent.load_game does."""
    import pyspiel
    game = pyspiel.load_game(name, params)
    if game_name == 'goofspiel':
        game = pyspiel.convert_to_turn_based(game)
    return game

def _format_state(game_name: str, state, player_id: int) -> str:
    if game_name == 'gin_rummy':
        return state.observation_string(player_id)
    if game_name == 'liars_dice':
        return _format_liars_dice(state, player_id)
    if game_name == 'leduc_poker':
        return _format_leduc(state, player_id)
    if game_name == 'othello':
        colour = 'x (Black)' if player_id == 0 else 'o (White)'
        return f'You play {colour}.\n{state.observation_string(player_id)}'
    if game_name == 'clobber':
        colour = 'o (White)' if player_id == 0 else 'x (Black)'
        return f'You play {colour}.\n{state.observation_string(player_id)}'
    if game_name == 'goofspiel':
        return f'You are Player {player_id} (P{player_id}).\n{state.observation_string(player_id)}'
    raise ValueError(f'unsupported game: {game_name!r}')

def _format_liars_dice(state, player_id: int) -> str:
    try:
        info_str = state.information_state_string(player_id)
    except (RuntimeError, AttributeError):
        return str(state)
    if not info_str:
        return str(state)
    parts = info_str.split()
    dice_part = parts[0]
    bid_parts = [p for p in parts[1:] if '-' in p]
    dice = [int(d) for d in dice_part if d.isdigit()]
    num_dice = len(dice)
    total_dice = num_dice * state.num_players()
    lines = [f"Your dice: {dice} (showing: {', '.join(map(str, dice))})", f'Dice per player: {num_dice}', f'Total dice in game: {total_dice}', f'Players: {state.num_players()}', f'Current player: Player {state.current_player()}']
    if bid_parts:
        (quantity, face) = bid_parts[-1].split('-')
        lines.append(f'\nCurrent bid: "{quantity}-{face}" (at least {quantity} dice showing {face} across all players)')
        lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    else:
        lines.append('No bid yet - you must make the first bid')
    return '\n'.join(lines)

def _leduc_card_name(card_id: int) -> str:
    ranks = ['J', 'Q', 'K', 'A']
    suits = ['♠', '♥']
    (rank_idx, suit_idx) = (card_id // 2, card_id % 2)
    return f'{ranks[rank_idx]}{suits[suit_idx]}' if rank_idx < len(ranks) else f'Card_{card_id}'

def _leduc_betting(seq: str) -> str:
    if not seq or not seq.strip():
        return '(none)'
    actions_map = {0: 'Fold', 1: 'Call', 2: 'Raise'}
    nums = [int(x) for x in seq.split() if x.isdigit()]
    return ', '.join((actions_map.get(a, f'Action{a}') for a in nums)) if nums else '(none)'

def _format_leduc(state, player_id: int) -> str:
    try:
        info_str = state.information_state_string(player_id)
    except (RuntimeError, AttributeError):
        return str(state)

    def ext(pattern: str) -> str:
        m = re.search(pattern, info_str)
        return m.group(1) if m else ''
    private_card = ext('\\[Private: (-?\\d+)\\]')
    round_num = ext('\\[Round (\\d+)\\]')
    pot = ext('\\[Pot: (\\d+)\\]')
    money = ext('\\[Money: ([\\d ]+)\\]')
    public_card = ext('\\[Public: (-?\\d+)\\]')
    round1_seq = ext('\\[Round1: ([^\\]]*)\\]')
    round2_seq = ext('\\[Round2: ([^\\]]*)\\]')
    lines: list[str] = []
    if private_card and private_card != '-10000':
        lines.append(f'Your card: {_leduc_card_name(int(private_card))}')
    else:
        lines.append('Your card: (not dealt yet)')
    if public_card and public_card != '-10000':
        lines.append(f'Public card: {_leduc_card_name(int(public_card))}')
        if private_card and private_card != '-10000' and (int(private_card) // 2 == int(public_card) // 2):
            lines.append('Hand: PAIR')
    lines.append(f'Round: {round_num}/2')
    lines.append(f'Pot: {pot} chips')
    if money:
        chips = money.split()
        if len(chips) >= 2:
            lines.append(f'Your chips: {chips[player_id]}')
            lines.append(f'Opponent chips: {chips[1 - player_id]}')
    if round1_seq:
        lines.append(f'Round 1 actions: {_leduc_betting(round1_seq)}')
    if round2_seq:
        lines.append(f'Round 2 actions: {_leduc_betting(round2_seq)}')
    return '\n'.join(lines)

def _make_mcts_bot(game, simulations: int, seed: int):
    import numpy as np
    from open_spiel.python.algorithms import mcts

    class _SafeRolloutEvaluator(mcts.RandomRolloutEvaluator):

        def evaluate(self, state):
            result = None
            for _ in range(self.n_rollouts):
                ws = state.clone()
                while not ws.is_terminal():
                    if ws.is_chance_node():
                        outcomes = ws.chance_outcomes()
                        (actions, probs) = zip(*outcomes)
                        ws.apply_action(self._random_state.choice(actions, p=probs))
                    else:
                        legal = ws.legal_actions()
                        if not legal:
                            break
                        ws.apply_action(self._random_state.choice(legal))
                returns = np.array(ws.returns())
                result = returns if result is None else result + returns
            return result / self.n_rollouts
    evaluator = _SafeRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(seed))
    return mcts.MCTSBot(game, uct_c=_MCTS_UCT_C, max_simulations=simulations, evaluator=evaluator, random_state=np.random.RandomState(seed))

def _make_teacher(game_name: str, game, teacher_sims: int, seed: int):
    import os
    if game_name == 'liars_dice':
        _lt = os.environ.get('LIARS_TEACHER', 'posterior').strip().lower()
        if _lt == 'posterior':
            # Posterior-binomial bid-reading stop-loss (local head-to-head): recovers the liars
            # head-to-head from 0.263 (flat-binomial) to 0.508 by inferring the opponent's
            # dice from their bid history. Near-Nash game → ~0.5 is the ceiling.
            from envs.mastering_liars import LiarsPosteriorBot
            return LiarsPosteriorBot(seed)
    if game_name == 'gin_rummy':
        _gt = os.environ.get('GIN_TEACHER', 'strong').strip().lower()
        if _gt == 'strong':
            # Stronger gin engine (local head-to-head): the prior GinExpert lost the P0 initiative
            # (0.167 — a strong opponent exploits the initiative + knocks decisively); this
            # exact-deadwood-DP + pile-safe discard + knock-timing recovers to 0.542.
            from envs.mastering_gin import GinStrongBot
            return GinStrongBot(seed)
    if game_name == 'leduc_poker':
        _lt = os.environ.get('LEDUC_TEACHER', 'handwin').strip().lower()
        if _lt == 'handwin':
            # Hand-win best-response (offline-measured 0.709 vs a near-Nash opponent);
            # the metric is hand-win not chip-EV, so CFR washes but this exploits.
            from envs.mastering_leduc import LeducHandWinBot
            return LeducHandWinBot(seed)
    # In-code default 'oneup' (matches the LIARS/GIN/LEDUC/OTHELLO pattern). Both goofspiel
    # branches in HEAD defaulted to '', so an invocation with GOOFSPIEL_TEACHER unset matched NO
    # branch and fell through to _make_mcts_bot(teacher_sims=200) — the unlearnable teacher —
    # SILENTLY. So for the UNSET case this flips MCTS-200 -> oneup; the valuematch -> oneup flip
    # is the separate TOURNAMENT_TEACHERS default in sft_env_config.py. The tournament path always
    # sets the var via the ${:-} shell prefix, but manual and local eval runs did not, so the
    # thing we measured by hand was not the thing we shipped.
    # Still unhandled on purpose: an EMPTY or misspelled GOOFSPIEL_TEACHER matches no branch and
    # falls through to MCTS-200 as before — a typo fails silently rather than loudly.
    if game_name == 'goofspiel' and os.environ.get('GOOFSPIEL_TEACHER', 'oneup').strip().lower() == 'oneup':
        # SHIPPED DEFAULT since 2026-08-04. Bid the CHEAPEST card that still beats the prize;
        # concede with the lowest card when none does. Replaces valuematch, which cost us the
        # 2026-08-03 tournament: R2 per-env from the real pvp_pair_results was liars 0.534 /
        # leduc 0.500 / GOOFSPIEL 0.298 — beaten by every opponent, burn hotkey included.
        # The old choice was gated on teacher-vs-MCTS, but PvP scores us against other MINERS,
        # and the policy space is an RPS cycle (valuematch ← oneup ← oneup_rank ← twoup_rank ←)
        # where value-match's counter is the first one anyone tries and is total: it always bids
        # exactly the prize, so one-up takes every round but the dearest — 1+..+(N-1) > N for the
        # tournament decks (N=5/8/10/13) → 0.000 for value-match at EVERY size. Offline lab on the
        # exact params, 3200 seat-swapped games/pair: worst case 0.374 vs 0.000, meta-Nash
        # (fictitious play) 0.500 vs 0.206 — the lab predicted 0.206, the tournament gave 0.298.
        # Robust to imperfect execution too: rank-1 by worst case at eps=0/.05/.10/.20 and still
        # 0.830 over value-match at eps=0.20. Learnable: a strict function of (prize, hand), and a
        # simpler rule than 'closest by value'. GATE ON local student head-to-head before trusting.
        from envs.mastering_goofspiel import GoofspielOneUpBot
        return GoofspielOneUpBot(seed)
    if game_name == 'goofspiel' and os.environ.get('GOOFSPIEL_TEACHER', '').strip().lower() == 'valuematch':
        # DETERMINISTIC value-matching (bid the card closest to the prize) + chain-of-thought.
        # THE LEARNABLE choice: MCTS-800 is the strongest goofspiel BOT but is STOCHASTIC on the
        # same info-state (measured 2026-07-06: one bot, same state, 12 calls → bids flip 1/2/2/1…),
        # so a 1.5B distils NOISE — the same othello-MCTS-800 failure mode (student 0.179). value-match
        # is a strict function of (prize, hand) → same state, same bid → learnable, AND it's the
        # strongest deterministic generalist: beats an LP-Nash policy 0.78 (nc5)/0.91
        # (nc8) and is the most competitive deterministic option vs MCTS-800 (loses only 0.26–0.28),
        # where LP-Nash's deterministic Nash-mode is exploitable and loses even to value-match
        # (0.11–0.24). All live pyspiel head-to-head, seat-swapped, imp_info+random-order+win_loss.
        # GATE ON local student head-to-head EVAL, never teacher-vs-MCTS (that metric picked unlearnable MCTS-800).
        from envs.mastering_goofspiel import GoofspielValueMatchBot
        return GoofspielValueMatchBot(seed)
    if game_name == 'goofspiel' and os.environ.get('GOOFSPIEL_TEACHER', '').strip().lower() == 'mcts':
        # STRONGEST BOT but UNLEARNABLE: MCTS-800 is stochastic on the same info-state (student
        # distils contradictory bids → noise). A/B backstop only — the shipped default is ONEUP
        # since 2026-08-04 (was valuematch, which lost R2). The exploiter below scores only 0.33 vs a generic
        # MCTS-50 (measured); MCTS-800 scores 0.56 vs MCTS-50 (higher sims 1600/3000 don't help →
        # goofspiel is low-ceiling/near-Nash), so as a raw bot it never loses — but it can't teach.
        return _make_mcts_bot(game, 800, seed)
    if game_name == 'othello':
        _ot = os.environ.get('OTHELLO_TEACHER', 'positional').strip().lower()
        if _ot == 'positional_corner':
            # OPT-IN A/B, default stays 'positional'. Same 3 rules, but the X/C-square penalty
            # only applies while the adjacent corner is still EMPTY (textbook othello: the
            # square is only bad because it can concede that corner). Motivation: the shipped
            # teacher never reads the board at all, while the student is shown it — and othello
            # had the WIDEST field spread of any env measured (tourn ...20260727, forfeit-free
            # 0.579/0.638/0.750 over three R1 groups) yet we target a DRAW there. Adds no search,
            # so the learnability constraint that rejected MCTS-800 and the mobility look-ahead
            # still holds. GATE ON local student head-to-head before promoting to default.
            from envs.mastering_othello import OthelloCornerAwareTeacher
            return OthelloCornerAwareTeacher(seed)
        if _ot in ('positional', 'weighttable'):
            # Static positional weight-table heuristic (corner-priority > avoid X/C > highest-weight
            # cell) + a short rationale. The label is a pure function of (position, legal set), so a
            # 1.5B student can reproduce it at inference — unlike a mobility look-ahead, which needs a
            # search the student can't run and which produced a weaker student despite a stronger
            # teacher-vs-MCTS score. Gate on local head-to-head eval. See mastering_othello.py.
            from envs.mastering_othello import OthelloPositionalTeacher
            return OthelloPositionalTeacher(seed)
        if _ot == 'expert':
            # opaque MCTS-800: strong vs MCTS-50 (0.94) but UNLEARNABLE at 1.5B (0.179 head-to-head). A/B only.
            return _make_mcts_bot(game, 800, seed)
    if game_name == 'clobber':
        _ct = os.environ.get('CLOBBER_TEACHER', 'negamax').strip().lower()
        if _ct in ('negamax', 'search'):
            # Iterative-deepening negamax alpha-beta with a MOBILITY-DIFFERENCE leaf eval
            # (mastering_clobber.pick_move), content=None (no CoT). An exact win/loss solver picks
            # the lowest-action-id winning move (an action-id ARTIFACT) + falls back to stochastic
            # MCTS → UNLEARNABLE; a mobility-argmax variant still measured ~0.40 in local head-to-head.
            # A negamax that scores leaves by mobility difference always returns the mobility-optimal
            # move — a smooth deterministic label the student can distill. Gate on local head-to-head.
            from envs.mastering_clobber import ClobberSearchTeacher
            return ClobberSearchTeacher(seed)
    return _make_mcts_bot(game, teacher_sims, seed)

def _action_lines(state, player_id: int, legal: list[int]) -> list[str]:
    lines = []
    for a in legal:
        try:
            lines.append(f'{a} -> {state.action_to_string(player_id, a)}')
        except (RuntimeError, AttributeError):
            lines.append(str(a))
    return lines

def play_game(game_name: str, game, teacher_sims: int, opponent_sims: int, model_seat: int, seed: int, long_term: dict | None=None):
    import random as _random
    teacher = _make_teacher(game_name, game, teacher_sims, seed + model_seat)
    # othello/clobber: match the target data-gen distribution — the opponent MCTS sim count is
    # sampled per game from the env's range (othello 25-75, clobber 50-100 —
    # {othello,clobber}_trajectories._MCTS_SIMS_MIN/_MAX), not a fixed count. A SEPARATE
    # seed-derived rng keeps the opening-ply stream below unchanged from before, so this
    # varies ONLY the opponent strength, nothing else.
    if game_name == 'othello':
        _opp_sims = _random.Random(seed * 2 + 1).randint(25, 75)
    elif game_name == 'clobber':
        _opp_sims = _random.Random(seed * 2 + 1).randint(50, 100)
    else:
        _opp_sims = opponent_sims
    opponent = _make_mcts_bot(game, _opp_sims, seed + (1 - model_seat))
    rng = _random.Random(seed)
    state = game.new_initial_state()
    if game_name in ('othello', 'clobber'):
        # Random opening plies widen the state COVERAGE the teacher labels — coverage gates
        # generalisation as much as teacher depth (searchless-chess: overfit @10k, generalises
        # @100k-1M; OOD-collapse 2510.20783). The max depth is a tunable coverage knob:
        # COVERAGE_OPENING_MAX (default 6 = current). Raise it to sample deeper/more-diverse
        # openings; GATE ON EVAL (too-deep openings can drift off the teacher's competent range).
        import os as _os
        _open_max = max(2, int(_os.environ.get('COVERAGE_OPENING_MAX', '6')))
        for _ in range(rng.randint(2, _open_max)):
            if state.is_terminal():
                break
            legal_open = state.legal_actions()
            if not legal_open:
                break
            state.apply_action(rng.choice(legal_open))
    examples = []
    opp_actions: list[str] = []
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            (actions, probs) = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cur = state.current_player()
        if cur == model_seat:
            legal = state.legal_actions(model_seat)
            state_desc = _format_state(game_name, state, model_seat)
            lines = _action_lines(state, model_seat, legal)
            action = teacher.step(state)
            if action is None:
                # Teacher has no legal move at this state (rare gin edge: the acting seat's
                # legal_actions is empty). Skip this game cleanly instead of crashing on
                # int(None) inside the row builder — was a noisy TypeError (game skipped anyway,
                # ~0.4%); this keeps the log clean and never emits a malformed example.
                raise RuntimeError('teacher.step returned None (no legal action)')
            _content = getattr(teacher, 'last_reason', None)
            if game_name in _ANCHORED_GAMES:
                # Anchor the assistant text on the integer the tool call must carry. The legal-action
                # block shows every move as `id -> label`, and in the games listed above the label
                # itself contains standalone integers that are not action ids: gin prefixes every
                # label with the seat (`Player: 0 Action: ...`), liars renders bids as `3-2` while the
                # state line prints the current bid in exactly the same shape, and goofspiel's label
                # `Bid: 5` is the card value, one above its id. A model that copies any of those into
                # `action_id` plays an illegal move (gin ~90% of turns, liars ~73%, goofspiel ~50%,
                # measured over the tournament game parameters) or, for goofspiel, a legal but wrong
                # one. Games whose labels carry no bare integer — leduc (`Call`/`Raise`), othello and
                # clobber (coordinates such as `d3`, `a1-b1`) — showed no illegal-copy forfeits, so
                # othello keeps the anchor only because its rationale names board squares.
                # The sentence is appended after the teacher's rationale, never in place of it: the
                # rationale is what carries the strategy, the anchor is what maps it to the id. For
                # gin the rationale is now produced on every turn (mastering_gin.describe_decision),
                # so each row demonstrates prose and a tool call in the same assistant turn; that is
                # the behaviour the rules text ("respond with the action ID number ONLY") would
                # otherwise steer a small model away from, and a turn with no tool call is a forfeit.
                _anchor = f'I will call game_action with action_id {action}.'
                _content = _anchor if _content is None else f'{_content} {_anchor}'
            examples.append(tc.build_toolcalling_example(game_name, state_desc, model_seat, lines, legal, action, long_term=long_term, content=_content))
            state.apply_action(action)
        else:
            try:
                opp_action = opponent.step(state)
            except Exception:
                legal_opp = state.legal_actions(cur)
                if not legal_opp:
                    raise
                opp_action = rng.choice(legal_opp)
            try:
                opp_actions.append(state.action_to_string(cur, opp_action))
            except (RuntimeError, AttributeError):
                pass
            state.apply_action(opp_action)
    final_state_desc = _format_state(game_name, state, model_seat)
    return (examples, state.returns()[model_seat], opp_actions, final_state_desc)
_NOTE_HINTS = {'liars_dice': {'Liar': 'they call Liar readily — bid conservatively, avoid thin bluffs', 'default': 'they keep raising bids — call Liar on implausible quantities'}, 'leduc_poker': {'Raise': "aggressive raiser — call down with medium hands, don't over-fold", 'Fold': 'folds easily — apply pressure and bluff more', 'default': 'balanced — value-bet strong hands, fold weak vs aggression'}, 'gin_rummy': {'default': 'track their discards; keep deadwood low and knock early'}}

def _synthesize_note(game_name: str, outcome: str, opp_counter, game_idx: int) -> str:
    top = opp_counter.most_common(3)
    freq = ', '.join((f'{a} x{c}' for (a, c) in top)) if top else 'n/a'
    hints = _NOTE_HINTS.get(game_name, {'default': 'adapt to their tendencies'})
    dom = top[0][0] if top else ''
    hint = hints['default']
    for (k, v) in hints.items():
        if k != 'default' and k in dom:
            hint = v
            break
    return f'Opponent read (after {game_idx + 1} games, last {outcome}): frequent actions {freq}. Counter: {hint}.'

def play_matchup(game_name: str, matchup_games: int, teacher_sims: int, opponent_sims: int, base_seed: int):
    import pyspiel
    from collections import Counter
    long_term: dict = {}
    opp_counter = Counter()
    n_slots = tc.PVP_LONGTERM_MEM_SLOTS
    game_action_groups = []
    reflection_examples = []
    skipped = 0   # games lost to the per-game except below; returned so main() can report a RATE
    for g in range(matchup_games):
        model_seat = g % 2
        (name, params) = _game_params(game_name, base_seed % 1000)
        game = _load_pyspiel_game(game_name, name, params)
        try:
            (exs, reward, opp_acts, final_sd) = play_game(game_name, game, teacher_sims, opponent_sims, model_seat, base_seed + g, long_term=dict(long_term))
        except Exception as exc:
            # Counted, not just printed. A broken teacher (ImportError inside _make_teacher, a
            # step() returning None, bad game params) fails on EVERY game, and one printed line
            # per skip is indistinguishable from sporadic noise once it is buried in a 25k-line
            # log. The count is returned so the caller can report a RATE — which is the number
            # that tells you WHY a run under-filled.
            skipped += 1
            print(f'[pvp-tc] skip game (seed {base_seed + g}): {type(exc).__name__}: {exc}', flush=True)
            continue
        game_action_groups.append((exs, reward))
        opp_counter.update(opp_acts)
        outcome = 'win' if reward > 0 else 'loss' if reward < 0 else 'draw'
        note = _synthesize_note(game_name, outcome, opp_counter, g)
        slot = g % n_slots + 1
        reflection_examples.append(tc.build_reflection_example(game_name, outcome, final_sd, slot, note, op='rewrite', long_term=dict(long_term)))
        long_term[slot] = note
    return (game_action_groups, reflection_examples, skipped)

def _row(example: dict) -> dict:
    import json
    return {'messages': example['messages'], 'tools_json': json.dumps(example['tools'])}

def main() -> None:
    import time
    _t0 = time.time()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', required=True, choices=['liars_dice', 'leduc_poker', 'gin_rummy', 'othello', 'goofspiel', 'clobber'])
    p.add_argument('--output_path', default=None, help="DatasetDict save dir (required unless --check, which doesn't save).")
    p.add_argument('--num_games', type=int, default=2000)
    p.add_argument('--target_samples', type=int, default=0, help='Early-stop once this many training rows are collected (0 = no limit). num_games then acts only as an UPPER BOUND, so an over-estimated num_games no longer over-generates — critical to fit the 1.5h tournament budget.')
    p.add_argument('--matchup_games', type=int, default=4, help='Games per opponent matchup. Long-term memory + reflection persist across the matchup (the env_memory_system thesis).')
    p.add_argument('--teacher_sims', type=int, default=0, help='MCTS simulations for the teacher (model side). 0 = per-game default (_TEACHER_SIMS: gin 100, others 400). Stronger than the opponent so it wins → winning expert moves.')
    p.add_argument('--opponent_sims', type=int, default=0, help='MCTS sims for the opponent (0 = eval strength per game).')
    p.add_argument('--workers', type=int, default=0, help='Parallel matchup workers (0 = auto = min(cpu_count,16); 1 = sequential). MCTS generation is CPU-bound, so this is a near-linear speedup.')
    p.add_argument('--wins-only', action='store_true', help='Keep game_action examples only from games the teacher won (score > 0).')
    p.add_argument('--include-draws', action='store_true', help='With --wins-only, also keep draws (score == 0).')
    p.add_argument('--no-reflection', action='store_true', help='Skip reflection trajectories (emit game_action examples only).')
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--time_limit_seconds', type=float, default=0.0, help="Hard wall-clock cap for THIS env's generation: stops submitting matchups once exceeded, keeping whatever rows it has. Bounds slow teachers (e.g. the clobber a-b solver, long gin games) so generation can never overrun and starve training. 0 = no cap (count-bounded only).")
    p.add_argument('--check', type=int, default=0, help="Smoke mode: play one small matchup, print samples + stats, don't save.")
    args = p.parse_args()
    if not args.check and (not args.output_path):
        p.error('--output_path is required unless --check')
    import random as _random
    opponent_sims = args.opponent_sims or _OPPONENT_SIMS[args.game]
    teacher_sims = args.teacher_sims or _TEACHER_SIMS.get(args.game, 400)
    seed_rng = _random.Random(args.seed)
    if args.check:
        (ga_groups, refl, _chk_skipped) = play_matchup(args.game, max(2, args.check), teacher_sims, opponent_sims, seed_rng.randint(1, 2000000000))
        ga = [e for (exs, _) in ga_groups for e in exs]
        print(f'[check] {args.game}: {len(ga_groups)} games | {len(ga)} game_action + {len(refl)} reflection examples')
        if ga:
            ex = ga[-1]
            sysp = ex['messages'][0]['content']
            lt = sysp[sysp.index('LONG_TERM'):]
            print('\n--- game_action: LONG_TERM memory block (later game) ---\n' + lt[:320])
            print('\n--- game_action tool_call ---\n' + str(ex['messages'][2]['tool_calls'][0]['function']))
        if refl:
            r = refl[-1]
            print('\n--- reflection user ---\n' + r['messages'][1]['content'])
            print('\n--- reflection writes long-term note ---\n' + str(r['messages'][2]['tool_calls'][0]['function']))
            print(f"--- reflection tools ({len(r['tools'])}): {[t['function']['name'] for t in r['tools']]} (no game_action) ---")
        print('\n[check] OK — not saving.')
        return
    import os
    n_matchups = max(1, args.num_games // max(1, args.matchup_games))
    seeds = [seed_rng.randint(1, 2000000000) for _ in range(n_matchups)]
    conversations = []
    wins = draws = losses = kept = 0
    _ga = [0]   # game_action rows actually emitted; a list so _consume mutates it without nonlocal
    _done_seeds = set()   # seeds whose matchup was fully consumed; lets the sequential fallback RESUME
    _skips = [0]   # games lost to the per-game except, summed across matchups AND workers
    _deadline = (_t0 + args.time_limit_seconds) if args.time_limit_seconds > 0 else None

    def _consume(idx, ga_groups, refl, skipped=0):
        nonlocal wins, draws, losses, kept
        # skipped defaults to 0 so any caller still passing the old 2-tuple keeps working.
        # _skips is a list for the same reason as _ga: mutated through the closure, no nonlocal.
        _skips[0] += skipped
        for (exs, reward) in ga_groups:
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
            if args.wins_only:
                is_loss = reward < 0 if args.include_draws else reward <= 0
                if is_loss:
                    continue
            # Materialise BEFORE extending: list.extend(generator) appends incrementally, so a
            # raise inside _row would leave partial rows in `conversations` while _ga[0] and
            # _done_seeds never update — and the sequential resume would then re-run that seed
            # and append the whole matchup again. HEAD was immune only because it cleared
            # everything on fallback; now that we resume, atomicity matters.
            _new_rows = [_row(e) for e in exs]
            conversations.extend(_new_rows)
            _ga[0] += len(_new_rows)   # game_action rows only — see the zero-move guard near the save
            kept += 1
        if not args.no_reflection:
            # NOTE the indentation: this sits OUTSIDE the per-game loop and OUTSIDE the
            # --wins-only filter above, so reflection rows survive even when every game was
            # dropped. That is deliberate (a reflection is emitted per game regardless of the
            # score filter) but it means len(conversations) alone CANNOT tell you whether the
            # model was ever shown a move. Hence _ga.
            conversations.extend((_row(e) for e in refl))
        if (idx + 1) % 10 == 0:
            print(f'[pvp-tc] {idx + 1}/{n_matchups} matchups | W-D-L {wins}-{draws}-{losses} | rows={len(conversations)}', flush=True)
    workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 16)
    use_parallel = workers > 1 and n_matchups > 1
    if use_parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f'[pvp-tc] generating {n_matchups} matchups across {workers} workers ...', flush=True)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # dict, not list: as_completed yields in COMPLETION order, so without a
                # future->seed map there is no way to know which seeds actually finished. That
                # is what forced the old fallback to throw everything away and restart.
                futs = {pool.submit(play_matchup, args.game, args.matchup_games, teacher_sims, opponent_sims, s): s for s in seeds}
                for (i, fut) in enumerate(as_completed(futs)):
                    _consume(i, *fut.result())
                    _done_seeds.add(futs[fut])
                    if _deadline and time.time() >= _deadline:
                        for f in futs:
                            f.cancel()
                        _rows, _tgt = len(conversations), args.target_samples
                        if _tgt and _rows < 0.9 * _tgt:
                            # UNDER-FILL: hit the wall-clock cap well short of target. The
                            # combine row-floor only DUPLICATES this thin set, it can't recover
                            # it — so make it a loud, visible signal (a future slow-teacher swap
                            # would otherwise silently starve this env then hide behind the floor).
                            print(f'[pvp-tc] ⚠️ UNDER-FILL {args.game}: time_limit {args.time_limit_seconds:.0f}s '
                                  f'reached with {_rows}/{_tgt} rows ({100 * _rows / _tgt:.0f}% of target, '
                                  f'{i + 1}/{n_matchups} matchups) — time-box starved.', flush=True)
                        else:
                            print(f'[pvp-tc] time_limit {args.time_limit_seconds:.0f}s reached ({_rows} rows, {i + 1}/{n_matchups} matchups) → stopping early', flush=True)
                        break
                    if args.target_samples and len(conversations) >= args.target_samples:
                        for f in futs:
                            f.cancel()
                        print(f'[pvp-tc] target_samples {args.target_samples} reached ({len(conversations)} rows) → stopping early', flush=True)
                        break
        except Exception as exc:
            print(f'[pvp-tc] parallel failed ({type(exc).__name__}: {exc}); falling back to sequential', flush=True)
            use_parallel = False
            # KEEP what the pool already produced, and run only the seeds it never finished.
            # The old code did conversations.clear() + reset every counter, then re-ran ALL
            # seeds — necessary back when there was no future->seed map, but it interacts badly
            # with the time box: _deadline is NOT reset (correctly — resetting it would let
            # generation run past the budget it exists to enforce), and the sequential loop
            # checks the deadline only AFTER its first _consume. So a pool that dies near the
            # cap used to yield exactly ONE matchup, ~4 games of rows, exit 0 — and
            # combine_datasets' row-floor would then OVERSAMPLE that handful up to half the
            # training mix. Resuming instead preserves both the finished work and the budget.
            seeds = [s for s in seeds if s not in _done_seeds]
            print(f'[pvp-tc] resuming sequentially: {len(_done_seeds)} matchups already done '
                  f'({len(conversations)} rows kept), {len(seeds)} remaining', flush=True)
    if not use_parallel:
        for (i, s) in enumerate(seeds):
            _consume(i, *play_matchup(args.game, args.matchup_games, teacher_sims, opponent_sims, s))
            if _deadline and time.time() >= _deadline:
                print(f'[pvp-tc] time_limit {args.time_limit_seconds:.0f}s reached ({len(conversations)} rows) → stopping early', flush=True)
                break
            if args.target_samples and len(conversations) >= args.target_samples:
                print(f'[pvp-tc] target_samples {args.target_samples} reached ({len(conversations)} rows) → stopping early', flush=True)
                break
    print(f'[pvp-tc] {args.game}: {n_matchups} matchups ({n_matchups * args.matchup_games} games) W-D-L {wins}-{draws}-{losses}; kept {kept} games; {len(conversations)} rows in {time.time() - _t0:.0f}s', flush=True)
    # ── FAIL LOUD HERE. Every early-exit path (parallel deadline, sequential deadline,
    # target reached, matchups exhausted, parallel->sequential fallback) converges on this
    # point, which is why the checks live here and not inside any loop.
    if not conversations:
        raise RuntimeError('No tool-calling trajectories generated (all games filtered or empty).')
    _played = wins + draws + losses
    if _skips[0]:
        # THE NUMBER THAT SAYS *WHY* A RUN UNDER-FILLED. Every skip already printed its own
        # line, but 25k identical lines in a log read as noise; a rate reads as a diagnosis.
        # A broken teacher fails on EVERY game, so a rate near 100% means "fix the teacher",
        # while a few percent is ordinary flakiness. Reported, deliberately NOT raised: the
        # catastrophic case (every game skipped) already trips the zero-game_action guard
        # below, and failing the task outright ships the untrained base for EVERY env — the
        # trap the combine_datasets sys.exit fell into on 2026-08-06.
        _rate = 100.0 * _skips[0] / max(1, _skips[0] + _played)
        print(f'[pvp-tc] ⚠️ SKIPPED {_skips[0]} games ({_rate:.0f}% of {_skips[0] + _played} attempted) '
              f'for {args.game} — grep "skip game" above for the exception. A rate near 100% means '
              f'the teacher itself is broken, not the games.', flush=True)
    if _ga[0] == 0:
        # THE SILENT CATASTROPHE this guard exists for. Reflection rows are appended outside the
        # --wins-only filter (see the note at the extend above), so a teacher that loads fine but
        # never WINS yields a dataset made entirely of memory-tool rows: no game_action anywhere.
        # It used to save cleanly and exit 0 — and because --target_samples counts reflections
        # too, the log even printed "target_samples reached". Training on it teaches a turn with
        # no move, which forfeits every PvP turn. wins-only is on for goofspiel/gin/clobber, so
        # this is reachable in three of six envs.
        raise RuntimeError(
            f'{args.game}: {len(conversations)} rows but ZERO game_action rows — every game was '
            f'filtered out (W-D-L {wins}-{draws}-{losses}, kept {kept}). A reflection-only '
            f'dataset teaches the model to make no move. Refusing to save.')
    if args.target_samples and len(conversations) < args.target_samples:
        # Advisory, but say it NUMERICALLY and on EVERY path. The pre-existing UNDER-FILL warning
        # only fired on the parallel deadline branch, so the two commonest shortfalls — the
        # sequential fallback, and simply exhausting all matchups without reaching target — were
        # silent. num_games is provisioned from a GUESSED win rate (WIN_RATE_ESTIMATE), so a bad
        # guess under-fills every run forever with nothing in the log to say so.
        _pct = 100.0 * len(conversations) / args.target_samples
        print(f'[pvp-tc] ⚠️ UNDER-FILL {args.game}: {len(conversations)}/{args.target_samples} rows '
              f'({_pct:.0f}% of target), {_ga[0]} game_action + {len(conversations) - _ga[0]} reflection, '
              f'kept {kept} games. Check WIN_RATE_ESTIMATE/GAME_TARGETS or the time limit.', flush=True)
    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_list(conversations)
    splits = dataset.train_test_split(test_size=VALIDATION_RATIO, seed=args.seed)
    dd = DatasetDict({'train': splits['train'], 'validation': splits['test']})
    dd.save_to_disk(args.output_path)
    print(f'[pvp-tc] saved: {args.output_path}', flush=True)
if __name__ == '__main__':
    main()
