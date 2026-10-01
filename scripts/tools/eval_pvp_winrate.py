"""Local PvP-fidelity win-rate harness (model vs in-harness MCTS).

⚠ WHICH OPPONENT YOU ARE MEASURING AGAINST — read this before using a number to decide.
DEFAULT (no --opponent-url): the opponent is an in-harness MCTS bot. The tournament's PvP
scores us against ANOTHER MINER'S MODEL, and that difference is not academic — it is
precisely the failure that cost the 2026-08-03 tournament. Goofspiel's value-match teacher
looked fine on a bot benchmark and then scored 0.298 in real PvP, because the goofspiel
policy space is an RPS cycle and the field played the counter. An MCTS opponent never plays
the counter, so NO number from the default mode — student or teacher — can detect
counter-exploitation. Treat a good default-mode number as "not broken", never as "will win".

ARM-VS-ARM (--opponent-url): a second served model plays the other seat through the same
strict turn loop. This is the mode that CAN see counter-exploitation, and the only one whose
number speaks to the question "which teacher should we ship".

What it DOES measure, and measures well: TURN-MECHANICS FIDELITY (single call per turn,
forfeits, wall clock, token budget, memory slots, reflection) and ABSOLUTE strength against
a fixed bot. What it does NOT measure: where we land against the field. The repo rule "gate
on local student head-to-head" is satisfied by this tool only in the first sense — treat a
good number here as "not broken", never as "will win". Applies to the shipped goofspiel
one-up teacher too: it is theoretically sound and UNMEASURED against a real opponent.

ARM-MODE FIDELITY NOTES (checked against the validator's own runner, not assumed):
  * PAIRED SEEDS. validator/evaluation/pvp/game_runner.py plays every seed TWICE with the
    seats swapped ("Each seed is played twice with swapped positions for fairness"). Arm
    mode does the same: one seed is drawn per PAIR and replayed with model_seat flipped,
    so board luck cancels within the pair instead of only in the limit. --num-games must
    therefore be EVEN in arm mode. Default MCTS mode still draws a fresh seed per game,
    exactly as before, so historical numbers stay reproducible.
  * FORFEIT REFLECTION. The validator skips reflect() only for the player who FORFEITED;
    the survivor still consolidates memory. Arm mode matches that. Getting this wrong
    penalises whichever arm forfeits LESS, which is backwards.
  * LONG-TERM MEMORY. The validator builds long-term memory once per MATCHUP and keeps it
    for every game against that opponent. Arm mode faces one opponent for the whole run,
    so --matchup-games defaults to --num-games there. Resetting it more often suppresses
    exactly the accumulated opponent-reading that counter-exploitation is made of.
  * A win can come from the OPPONENT forfeiting rather than from play. The opponent's
    forfeit rate is reported separately for that reason — read it before believing a
    win-rate.

STRICT mode (default) mirrors the tournament bot in core/pvp/bot.py:
  * a turn is EXACTLY ONE model call — no nudge, no retry, no inner loop
  * the conversation is rebuilt fresh every turn (no assistant/tool history)
  * every memory tool call in the response is executed; the FIRST *legal*
    game_action wins (order-independent, later game_action calls are still
    validated if the earlier ones were illegal)
  * no legal game_action -> the turn forfeits -> the game is a loss
  * per-turn SIGALRM wall clock (PVP_TURN_TIMEOUT_SECONDS) -> forfeit
  * chat timeout after retries -> forfeit; server unreachable -> infra, aborts
  * memory slots are TOKEN-BUDGETED exactly like core/pvp/memory.py SlotMemory
    (rewrite drops the TAIL, append drops the HEAD/oldest), 128 tokens/slot
  * after every non-forfeited game the bot REFLECTS (core/pvp/bot.py reflect):
    a SECOND model call, memory tools only, max_tokens 384, its own 30s wall
    clock, all exceptions swallowed, never counted as a game turn

LENIENT mode (--lenient) restores the old multi-step nudge/retry loop. It is
DEBUG ONLY: it measurably overstates win-rate because it hands the model up to
PVP_MAX_INNER_STEPS attempts per turn that the tournament never gives.

Constants below are mirrored from the validator sources (read 2026-08-02):
  core/pvp/constants.py, core/models/pvp_models.py (ChatCompletionConfig,
  PvPEvalConfig), core/pvp/chat.py, core/pvp/bot.py.
"""
from __future__ import annotations
import argparse
import json
import random
import signal
from envs import pvp_toolcalling as tc
from envs import generate_pvp_toolcalling as gen

# --- tournament-faithful constants (source: core/pvp/constants.py) ----------
PVP_TURN_TIMEOUT_SECONDS = 30       # bot._wall_clock -> TurnTimeoutError -> forfeit
PVP_TURN_MAX_TOKENS = 512           # bot overrides config.max_tokens with this
PVP_REFLECTION_TIMEOUT_SECONDS = 30 # bot.reflect gets its OWN wall clock
PVP_REFLECTION_MAX_TOKENS = 384     # bot._reflection_config overrides max_tokens
PVP_HTTP_READ_TIMEOUT_SECONDS = 35  # sits ABOVE the wall clock on purpose
PVP_HTTP_MAX_RETRIES = 1            # chat._with_retries -> attempts = 1 + 1
PVP_RETRY_BACKOFF_CAP_SECONDS = 32
# PvPEvalConfig defaults: seed=42 is BOTH the game base seed and the chat seed.
PVP_DEFAULT_SEED = 42
PVP_DEFAULT_TEMPERATURE = 0.0

# --- lenient (legacy, debug-only) ------------------------------------------
_NUDGE = 'You did not call a tool. Call game_action with a legal action id to make your move.'
_MAX_INNER_STEPS = tc.PVP_MAX_INNER_STEPS

# --- forfeit taxonomy -------------------------------------------------------
FORFEIT_NO_TOOL_CALL = 'no_tool_call'        # response carried no tool calls
FORFEIT_NO_GAME_ACTION = 'no_game_action'    # tool calls, but none was game_action
FORFEIT_ILLEGAL_ACTION = 'illegal_action'    # game_action with unparseable/illegal id
FORFEIT_TURN_TIMEOUT = 'turn_timeout'        # SIGALRM wall clock
FORFEIT_CHAT_TIMEOUT = 'chat_timeout'        # inference timed out after retries
FORFEIT_CONTEXT_OVERFLOW = 'context_overflow'
FORFEIT_REASONS = (
    FORFEIT_NO_TOOL_CALL,
    FORFEIT_NO_GAME_ACTION,
    FORFEIT_ILLEGAL_ACTION,
    FORFEIT_TURN_TIMEOUT,
    FORFEIT_CHAT_TIMEOUT,
    FORFEIT_CONTEXT_OVERFLOW,
)


class TurnTimeoutError(Exception):
    """Per-turn wall clock exceeded (mirrors bot.TurnTimeoutError)."""


class ChatTimeoutForfeitError(Exception):
    """Inference timed out after retries — miner's fault, forfeits."""


class ContextOverflowError(Exception):
    """Input exceeded the model's context length — forfeits."""


class ModelUnreachableError(Exception):
    """Inference server unreachable — infra failure, NOT a forfeit (aborts)."""


class TurnStats:
    """Per-(model, env) turn accounting — the forfeit-rate we never measured."""

    def __init__(self, model: str, game: str):
        self.model = model
        self.game = game
        self.turns = 0
        self.forfeits = 0
        self.reasons = {reason: 0 for reason in FORFEIT_REASONS}
        self.games = 0
        self.games_lost_to_forfeit = 0

    def record_turn(self) -> None:
        self.turns += 1

    def record_forfeit(self, reason: str) -> None:
        self.forfeits += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    @property
    def forfeit_rate(self) -> float:
        return self.forfeits / self.turns if self.turns else 0.0

    def as_dict(self) -> dict:
        return {
            'model': self.model,
            'env': self.game,
            'turns': self.turns,
            'forfeit_turns': self.forfeits,
            'forfeit_rate': round(self.forfeit_rate, 6),
            'games': self.games,
            'games_lost_to_forfeit': self.games_lost_to_forfeit,
            'forfeit_reasons': {k: v for (k, v) in self.reasons.items() if v},
        }


class _Mem:
    """Fixed-slot, token-BUDGETED memory — mirror of core/pvp/memory.py SlotMemory.

    Budget semantics copied verbatim from the validator (memory.py):
      * rewrite -> _fit(content, keep='head'): drops the TAIL ("write tighter")
      * append  -> _fit(existing + '\n' + content, keep='tail'): drops the HEAD
        (FIFO within the slot — the oldest text goes first)
      * a bad slot index returns an error string and NEVER raises
      * result strings match the validator's ('ok: slot N rewritten, K tokens
        (truncated to budget)' / 'ok: slot N appended, K tokens (oldest dropped)')

    Counting uses the validator's dependency-free DEFAULT counter,
    WhitespaceTokenCounter (memory.py): one token per whitespace word.
    KNOWN DIVERGENCE: production injects a tokenizer-backed counter
    (core/pvp/tokenizer_counter.py load_token_counter, wired in
    core/pvp/baseline.py), so real budgets are model tokens. A word is usually
    >= 1 token, so whitespace counting truncates LESS than the tournament does
    — local prompts are an upper bound on tournament prompt size, never lower.
    """
    _SEP = '\n'

    def __init__(self, n_slots: int, slot_token_budget: int):
        if n_slots < 1:
            raise ValueError('n_slots must be >= 1')
        if slot_token_budget < 1:
            raise ValueError('slot_token_budget must be >= 1')
        self.n_slots = n_slots
        self.slot_token_budget = slot_token_budget
        self.slots = {i: '' for i in range(1, n_slots + 1)}

    @staticmethod
    def _count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _truncate(text: str, max_tokens: int, keep: str) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        kept = words[:max_tokens] if keep == 'head' else words[len(words) - max_tokens:]
        return ' '.join(kept)

    def _valid(self, slot) -> bool:
        return isinstance(slot, int) and (not isinstance(slot, bool)) and 1 <= slot <= self.n_slots

    def _fit(self, text: str, keep: str) -> str:
        if self._count(text) <= self.slot_token_budget:
            return text
        return self._truncate(text, self.slot_token_budget, keep)

    def rewrite(self, slot: int, content: str) -> str:
        if not self._valid(slot):
            return f'error: slot {slot} out of range (1-{self.n_slots})'
        stored = self._fit(content, keep='head')
        self.slots[slot] = stored
        note = ' (truncated to budget)' if stored != content else ''
        return f'ok: slot {slot} rewritten, {self._count(stored)} tokens{note}'

    def append(self, slot: int, content: str) -> str:
        if not self._valid(slot):
            return f'error: slot {slot} out of range (1-{self.n_slots})'
        existing = self.slots[slot]
        combined = f'{existing}{self._SEP}{content}' if existing else content
        stored = self._fit(combined, keep='tail')
        self.slots[slot] = stored
        note = ' (oldest dropped)' if stored != combined else ''
        return f'ok: slot {slot} appended, {self._count(stored)} tokens{note}'

    def render(self) -> dict:
        return {i: v for (i, v) in self.slots.items() if v}

    def edit(self, op: str, slot: int, content: str) -> str:
        if op not in ('rewrite', 'append'):
            return f'error: unknown memory op {op}'
        return getattr(self, op)(slot, content)

    def reset(self) -> None:
        self.slots = {i: '' for i in range(1, self.n_slots + 1)}

def _coerce_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return {}
    return v if isinstance(v, dict) else {}

def _dispatch_memory(working: _Mem, long_term: _Mem, name: str, args: dict) -> str:
    (area, _, op) = name.rpartition('_')
    target = working if area == 'working_memory' else long_term if area == 'long_term_memory' else None
    if target is None or op not in ('rewrite', 'append'):
        return f'error: unknown tool {name}'
    raw_slot = args.get('slot')
    # bot/tools.execute_memory_tool: bool is not an acceptable slot, only int|str.
    if isinstance(raw_slot, bool) or not isinstance(raw_slot, (int, str)):
        return f"error: tool {name} requires an integer 'slot'"
    try:
        slot = int(raw_slot)
    except (TypeError, ValueError):
        return f"error: tool {name} requires an integer 'slot'"
    return target.edit(op, slot, str(args.get('content', '')))

def _validate_action(args: dict, legal_set: set) -> int | None:
    """Byte-for-byte mirror of bot.LLMBot._validate_action."""
    raw = args.get('action_id')
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return None
    try:
        action = int(raw)
    except ValueError:
        return None
    return action if action in legal_set else None

class _wall_clock:
    """SIGALRM turn budget (mirrors bot._wall_clock). No-op where unsupported."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._prev = None
        self._armed = hasattr(signal, 'SIGALRM') and seconds > 0

    def __enter__(self):
        if not self._armed:
            return self

        def _handler(signum, frame):
            raise TurnTimeoutError(f'turn exceeded {self.seconds}s')
        try:
            self._prev = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.seconds)
        except ValueError:
            self._armed = False  # not the main thread
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._armed:
            signal.alarm(0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)
        return False

def make_openai_chat_fn(base_url: str, model: str, temperature: float, api_key: str='dummy', seed: int | None=PVP_DEFAULT_SEED, max_tokens: int=PVP_TURN_MAX_TOKENS, read_timeout: float=PVP_HTTP_READ_TIMEOUT_SECONDS, max_retries: int=PVP_HTTP_MAX_RETRIES):
    """OpenAI-compatible chat fn wired like core/pvp/chat.py.

    Client-level retries are disabled and re-implemented here so a slow model
    (timeout) can be classified as a forfeit while an unreachable server stays
    an infra failure, exactly as the tournament does.
    """
    import time
    import openai
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=read_timeout, max_retries=0)

    def _fn(messages, tools, call_max_tokens: int | None=None):
        # bot.LLMBot keeps two configs off ONE inbound config: the turn config
        # (max_tokens=PVP_TURN_MAX_TOKENS) and the reflection config
        # (max_tokens=PVP_REFLECTION_MAX_TOKENS). Same client, same seed,
        # same temperature — only the cap differs.
        kwargs = {'model': model, 'messages': messages, 'temperature': temperature, 'seed': seed, 'max_tokens': call_max_tokens or max_tokens}
        if tools:
            kwargs['tools'] = tools
            kwargs['tool_choice'] = 'auto'
        attempts = max_retries + 1
        transient = []
        for attempt in range(attempts):
            try:
                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message if resp.choices else None
                calls = [{'id': t.id, 'type': 'function', 'function': {'name': t.function.name, 'arguments': t.function.arguments}} for t in (getattr(msg, 'tool_calls', None) or [])]
                return {'content': getattr(msg, 'content', None), 'tool_calls': calls}
            except openai.BadRequestError as exc:
                if 'context length' in str(exc).lower():
                    raise ContextOverflowError(str(exc)) from exc
                raise
            except (TimeoutError, openai.APITimeoutError, openai.APIConnectionError) as exc:
                transient.append(exc)
                if attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, PVP_RETRY_BACKOFF_CAP_SECONDS))
            except openai.APIStatusError as exc:
                if exc.status_code >= 500 and attempt < attempts - 1:
                    transient.append(exc)
                    time.sleep(min(2 ** attempt, PVP_RETRY_BACKOFF_CAP_SECONDS))
                    continue
                raise
        timed_out = any(isinstance(e, (TimeoutError, openai.APITimeoutError)) for e in transient)
        detail = ', '.join(type(e).__name__ for e in transient) or 'unknown'
        if timed_out:
            raise ChatTimeoutForfeitError(f'chat timed out after {attempts} attempts ({detail})')
        raise ModelUnreachableError(f'inference server unreachable after {attempts} attempts ({detail})')
    return _fn

def make_stub_chat_fn():
    import re

    def _fn(messages, tools, call_max_tokens: int | None=None):
        desc = next((t['function']['description'] for t in tools if t['function']['name'] == 'game_action'), None)
        if desc is None:
            # Reflection call: memory tools only. Emit a long_term append that is
            # deliberately long enough to exercise the slot token budget/FIFO.
            note = 'stub reflection note ' + ' '.join((str(i) for i in range(60)))
            return {'content': None, 'tool_calls': [{'id': 'call_r', 'type': 'function', 'function': {'name': 'long_term_memory_append', 'arguments': json.dumps({'slot': 1, 'content': note})}}]}
        ids = [int(x) for x in re.findall('\\d+', desc.split('Legal action ids:')[-1])]
        return {'content': None, 'tool_calls': [{'id': 'call_0', 'type': 'function', 'function': {'name': 'game_action', 'arguments': json.dumps({'action_id': ids[0]})}}]}
    return _fn

def _build_turn_messages(game_name, state, player_id, legal, working, long_term):
    state_desc = gen._format_state(game_name, state, player_id)
    lines = gen._action_lines(state, player_id, legal)
    return [{'role': 'system', 'content': tc.build_system_prompt(game_name, working.render(), long_term.render())}, {'role': 'user', 'content': tc.build_user_prompt(state_desc, player_id, lines)}]

def run_turn_strict(game_name, state, player_id, chat_fn, working, long_term, stats: TurnStats | None=None):
    """EXACTLY one model call. Returns (action, forfeit_reason).

    Mirrors bot.LLMBot._run_turn: apply every memory edit, capture the first
    LEGAL game_action, otherwise forfeit. No nudge, no retry, no history.
    """
    legal = state.legal_actions(player_id)
    if not legal:
        return (None, None)
    legal_set = set(legal)
    messages = _build_turn_messages(game_name, state, player_id, legal, working, long_term)
    tools = tc.build_tools(legal)
    if stats is not None:
        stats.record_turn()
    try:
        with _wall_clock(PVP_TURN_TIMEOUT_SECONDS):
            result = chat_fn(messages, tools)
    except TurnTimeoutError:
        return (None, FORFEIT_TURN_TIMEOUT)
    except ChatTimeoutForfeitError:
        return (None, FORFEIT_CHAT_TIMEOUT)
    except ContextOverflowError:
        return (None, FORFEIT_CONTEXT_OVERFLOW)
    calls = result.get('tool_calls') or []
    if not calls:
        return (None, FORFEIT_NO_TOOL_CALL)
    action = None
    saw_game_action = False
    for call in calls:
        name = call['function']['name']
        args = _coerce_args(call['function']['arguments'])
        if name == 'game_action':
            saw_game_action = True
            if action is None:
                action = _validate_action(args, legal_set)
        else:
            _dispatch_memory(working, long_term, name, args)
    if action is not None:
        return (action, None)
    return (None, FORFEIT_ILLEGAL_ACTION if saw_game_action else FORFEIT_NO_GAME_ACTION)

def run_turn_lenient(game_name, state, player_id, chat_fn, working, long_term, stats: TurnStats | None=None):
    """LEGACY multi-step nudge/retry loop — DEBUG ONLY, not tournament-faithful."""
    legal = state.legal_actions(player_id)
    if not legal:
        return (None, None)
    legal_set = set(legal)
    messages = _build_turn_messages(game_name, state, player_id, legal, working, long_term)
    tools = tc.build_tools(legal)
    if stats is not None:
        stats.record_turn()
    saw_game_action = False
    for _ in range(_MAX_INNER_STEPS):
        result = chat_fn(messages, tools)
        calls = result.get('tool_calls') or []
        if not calls:
            messages.append({'role': 'assistant', 'content': result.get('content') or ''})
            messages.append({'role': 'user', 'content': _NUDGE})
            continue
        messages.append({'role': 'assistant', 'content': result.get('content'), 'tool_calls': calls})
        for call in calls:
            name = call['function']['name']
            args = _coerce_args(call['function']['arguments'])
            if name == 'game_action':
                saw_game_action = True
                action = _validate_action(args, legal_set)
                if action is not None:
                    return (action, None)
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''), 'content': f'error: not a legal action; legal ids = {legal}'})
            else:
                out = _dispatch_memory(working, long_term, name, args)
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''), 'content': out})
    return (None, FORFEIT_ILLEGAL_ACTION if saw_game_action else FORFEIT_NO_TOOL_CALL)

def run_turn(game_name, state, player_id, chat_fn, working, long_term, stats: TurnStats | None=None, strict: bool=True):
    runner = run_turn_strict if strict else run_turn_lenient
    return runner(game_name, state, player_id, chat_fn, working, long_term, stats)

def run_reflection(game_name, state, player_id, outcome, chat_fn, working, long_term):
    """Post-game memory consolidation — mirror of bot.LLMBot.reflect.

    core/pvp/bot.py reflect(): a SINGLE extra model call under its own
    PVP_REFLECTION_TIMEOUT_SECONDS wall clock, prompted with the reflection
    system/user prompts, offered the MEMORY TOOLS ONLY (no game_action), capped
    at PVP_REFLECTION_MAX_TOKENS, every resulting non-game_action tool call
    applied to memory, and EVERY exception swallowed ('the game is already
    decided, so a flaky or slow reflection must never affect the match').

    This is not a scored turn: it is deliberately absent from TurnStats, so the
    forfeit numbers keep counting game turns only.
    """
    try:
        with _wall_clock(PVP_REFLECTION_TIMEOUT_SECONDS):
            state_desc = gen._format_state(game_name, state, player_id)
            messages = [{'role': 'system', 'content': tc.build_reflection_system_prompt(game_name, working.render(), long_term.render())}, {'role': 'user', 'content': tc.build_reflection_user_prompt(outcome, state_desc)}]
            result = chat_fn(messages, tc._memory_tools(), PVP_REFLECTION_MAX_TOKENS)
            for call in result.get('tool_calls') or []:
                name = call['function']['name']
                if name == 'game_action':
                    continue
                _dispatch_memory(working, long_term, name, _coerce_args(call['function']['arguments']))
    except Exception as exc:
        # Mirrors bot.reflect's blanket except: even ModelUnreachableError is
        # swallowed here (it only aborts the run when raised on a SCORED turn).
        print(f'[eval] reflection failed for player {player_id} (ignored): {type(exc).__name__}: {exc}', flush=True)

def play_game(game_name, game, model_seat, chat_fn, mcts_bot, working, long_term, seed=0, stats: TurnStats | None=None, strict: bool=True, reflect: bool=True, opp: dict | None=None):
    """opp=None keeps the original model-vs-MCTS behaviour byte-for-byte.

    Pass opp={'chat_fn','working','long_term','stats'} to drive the OTHER seat with a
    second served model instead of mcts_bot — the arm-vs-arm mode the tournament actually
    scores. Both sides then run the same strict turn loop, so a forfeit by either side is
    decided by the same rules. mcts_bot is simply unused in that mode; it stays in the
    signature so every existing caller keeps working unchanged.
    """
    rng = random.Random(seed)
    state = game.new_initial_state()
    if game_name in ('othello', 'clobber'):
        for _ in range(rng.randint(2, 6)):
            if state.is_terminal():
                break
            lo = state.legal_actions()
            if not lo:
                break
            state.apply_action(rng.choice(lo))
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            (actions, probs) = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cur = state.current_player()
        if cur == model_seat:
            (action, reason) = run_turn(game_name, state, model_seat, chat_fn, working, long_term, stats, strict)
            if action is None:
                if reason is None:
                    # NOT a forfeit: reason is None only on the empty-legal-actions path
                    # (run_turn_strict `if not legal`). Validator maps that to
                    # EmptyLegalActionsError -> game_eval [0.0]*n -> DRAW, with no forfeiter,
                    # so BOTH bots still reflect. Booking it as a loss (and, on the other
                    # seat, a win) would turn 0.5/0.5 into 1/0 whose sign depends on the
                    # seat — and it does NOT cancel across a swapped pair, because an
                    # empty-legal state is a property of the position, not of who sits where.
                    if reflect:
                        run_reflection(game_name, state, model_seat, 'draw', chat_fn, working, long_term)
                        if opp is not None:
                            run_reflection(game_name, state, 1 - model_seat, 'draw', opp['chat_fn'], opp['working'], opp['long_term'])
                    return ('draw', None)
                if stats is not None:
                    stats.record_forfeit(reason)
                # validator game_runner: only the FORFEITER skips reflect(); the survivor
                # still consolidates memory. Skipping BOTH would penalise whichever arm
                # forfeits less — backwards. (opp is None in MCTS mode: nothing to do,
                # so the default path is untouched.)
                if reflect and opp is not None:
                    run_reflection(game_name, state, 1 - model_seat, 'win', opp['chat_fn'], opp['working'], opp['long_term'])
                return ('loss', reason)
            state.apply_action(action)
        elif opp is None:
            state.apply_action(mcts_bot.step(state))
        else:
            # Arm-vs-arm: the other seat is a second model, played through the SAME strict
            # turn loop, so both arms face identical forfeit rules. `cur` and 1 - model_seat
            # are provably equal here (2-player, non-terminal, non-chance, cur != model_seat);
            # cur is used simply because it is what the game just told us.
            (oaction, oreason) = run_turn(game_name, state, cur, opp['chat_fn'], opp['working'], opp['long_term'], opp.get('stats'), strict)
            if oaction is None:
                if oreason is None:
                    # Same empty-legal-actions case as above, seen from the other seat —
                    # a DRAW, not a free win for us.
                    if reflect:
                        run_reflection(game_name, state, model_seat, 'draw', chat_fn, working, long_term)
                        run_reflection(game_name, state, 1 - model_seat, 'draw', opp['chat_fn'], opp['working'], opp['long_term'])
                    return ('draw', None)
                if opp.get('stats') is not None:
                    opp['stats'].record_forfeit(oreason)
                # Their forfeit is our win, and WE are the survivor — so we still reflect.
                if reflect:
                    run_reflection(game_name, state, model_seat, 'win', chat_fn, working, long_term)
                return ('win', None)
            state.apply_action(oaction)
    r = state.returns()[model_seat]
    outcome = 'win' if r > 0 else 'loss' if r < 0 else 'draw'
    # game_runner._play_game / baseline.run_mcts_baseline: reflect after every
    # game EXCEPT for a bot that forfeited (the forfeit paths return above).
    if reflect:
        run_reflection(game_name, state, model_seat, outcome, chat_fn, working, long_term)
        if opp is not None:
            # Both arms must reflect, or the one that does gets a memory advantage the
            # tournament would have given both. 1 - model_seat is safe: every game in
            # --game is 2-player.
            opp_outcome = 'loss' if outcome == 'win' else ('win' if outcome == 'loss' else 'draw')
            run_reflection(game_name, state, 1 - model_seat, opp_outcome, opp['chat_fn'], opp['working'], opp['long_term'])
    return (outcome, None)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', required=True, choices=['liars_dice', 'leduc_poker', 'gin_rummy', 'othello', 'clobber', 'goofspiel'])
    p.add_argument('--num-games', type=int, default=100)
    p.add_argument('--matchup-games', type=int, default=4, help='Games per opponent before long-term memory resets (memory persists within).')
    p.add_argument('--mcts-sims', type=int, default=0, help='Opponent MCTS sims (0 = eval strength).')
    p.add_argument('--model-url', default='http://127.0.0.1:30000/v1', help='OpenAI-compatible base url (default: localhost:30000).')
    p.add_argument('--model-name', default=None, help='Served model name; auto-detected from /v1/models if omitted.')
    p.add_argument('--temperature', type=float, default=PVP_DEFAULT_TEMPERATURE)
    p.add_argument('--stub', action='store_true', help='No model — emits game_action(first legal id); mechanics test only.')
    p.add_argument('--seed', type=int, default=PVP_DEFAULT_SEED, help='Base seed: game generation AND the chat completion seed (validator uses one value for both).')
    p.add_argument('--lenient', action='store_true', help='DEBUG ONLY: restore the legacy nudge/retry inner loop. Overstates win-rate vs the tournament, which allows exactly one call per turn.')
    p.add_argument('--max-tokens', type=int, default=PVP_TURN_MAX_TOKENS, help='Per-turn generation cap (tournament: PVP_TURN_MAX_TOKENS).')
    p.add_argument('--no-reflect', action='store_true', help='DEBUG ONLY: skip the post-game reflect() call. The tournament always reflects after a non-forfeited game, so this understates long-term-memory usage.')
    p.add_argument('--read-timeout', type=float, default=PVP_HTTP_READ_TIMEOUT_SECONDS, help='HTTP read timeout (tournament: PVP_HTTP_READ_TIMEOUT_SECONDS).')
    p.add_argument('--opponent-url', default=None, help='ARM-VS-ARM: base url of a SECOND served model that plays the other seat INSTEAD of MCTS. This is the only mode that can detect counter-exploitation — see the module docstring. Omit for the legacy model-vs-MCTS behaviour.')
    p.add_argument('--opponent-name', default=None, help='Served model name for --opponent-url; auto-detected from /v1/models if omitted.')
    args = p.parse_args()
    import pyspiel
    strict = not args.lenient
    if not hasattr(signal, 'SIGALRM') and not args.stub:
        # _wall_clock silently degrades to a no-op without SIGALRM (Windows), which makes
        # FORFEIT_TURN_TIMEOUT structurally UNREACHABLE: a model taking five minutes a turn
        # scores here as if it were instant, while the tournament forfeits it every turn.
        # A zero in the turn_timeout column would then be an artefact, not evidence — so
        # refuse rather than hand back an optimistic number that reads as clean.
        p.error('SIGALRM is unavailable on this platform, so the per-turn wall clock never arms and turn_timeout forfeits cannot be measured — every number from this run would be optimistic. Run on Linux, or pass --stub for a mechanics-only check.')
    opp_sims = args.mcts_sims or gen._OPPONENT_SIMS[args.game]
    if args.stub:
        chat_fn = make_stub_chat_fn()
        model_name = 'stub'
    else:
        model_name = args.model_name
        if model_name is None:
            from openai import OpenAI
            try:
                model_name = OpenAI(base_url=args.model_url, api_key='dummy').models.list().data[0].id
                print(f'[eval] auto-detected served model: {model_name}', flush=True)
            except Exception as exc:
                p.error(f'could not auto-detect model from {args.model_url} ({exc}); pass --model-name')
        chat_fn = make_openai_chat_fn(args.model_url, model_name, args.temperature, seed=args.seed, max_tokens=args.max_tokens, read_timeout=args.read_timeout)
    opp_chat_fn = None
    opp_stats = None
    opp_model_name = None
    if args.opponent_url:
        if args.stub:
            p.error('--stub and --opponent-url are mutually exclusive (the stub is a mechanics test, not an arm).')
        opp_model_name = args.opponent_name
        if opp_model_name is None:
            from openai import OpenAI
            try:
                opp_model_name = OpenAI(base_url=args.opponent_url, api_key='dummy').models.list().data[0].id
                print(f'[eval] auto-detected opponent model: {opp_model_name}', flush=True)
            except Exception as exc:
                p.error(f'could not auto-detect opponent model from {args.opponent_url} ({exc}); pass --opponent-name')
        # Same temperature/seed/token budget as our arm: the point of arm-vs-arm is to
        # isolate the POLICY difference, so every other knob must be identical.
        opp_chat_fn = make_openai_chat_fn(args.opponent_url, opp_model_name, args.temperature, seed=args.seed, max_tokens=args.max_tokens, read_timeout=args.read_timeout)
        opp_stats = TurnStats(opp_model_name, args.game)
        if args.num_games % 2 != 0:
            p.error('--num-games must be EVEN in arm mode: each seed is played twice with the seats swapped, so an odd count leaves one board unpaired and hands one arm an extra first-seat game.')
    if args.matchup_games == p.get_default('matchup_games'):
        # BOTH modes. The validator never resets long-term memory mid-matchup: core/pvp/
        # baseline.py builds it ONCE before its game loop, and game_runner builds one per
        # matchup. The old default of 4 wiped it ~25x more often than the tournament does,
        # which suppresses exactly the accumulated opponent-reading that counter-exploitation
        # is made of. Explicit --matchup-games is still honoured.
        args.matchup_games = args.num_games
        print(f'[eval] --matchup-games defaulted to {args.num_games} (validator keeps long-term memory for a WHOLE matchup). ⚠ the old default was 4, so numbers recorded before 2026-08-05 are not directly comparable to this run.', flush=True)
    mode = 'STRICT (1 model call/turn, no nudge — tournament-faithful)' if strict else 'LENIENT (legacy nudge/retry — DEBUG ONLY, overstates win-rate)'
    print(f'[eval] mode: {mode}', flush=True)
    print(f'[eval] turn budget {PVP_TURN_TIMEOUT_SECONDS}s | max_tokens {args.max_tokens} | temperature {args.temperature} | seed {args.seed} | read_timeout {args.read_timeout}s', flush=True)
    _reflect_on = not args.no_reflect
    _reflect_desc = f'ON (max_tokens {PVP_REFLECTION_MAX_TOKENS}, {PVP_REFLECTION_TIMEOUT_SECONDS}s clock, errors ignored)' if _reflect_on else 'OFF (--no-reflect, DEBUG)'
    print(f'[eval] memory: working {tc.PVP_WORKING_MEM_SLOTS}x{tc.PVP_WORKING_SLOT_TOKENS} tok, long-term {tc.PVP_LONGTERM_MEM_SLOTS}x{tc.PVP_LONGTERM_SLOT_TOKENS} tok (whitespace counter) | reflect: {_reflect_desc}', flush=True)
    stats = TurnStats(model_name, args.game)
    seed_rng = random.Random(args.seed)
    wins = draws = losses = 0
    long_term = _Mem(tc.PVP_LONGTERM_MEM_SLOTS, tc.PVP_LONGTERM_SLOT_TOKENS)
    opp_long_term = _Mem(tc.PVP_LONGTERM_MEM_SLOTS, tc.PVP_LONGTERM_SLOT_TOKENS)
    # validator game_runner.warmup_player: one throwaway completion per player, so SGLang's
    # CUDA-graph capture does not land inside the first SCORED turn's 30s clock. Without it
    # that cost is charged to whoever moves first in game 0 — systematically OUR arm, since
    # model_seat = i % 2 starts at 0. Swallow everything, exactly as the validator does.
    for (_wfn, _wlabel) in ((chat_fn, 'model'), (opp_chat_fn, 'opponent')):
        if _wfn is None:
            continue
        try:
            _wfn([{'role': 'user', 'content': 'warmup'}], tc._memory_tools(), 16)
            print(f'[eval] warmup ok: {_wlabel}', flush=True)
        except Exception as exc:
            print(f'[eval] warmup failed for {_wlabel} (ignored): {type(exc).__name__}: {exc}', flush=True)
    for i in range(args.num_games):
        if i % args.matchup_games == 0:
            long_term = _Mem(tc.PVP_LONGTERM_MEM_SLOTS, tc.PVP_LONGTERM_SLOT_TOKENS)
            # Reset BOTH arms on the same boundary — a stale opponent memory across a
            # matchup reset would hand one side history the other no longer has.
            opp_long_term = _Mem(tc.PVP_LONGTERM_MEM_SLOTS, tc.PVP_LONGTERM_SLOT_TOKENS)
        working = _Mem(tc.PVP_WORKING_MEM_SLOTS, tc.PVP_WORKING_SLOT_TOKENS)
        # Arm mode PAIRS boards: one seed per pair, replayed with model_seat flipped, which
        # is what validator game_runner does ("each seed is played twice with swapped
        # positions"). That cancels board luck within the pair instead of only in the limit.
        # MCTS mode still draws per game, so its historical numbers stay reproducible.
        # i == 0 is even, so base_seed is always bound before first use.
        if opp_chat_fn is None or i % 2 == 0:
            base_seed = seed_rng.randint(1, 2000000000)
        (name, params) = gen._game_params(args.game, base_seed % 1000)
        game = gen._load_pyspiel_game(args.game, name, params)
        model_seat = i % 2
        mcts_bot = gen._make_mcts_bot(game, opp_sims, base_seed + (1 - model_seat))
        opp_bundle = None
        if opp_chat_fn is not None:
            opp_bundle = {'chat_fn': opp_chat_fn, 'working': _Mem(tc.PVP_WORKING_MEM_SLOTS, tc.PVP_WORKING_SLOT_TOKENS), 'long_term': opp_long_term, 'stats': opp_stats}
        (outcome, forfeit_reason) = play_game(args.game, game, model_seat, chat_fn, mcts_bot, working, long_term, base_seed, stats, strict, _reflect_on, opp_bundle)
        stats.games += 1
        if opp_stats is not None:
            opp_stats.games += 1
        if forfeit_reason is not None:
            stats.games_lost_to_forfeit += 1
        wins += outcome == 'win'
        draws += outcome == 'draw'
        losses += outcome == 'loss'
        if (i + 1) % 20 == 0:
            print(f'[eval] {i + 1}/{args.num_games} | W-D-L {wins}-{draws}-{losses} | win-rate {wins / (i + 1):.3f} score {(wins + 0.5 * draws) / (i + 1):.3f} | forfeit-turns {stats.forfeits}/{stats.turns} ({stats.forfeit_rate:.4f})', flush=True)
    n = args.num_games
    _opp_desc = f'MCTS-{opp_sims}' if opp_chat_fn is None else f'ARM {opp_model_name}'
    print(f'\n=== {args.game} vs {_opp_desc} ({n} games) | model={model_name} | mode={"strict" if strict else "lenient"} ===')
    print(f'W-D-L: {wins}-{draws}-{losses}')
    print(f'win-rate:   {wins / n:.3f}')
    print(f'mean score: {(wins + 0.5 * draws) / n:.3f}   (win=1, draw=0.5, loss=0 — the baseline metric)')
    print(f'forfeit-rate: {stats.forfeits}/{stats.turns} turns = {stats.forfeit_rate:.4f}')
    print(f'games lost to forfeit: {stats.games_lost_to_forfeit}/{n} = {stats.games_lost_to_forfeit / n if n else 0.0:.4f}')
    breakdown = ', '.join(f'{k}={v}' for (k, v) in stats.reasons.items() if v) or 'none'
    print(f'forfeit reasons: {breakdown}')
    print('[eval-json] ' + json.dumps(stats.as_dict(), sort_keys=True))
    if opp_stats is not None:
        # Every opponent forfeit ends its game immediately (record_forfeit is followed by a
        # return), so for THIS arm forfeit-turns and games-lost-to-forfeit coincide. Set it
        # explicitly rather than emit a contradictory games_lost_to_forfeit=0 in the JSON.
        opp_stats.games_lost_to_forfeit = opp_stats.forfeits
        # Read this BEFORE celebrating a win-rate. A win driven by the opponent arm's
        # forfeits is a mechanics failure on their side, not a policy edge on ours — and
        # the tournament forfeit rules are harsh enough that this is a live possibility.
        print(f'opponent forfeit-rate: {opp_stats.forfeits}/{opp_stats.turns} turns = {opp_stats.forfeit_rate:.4f}')
        print(f'our wins from THEIR forfeit: {opp_stats.forfeits}/{n} = {opp_stats.forfeits / n if n else 0.0:.4f} (discount these before reading a policy edge)')
        print('[eval-json-opponent] ' + json.dumps(opp_stats.as_dict(), sort_keys=True))
if __name__ == '__main__':
    main()
