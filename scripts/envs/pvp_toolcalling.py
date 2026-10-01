from __future__ import annotations
import json
from pathlib import Path
PVP_WORKING_MEM_SLOTS = 4
PVP_WORKING_SLOT_TOKENS = 128
PVP_LONGTERM_MEM_SLOTS = 8
PVP_LONGTERM_SLOT_TOKENS = 128
PVP_PER_STEP_MAX_TOKENS = 256
PVP_MAX_INNER_STEPS = 4
_TOOL_GUIDANCE = 'You get ONE response this turn. In it, optionally edit your memory notes, and then call game_action with a legal action id to commit your move. If you do not call game_action, you forfeit the turn — so always include it.'
_REFLECTION_GUIDANCE = 'The game is over. Use the memory tools to update your long-term notes on this opponent for future games — keep durable, generalisable reads (their tendencies, your counter-strategy) and drop move-by-move detail. There is no move to make.'
_PROMPTS_PATH = Path(__file__).resolve().parent / 'pvp_prompts' / 'pvp_game_prompts_toolcalling.yml'
_RULES_KEY = {'liars_dice': 'liars_dice_rules', 'leduc_poker': 'leduc_poker_rules', 'gin_rummy': 'gin_rummy_rules', 'othello': 'othello_rules', 'goofspiel': 'goofspiel_rules', 'clobber': 'clobber_rules'}

def _load_prompts() -> dict:
    import yaml
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)

def system_rules_prompt(game_name: str) -> str:
    p = _load_prompts()
    return p['system_prompt_template'].format(game_name=game_name, rules=p[_RULES_KEY[game_name]])

def _render_area(title: str, n_slots: int, slots: dict | None) -> str:
    slots = slots or {}
    lines = [f"  [{i}] {slots.get(i) or '(empty)'}" for i in range(1, n_slots + 1)]
    return f'{title}\n' + '\n'.join(lines)

def memory_block(working: dict | None=None, long_term: dict | None=None) -> str:
    return '\n\n'.join([_render_area('WORKING_MEMORY (your notes):', PVP_WORKING_MEM_SLOTS, working), _render_area('LONG_TERM_MEMORY (your notes):', PVP_LONGTERM_MEM_SLOTS, long_term)])

def build_system_prompt(game_name: str, working: dict | None=None, long_term: dict | None=None) -> str:
    return '\n\n'.join([system_rules_prompt(game_name), memory_block(working, long_term), _TOOL_GUIDANCE])

def build_user_prompt(state_desc: str, player_id: int, action_lines: list[str]) -> str:
    return f'Current state:\n{state_desc}\n\nYou are Player {player_id}.\nLegal actions:\n' + '\n'.join(action_lines)

# Key order below (properties, required, type at the parameters level; description,
# type, then minimum/maximum/enum inside each property) mirrors what pydantic's
# model_json_schema() actually emits for the validator's FunctionSchema/ToolSchema
# models, once _params_schema() pops "title"/model "description" and bolts on
# minimum/maximum/enum by post-hoc dict assignment (see G.O.D core/pvp/tools.py:55-71
# and core/models/pvp_models.py's GameActionArgs/MemorySlotEdit). We reconstruct that
# order literally rather than importing validator code, so json.dumps(tool) (no
# sort_keys) and the chat-template rendering land byte-identical to the eval request.

def _game_action_tool(legal_action_ids: list[int]) -> dict:
    legal_hint = 'Legal action ids: ' + ', '.join((str(a) for a in legal_action_ids)) + '.'
    return {'type': 'function', 'function': {'name': 'game_action', 'description': f'Commit your move and end your turn. {legal_hint}', 'parameters': {'properties': {'action_id': {'description': 'A legal action id for the current state.', 'type': 'integer', 'enum': list(legal_action_ids)}}, 'required': ['action_id'], 'type': 'object'}}}
_AREAS = [('working_memory', PVP_WORKING_MEM_SLOTS, 'notes for THIS game, reset each game'), ('long_term_memory', PVP_LONGTERM_MEM_SLOTS, 'notes on THIS opponent, persist across games')]
_OPS = [('rewrite', 'Overwrite', "replaces the slot's previous content"), ('append', 'Append to', 'oldest text drops if the slot is full')]

def _memory_tools() -> list[dict]:
    out = []
    for (area, n, purpose) in _AREAS:
        for (op, verb, effect) in _OPS:
            out.append({'type': 'function', 'function': {'name': f'{area}_{op}', 'description': f'{verb} a {area} slot (slots 1-{n}; {purpose}); {effect}.', 'parameters': {'properties': {'slot': {'description': 'Target slot number.', 'type': 'integer', 'minimum': 1, 'maximum': n}, 'content': {'description': 'Text content for the slot.', 'type': 'string'}}, 'required': ['slot', 'content'], 'type': 'object'}}})
    return out

def build_tools(legal_action_ids: list[int]) -> list[dict]:
    return _memory_tools() + [_game_action_tool(legal_action_ids)]

def game_action_message(action_id: int, content: str | None=None, call_id: str='call_0') -> dict:
    return {'role': 'assistant', 'content': content, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'game_action', 'arguments': json.dumps({'action_id': int(action_id)})}}]}

def memory_tool_message(name: str, slot: int, text: str, content: str | None=None, call_id: str='call_m') -> dict:
    return {'role': 'assistant', 'content': content, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps({'slot': int(slot), 'content': text})}}]}

def tool_result_message(call_id: str, content: str) -> dict:
    return {'role': 'tool', 'tool_call_id': call_id, 'content': content}

def build_toolcalling_example(game_name: str, state_desc: str, player_id: int, action_lines: list[str], legal_action_ids: list[int], action_id: int, *, working: dict | None=None, long_term: dict | None=None, content: str | None=None) -> dict:
    return {'messages': [{'role': 'system', 'content': build_system_prompt(game_name, working, long_term)}, {'role': 'user', 'content': build_user_prompt(state_desc, player_id, action_lines)}, game_action_message(action_id, content=content)], 'tools': build_tools(legal_action_ids)}

def build_reflection_system_prompt(game_name: str, working: dict | None=None, long_term: dict | None=None) -> str:
    return '\n\n'.join([system_rules_prompt(game_name), memory_block(working, long_term), _REFLECTION_GUIDANCE])

def build_reflection_user_prompt(outcome: str, state_desc: str) -> str:
    return f'The game is over. Result for you: {outcome.upper()}.\n\nFinal state:\n{state_desc}\n\nUpdate your long-term notes on this opponent for future games.'

def build_reflection_example(game_name: str, outcome: str, state_desc: str, slot: int, note: str, *, op: str='append', working: dict | None=None, long_term: dict | None=None, content: str | None=None) -> dict:
    name = f'long_term_memory_{op}'
    return {'messages': [{'role': 'system', 'content': build_reflection_system_prompt(game_name, working, long_term)}, {'role': 'user', 'content': build_reflection_user_prompt(outcome, state_desc)}, memory_tool_message(name, slot, note, content=content)], 'tools': _memory_tools()}
