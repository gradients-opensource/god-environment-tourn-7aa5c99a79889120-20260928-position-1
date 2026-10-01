import os
from copy import deepcopy
from shared.model_utility import disable_flash_attention, get_gpu_count, get_model_architecture, get_model_num_params, get_use_liger

_SFT_TRAJECTORY_ENVS = {'liars_dice', 'gin_rummy', 'leduc_poker'}
_INDIVIDUAL_ENVS = {'intercode', 'swe_infinite'}  # swe_infinite: INDIVIDUAL eval, SFT-distilled from SWE-ZERO via envs.mastering_swe (see pvp_infra.sft_env_config._build_generate_cmd). MUST stay here or text_trainer drops it → SWE task trains on nothing → draw → the incumbent baseline retains the lead.
_TOOLCALLING_ENVS = {'othello', 'goofspiel', 'clobber'}


def supports_sft_trajectory(env_name: str) -> bool:
    return env_name in _SFT_TRAJECTORY_ENVS or env_name in _INDIVIDUAL_ENVS or env_name in _TOOLCALLING_ENVS

SFT_ENV_SIZE_CONFIG: dict[str, dict] = {'0_1_b': {'lr': 0.0001, 'distributed': 'ddp', 'gpu_count': 1, 'batch_size': 8, 'use_lora': False, 'max_seq_length': 2048}, '1_2_b': {'lr': 0.0001, 'distributed': 'ddp', 'gpu_count': 4, 'batch_size': 8, 'use_lora': False, 'max_seq_length': 4096}, '2_4_b': {'lr': 6e-05, 'distributed': 'ddp', 'gpu_count': 4, 'batch_size': 8, 'use_lora': True, 'max_seq_length': 4096}, '4_5_b': {'lr': 3.571e-05, 'distributed': 'ddp', 'gpu_count': 4, 'batch_size': 8, 'use_lora': True, 'max_seq_length': 4096}, '5_9_b': {'lr': 4.302e-05, 'distributed': 'ddp', 'gpu_count': 4, 'batch_size': 6, 'use_lora': True, 'max_seq_length': 4096}, '9_12_b': {'lr': 0.0001, 'distributed': 'ddp', 'gpu_count': 2, 'batch_size': 4, 'use_lora': True, 'max_seq_length': 2048}, '12_15_b': {'lr': 0.0001, 'distributed': 'ds', 'gpu_count': 4, 'batch_size': 4, 'use_lora': True, 'max_seq_length': 2048}, '15_40_b': {'lr': 8e-05, 'distributed': 'ds', 'gpu_count': 4, 'batch_size': 2, 'use_lora': True, 'max_seq_length': 2048}, '40_80_b': {'lr': 8e-05, 'distributed': 'ds', 'gpu_count': 8, 'batch_size': 2, 'use_lora': True, 'max_seq_length': 2048}}
# T1 (2026-09-14, Audit-config-coherence.md §3/§5.4/§7-T1) — 1_2_b max_seq_length 2048 -> 4096.
# The stale version of this comment argued from a `gpu_count=1` that no longer exists (gpu_count
# here has read 4 since the 2_4_b fix landed, see git history) — that thread is closed, this is
# about the OTHER unexplained 1_2_b outlier: max_seq_length. Measured with the cached
# Qwen/Qwen2.5-1.5B-Instruct tokenizer on our own generator's output (parse-only, no dataset exec):
# liars_dice p50/p90/max 1874-1937 / 2173-2197 / 2223 tokens, 34-37% of rows over 2048 (two
# independent samples, n=32 and n=38); gin_rummy p50/p90/max 1754/1918/1997, 0/312 rows over 2048
# (a near-miss, not a violation). `train_sft_env.py:348` FRONT-truncates any row over
# max_seq_length (keeps the TAIL, `ids[-max_length:]`) so the assistant response at the end
# survives. On the over-length liars rows the discarded head is only 62-149 tokens: the
# `<|im_start|>system` role header plus the opening of the rules text. The `<tools>` block is
# rendered AFTER the system prompt body (it starts at token ~432) and survived in 11/11 measured
# truncated rows — the tool-schema-loss regime only begins above ~2480 tokens, which no sample
# reached. The damage is therefore a prompt shape that never occurs at inference (no system role
# marker, truncated rules preamble) on a third of liars rows. 4096 matches every other whitelist bucket
# (2_4_b/4_5_b/5_9_b already use 4096) and covers the measured liars max (2223) with headroom;
# 0_1_b is UNCHANGED (no whitelist model maps there, and no measurement was taken for it here).
for _key in SFT_ENV_SIZE_CONFIG:
    SFT_ENV_SIZE_CONFIG[_key]['label'] = _key

# E9 (2026-09-14, Loki R1 ce540755 vs the fork's 79368606/f495a3c2): our R1 run on Llama-3.1-8B did
# ~4.2 rows/s (per_device 6, ga 5, 2 GPU, eff-batch 60) and never finished the anneal inside 1.5h
# (LR stalled at 42% of peak); the fork on the SAME base model used per_device 16 (5EqnMm) / 32
# (5CRjuf) and reached 9.8-11.1 rows/s. Originally gated to `architecture == 'llamaforcausallm'`
# because that was the only cell with fork evidence. T3 (2026-09-14, audit
# tourn_4f605297e6d85428_20260907 §5.6(f)/§7 T3) widens the gate to the WHOLE `5_9_b` bucket —
# Qwen2.5-7B shares this bucket and was left on the pre-E9 config (batch 6, eff 48/60), which the
# audit's own arithmetic (§4) predicts stalls the anneal at 1.29-1.45 epochs / 38-46% of peak LR —
# the exact failure mode E9 fixed for Llama. Memory case for including Qwen2.5-7B (§5.4, §5.6(f)):
# batch-16/seq-4096 activations for Qwen2.5-7B are STRONGLY-INFERRED lighter than the Llama-3.1-8B
# batch-16 cells already measured safe (sel B/C/D/F/G) — 28 layers x 3584 hidden vs Llama's 32 x
# 4096 gives ~0.765x the activation footprint at equal batch/seq; weights are smaller (15.2GB vs
# 16.1GB, both well under 80GB with LoRA); GQA halves KV-cache heads (4 vs 8); liger is active for
# `qwen2forcausallm` too (`shared/model_utility.py:25`), so logits are never materialized for
# either architecture. It is lighter in AGGREGATE (~0.77x layer-boundary activations, smaller
# weights, half the KV heads) DESPITE a 1.32x wider MLP intermediate (18944 vs 14336) whose
# per-layer recompute transient is larger; the totals land ~36GB (Qwen) vs ~39GB (Llama) at
# batch 16 / seq 4096 with checkpointing + liger. No single factor is measured directly on
# Qwen2.5-7B at batch 16 — this is arithmetic against an already-measured-safe cell, not a
# measurement of Qwen2.5-7B itself (see SMOKE-NEEDED in the README). The constant is renamed so it
# no longer claims an architecture it no longer gates on. 16 (not the fork's 32) remains the
# conservative precedent. gpu_nums==1 is left alone — there is no fork evidence for a single-GPU
# cell in this bucket for either architecture.
MICROBATCH_5_9B = 16

def _is_5_9b_bucket_arm(config: dict) -> bool:
    return config.get('label') == '5_9_b'

def get_adaptive_epochs(hours_to_complete: float) -> int:
    if hours_to_complete >= 3.5:
        return 4
    if hours_to_complete >= 2.5:
        return 3
    return 2

def _is_prior_repo(model_ref: str) -> bool:
    return bool(model_ref) and 'gradients-io-tournaments/tournament-' in str(model_ref)
REGULAR_HOURS = 1.5
LONG_TIER_HOURS = 3.0
GAME_TARGETS = {'gin_rummy': {(1.5, 1): 15000, (1.5, 2): 13500, (1.5, 3): 9000, (1.5, 4): 6800, (1.5, 5): 5400, (1.5, 6): 4500, (3.0, 1): 25000, (3.0, 2): 15500, (3.0, 3): 10500, (3.0, 4): 7800, (3.0, 5): 6200, (3.0, 6): 5200}, 'liars_dice': {(1.5, 1): 27000, (1.5, 2): 15100, (1.5, 3): 10100, (1.5, 4): 7600, (1.5, 5): 6000, (1.5, 6): 5000, (3.0, 1): 31000, (3.0, 2): 17400, (3.0, 3): 11800, (3.0, 4): 8700, (3.0, 5): 6900, (3.0, 6): 5800}, 'leduc_poker': {(1.5, 1): 27000, (1.5, 2): 11300, (1.5, 3): 7500, (1.5, 4): 5700, (1.5, 5): 4600, (1.5, 6): 3800, (3.0, 1): 31000, (3.0, 2): 12800, (3.0, 3): 8700, (3.0, 4): 6500, (3.0, 5): 5200, (3.0, 6): 4400}, 'othello': {(1.5, 1): 15000, (1.5, 2): 13500, (1.5, 3): 9000, (1.5, 4): 6800, (1.5, 5): 5400, (1.5, 6): 4500, (3.0, 1): 25000, (3.0, 2): 15500, (3.0, 3): 10500, (3.0, 4): 7800, (3.0, 5): 6200, (3.0, 6): 5200}, 'intercode': {(1.5, 1): 27000, (1.5, 2): 13500, (1.5, 3): 9000, (1.5, 4): 6800, (1.5, 5): 5400, (1.5, 6): 4500, (3.0, 1): 31000, (3.0, 2): 15500, (3.0, 3): 10500, (3.0, 4): 7800, (3.0, 5): 6200, (3.0, 6): 5200}, 'goofspiel': {(1.5, 1): 27000, (1.5, 2): 13500, (1.5, 3): 9000, (1.5, 4): 6800, (1.5, 5): 5400, (1.5, 6): 4500, (3.0, 1): 31000, (3.0, 2): 15500, (3.0, 3): 10500, (3.0, 4): 7800, (3.0, 5): 6200, (3.0, 6): 5200}, 'clobber': {(1.5, 1): 15000, (1.5, 2): 13500, (1.5, 3): 9000, (1.5, 4): 6800, (1.5, 5): 5400, (1.5, 6): 4500, (3.0, 1): 25000, (3.0, 2): 15500, (3.0, 3): 10500, (3.0, 4): 7800, (3.0, 5): 6200, (3.0, 6): 5200}}
# swe_infinite: always num_envs=1 (single-env final-round task-3), so only the (*, 1) keys are used; the
# rest mirror a single-env game for the _compute_target_samples .get() fallback. Sized so a full
# ~2-epoch pass over long multi-turn SWE trajectories fits the 1.5h/3.0h windows (mastering_swe
# caps to this; real SWE-ZERO tops out near per_instance*2947 rows, remainder self-gen/incomplete).
GAME_TARGETS['swe_infinite'] = {(1.5, 1): 12000, (1.5, 2): 10000, (1.5, 3): 9000, (1.5, 4): 6800, (1.5, 5): 5400, (1.5, 6): 4500, (3.0, 1): 18000, (3.0, 2): 12000, (3.0, 3): 10500, (3.0, 4): 7800, (3.0, 5): 6200, (3.0, 6): 5200}
WINS_ONLY_PER_ENV = {'gin_rummy': True, 'liars_dice': False, 'leduc_poker': False, 'othello': False, 'goofspiel': True, 'clobber': True}  # liars->False (2026-07-06): keep ALL turns; wins-only on a near-Nash game keeps a biased thin label set. Paired with the SHIPPED liars teacher = 'posterior' (NOT mccfr, NOT binomial — neither branch exists in the repo; see the liars note on TOURNAMENT_TEACHERS below).  # othello->False (2026-07-06): match the target distribution — the othello data-gen (generate_trajectories._ENV_DEFAULT_ARGS) sets NO score filter, keeps ALL games. A deterministic heuristic teacher's MOVES are correct even in games the random-sim MCTS opponent happened to win; wins-only threw away good move examples. Paired with the static weight-table othello teacher.
# E4/ARM-A1 (2026-09-07, Loki R2 cc09458f, Llama-3.1-8B): the winning fork (group 5EqnMm) shipped
# liars_dice with --no-reflection (generate_pvp_toolcalling.py already has this flag, wired to
# args.no_reflection at line ~500) and pure game_action rows, vs our mix where ~25% of liars rows
# are reflection turns that teach nothing about the bidding policy. Gate this per-env, same pattern
# as WINS_ONLY_PER_ENV above, so every other env's command is byte-identical. Default True
# (reflection kept) preserves every non-liars env exactly as before this dict existed. Deliberately
# does NOT touch target_samples/NASH_DATA_BOOST — that is ARM-A2 (see below), split out after guard
# review found the row-floor in combine_datasets.py inflates OTHER envs when one env's target grows
# disproportionately; A1 alone changes composition (fewer wasted reflection rows in the SAME liars
# target), not volume, so the row-floor math is untouched.
REFLECTION_PER_ENV = {'liars_dice': False}
INCLUDE_DRAWS_PER_ENV = {'gin_rummy': True, 'liars_dice': False, 'leduc_poker': True, 'othello': True, 'goofspiel': True, 'clobber': False}
WIN_RATE_ESTIMATE = {'gin_rummy': 0.38, 'liars_dice': 0.6, 'leduc_poker': 0.5, 'othello': 0.6, 'goofspiel': 0.33, 'clobber': 0.6}  # goofspiel 0.5->0.33 (2026-08-04, PAIRED with teacher valuematch->oneup — do not change one without the other). This constant provisions GAMES (games = target/win_rate/...) against the DATA-GEN opponent, which is MCTS-50 (_OPPONENT_SIMS), NOT the PvP field. one-up scores ~0.33 vs MCTS-50 where value-match scored 0.388-0.400, so we must provision MORE games to reach the same sample target: games is proportional to 1/win_rate, so 0.5->0.33 RAISES the ceiling by +51.5% (multiplier 0.550->0.833; R1 2-env example: 3062 -> 4640 games, and the extreme (1.5,1) case saturates the _MAX_NUM_GAMES=25000 cap). ⚠ an earlier version of this line read "the CEILING must come down", which is BACKWARDS arithmetic — lowering a divisor raises the quotient. Do NOT "fix" the code to match that phrasing. ⚠ CORRECTED 2026-08-04 (adversarial review): this does NOT add rows, and an earlier version of this comment wrongly claimed it did. num_games is only an UPPER BOUND — --target_samples early-stops generation (generate_pvp_toolcalling: the stop fires in both the parallel and sequential paths) and it already bound at 0.5 with ~2.2x headroom, so goofspiel's row count was never set by this constant. Rows/game = keep*num_cards + 1 (the reflection row is appended for EVERY game, outside the --wins-only filter), so num_games could only bind below keep~0.09. TO ACTUALLY ADD GOOFSPIEL ROWS change NASH_DATA_BOOST['goofspiel'] or GAME_TARGETS['goofspiel'], not this. Erring LOW here is free: over-planning just submits futures that get cancelled. Draws are kept (INCLUDE_DRAWS goofspiel=True) so the true keep-rate is bracketed [0.33, 0.66] — re-measure on the next gen run and tighten. ⚠ the old tail of this comment called goofspiel a VARIANCE-DOMINATED WASH; that is DISPROVEN (R2 2026-08-03: us 0.298 vs a field near 0.70) — see the TOURNAMENT_TEACHERS block below.
# Validated/strong teacher per PvP env, wired into the TOURNAMENT generation path (_build_generate_cmd),
# NOT just examples/run-r1.sh — so eval == tournament. Before this the tournament ran WEAK defaults
# (binomial / gin-expert / mcts...) even though our head-to-head eval used the strong teachers. Local
# head-to-head 2026-07-05: gin=strong WON 0.611. othello: MCTS-800 was UNLEARNABLE (student 0.179 in
# head-to-head) → switched to heuristic+CoT (learnable, 0.729 vs MCTS-50) — WINNABLE board game, re-eval pending.
# ⚠ VALIDATED WASHES (~0.5 draw, knockout-safe, CANNOT win at 1.5B — 2
# deep-dives + deterministic test 2026-07-06): liars=posterior (the SHIPPED teacher; the 'mccfr' named here previously never existed as code), leduc=handwin
# (a non-folding-random opponent → card-determined; handwin=calldown=mcts ALL 0.445; kept = never-fold 0-self-
# fold safe-floor + insurance vs a future folding opponent=0.709). leduc's wash label is now CONFIRMED by real
# tournament play: 0.500 in ALL FOUR R2 matchups over 296 games (2026-08-03) — but that holds for R1/R2 GROUP play (rank-quantile among peers) ONLY. ⚠ WASH LABEL WITHDRAWN 2026-08-05: the Aug-3 R3 BOSS round, a 2-player head-to-head, has leduc at 16-48 over 74 games (plus a second leg at 0-10 that carries the forfeit signature and is an artifact, so the 74-game leg is the load-bearing evidence). Same env, different scoring geometry. Under this file's own RULE — a wash may only be declared when ALL miners land on the SAME number — leduc no longer qualifies. Its status is UN-EVALUATED, not "knockout-safe". LIMIT: another miner's model, so this kills the structural "cannot separate" claim without telling us whether OUR handwin teacher wins or loses there.
# ⚠⚠ goofspiel was ALSO listed here as a wash. IT IS NOT, AND THE MISLABEL COST US THE 2026-08-03 TOURNAMENT.
# R2 per-env from the real pvp_pair_results: liars 0.534 / leduc 0.500 / GOOFSPIEL 0.298 — last of 5, beaten by
# EVERY opponent incl. the burn hotkey. Between miners goofspiel spreads 0.298→~0.70, i.e. it is the most
# skill-discriminating env we play, the opposite of a wash. Root cause of the mislabel: "no deterministic policy
# beats MCTS-800" was read as "nobody can win it", but PvP never scores us against MCTS — it scores us against
# other MINERS. ⇒ RULE: a wash may only be declared when ALL miners land on the SAME number (leduc), never from a
# teacher-vs-MCTS bot benchmark. Fix shipped 2026-08-04: teacher valuematch → ONEUP (bid the cheapest card that
# still beats the prize). valuematch always bids exactly the prize, so one-up takes every round but the dearest
# → 0.000 for valuematch at every deck size; lab worst-case 0.374 vs 0.000, meta-Nash 0.500 vs 0.206 (the lab
# predicted 0.206, the tournament delivered 0.298). Revert with GOOFSPIEL_TEACHER=valuematch. See
# mastering_goofspiel.py for the full evidence + the sharpened lesson that DETERMINISM ≠ LEARNABILITY.
# clobber = negamax, NOT 'solver' — the solver variant was rejected as unlearnable (student 0.45); see the per-env note below.
# ⚠ THE WIN/WASH TAXONOMY IS UNSETTLED as of 2026-08-05 — it previously read "WINS come from clobber/othello/gin/
# intercode/GOOFSPIEL, liars/leduc are the only washes", and the Aug-3 R3 boss round (a clean 2-player head-to-head,
# straight from pvp_pair_results) points close to the opposite: clobber separated decisively (35-19 and 13-5) and so
# did othello (7-3) — both of which the notes below still label "target DRAW" — while gin, listed as a WIN env, TIED
# both tasks (9-9 and 10-10), and liars AND leduc both LOST. ⚠ LIMIT, do not overread: that was ANOTHER miner's model
# (5EEaxgnm vs boss 5GU4Xkd3), small n. It is enough to kill the STRUCTURAL claim "this env can only draw", because
# that is a claim about the env. It says NOTHING about where OUR teachers land — treat those envs as UN-EVALUATED.
# Overridable per env via export (${VAR:-default}) for A/B experiments.
TOURNAMENT_TEACHERS = {'gin_rummy': ('GIN_TEACHER', 'strong'), 'othello': ('OTHELLO_TEACHER', 'positional'), 'leduc_poker': ('LEDUC_TEACHER', 'handwin'), 'goofspiel': ('GOOFSPIEL_TEACHER', 'oneup'), 'liars_dice': ('LIARS_TEACHER', 'posterior'), 'clobber': ('CLOBBER_TEACHER', 'negamax')}  # clobber: the solver's move-label was an action-id ARTIFACT (lowest-id winning move) + stochastic MCTS fallback → unlearnable (student 0.45); the 'learnable' variant (mobility-argmax over exact-winners + MCTS backbone) STILL LOST 0.40 (43-65) in head-to-head. negamax = an independent iterative-deepening negamax αβ with a SIGNED SIDE-TO-MOVE MOBILITY leaf eval (mastering_clobber), content=None. ⚠ WORDING CORRECTED 2026-08-05: this said "MOBILITY-DIFFERENCE", which is wrong twice over. OpenSpiel's State::LegalActions(Player) returns an EMPTY list for the off-turn player (spiel.h base implementation; clobber overrides only the no-arg LegalActions()), so `len(legal(side)) - len(legal(flip))` evaluates to +m for the mover and -m for the non-mover — the mover's mobility, signed, never a difference. And a genuine difference would be USELESS here anyway: a clobber move is one ordered pair (my piece, adjacent enemy piece), a set in bijection read from either end, so both players always have the same true move count and the difference is identically zero. The code is right; only this sentence was wrong. mastering_clobber's self-test now asserts the empty-off-turn premise, so if a future OpenSpiel changes it the collapse to a constant eval is caught loudly instead of silently. It NEVER solves exact win/loss → smooth deterministic mobility label → learnable → student converges to a draw → target DRAW. (WINS_ONLY clobber=True ≡ sample_by_score power=1.0 on a no-draw game.) # othello: MCTS-800 UNLEARNABLE (0.179); the mobility 'heuristic' was STRONGER-as-teacher (0.79 vs 0.15 vs MCTS) but STILL UNLEARNABLE → student 0.11 in head-to-head EVEN AT 3x data. positional = a 3-rule static positional policy (corner>avoid-X/C>STATIC-WEIGHT-TABLE + CoT; mastering_othello) — the static table is a learnable 1.5B target where mobility (a look-ahead) is not. A learnable static policy → student converges to a draw → target DRAW. Paired: WINS_ONLY othello=False + NASH_DATA_BOOST['othello'] = 1.8 — read the dict, it is the only number that has actually run. Prose elsewhere in this file said 2.0 here and 3.0x twice below; NEITHER shipped. LESSON: gate on local student head-to-head, teacher-strength is the wrong metric.  # liars_dice SHIPS 'posterior' (LiarsPosteriorBot, mastering_liars.py) — that is the in-code default in generate_pvp_toolcalling._make_teacher, and the measured local head-to-head is 0.263 (flat-binomial) -> 0.508, by inferring the opponent's dice from their bid history. ⚠ DANGEROUS STALE TEXT REMOVED 2026-08-05: an earlier version of this note recommended 'mccfr' and claimed posterior gets punished to 0.27. DO NOT ACT ON THAT. There is NO mccfr branch anywhere in this repo — grep shows 'mccfr' occurring ONLY in this file's own comments — so LIARS_TEACHER=mccfr matches no branch and falls through to _make_mcts_bot, the teacher this very file documents as UNLEARNABLE (student 0.179/0.11). "Fixing" the code to match that old text would silently swap a 0.508 teacher for MCTS. Near-Nash game, so ~0.5 is the practical ceiling FOR THE TEACHER; that is a lab statement, NOT a PvP forecast — see the RULE above (a wash may only be declared when ALL miners land on the SAME number).
# Extra training-data (coverage) for LIARS only — helps the student reach the mccfr DRAW (~0.5)
# instead of under-fitting to a loss (posterior sat at 0.377). leduc DROPPED (was 1.5): the
# deterministic test 2026-07-06 shows leduc is a CARD-DETERMINED wash vs the current non-folding opponent
# (handwin=calldown=mcts ALL 0.445) — the student only needs to learn "never fold" (trivial), extra
# rows can't lift a metric ceiling. othello boost ADDED — proposed at 3.0x in this paragraph, but the SHIPPED value is 1.8 (see NASH_DATA_BOOST; do not restore 3.0 or 2.0 from prose): agent-1 head-to-head (live pyspiel) proved the
# othello TEACHER is STRONGER than the reference (0.79 vs 0.15 vs MCTS-50, beats the reference teacher 5:1) — the
# 0.18 student loss is NOT teacher quality, it's DATA STARVATION: a standalone othello run trains on ~8000 games,
# ours got ~123 games in the gin+othello bundle (~65x less). The heuristic teacher is FAST
# (904 games/59s) so it can fill a bigger target. The SHIPPED 1.8 (not the 3.0x written above, which never ran) is a partial catch-up (feasible in the 2-epoch
# window). ⚠ verify the bundle still completes 2 epochs (watchdog) — if it deadline-saves early, lower.
# ⚠ ALLOCATION vs MEASURED SPREAD (2026-08-05) — evidence recorded, numbers deliberately NOT changed.
# Effective per-env share is GAME_TARGETS[env][(hours,n)] * NASH_DATA_BOOST[env] (the _hw_scale and
# _epoch_fit_scale factors are common to all envs and cancel in a comparison; _weights is the dynamic
# headroom term and CANNOT be evaluated offline, so treat these shares as indicative, not exact).
# Cross-opponent spread was tallied from the real pvp_pair_results of Aug-3 + Jul-27 (scratchpad
# env_discrim.pl); IQR is the robust measure, forfeit-signature pairings (10 games, 10-0) excluded:
#
#   env          share(R3)   IQR     games   forfeit-pairs   reading
#   othello        20.0%    0.246      264       38%   highest IQR but THIN + forfeit-dominated
#   liars_dice     17.3%    0.125     1520       12%   real spread, largest field (17 miners)
#   goofspiel      16.7%    0.130      536       11%   real spread
#   gin_rummy      13.3%      -          38        0%   NO DATA AT ALL (2 miners)
#   clobber        12.2%    0.039     1176        7%   WASH on a large sample
#   intercode      11.1%      -           -        -   INDIVIDUAL eval, not PvP — not comparable here
#   leduc_poker     9.4%    0.016     2946       10%   WASH, the strongest such result in the dataset
#
# So the LARGEST slice goes to othello, whose apparent discriminativeness is substantially a FORFEIT
# phenomenon (38% of its pairings are 10-game 10-0 sweeps, 3-5x every other env, and it averages only
# 14 games/pairing vs leduc's 74.7 — othello games are expensive, so few fit the 900s matchup budget).
# If that reading is right, the othello edge is RELIABILITY (don't forfeit, don't be slow), which more
# training data does not buy. Meanwhile 21.6% goes to two MEASURED washes (clobber + leduc).
#
# NOT ACTED ON, and these are the honest reasons:
#   * The othello boost was justified by a DIFFERENT argument — data starvation (~123 games in the
#     gin+othello bundle vs ~8000 standalone). Starvation and spread are separate claims; if we really
#     are starved there, boosting is right regardless of how discriminative the env is.
#   * Every env can be drawn at random in R1, so a wash cannot be zeroed out — you still need enough
#     data to REACH the wash. What the evidence argues against is boosting a wash ABOVE baseline.
#   * The sum of the extra weight is fixed, so this is zero-sum: moving mass to liars/goofspiel takes
#     it from somewhere, and no student head-to-head has been run to show that trade wins.
# ↩ PARTIAL RETRACTION, 2026-08-05, same day: the othello slice now looks DEFENSIBLE, for a third
# reason neither argument above captured. validator core/pvp/tools.py:90-95 says the legal-action
# `enum` on game_action.action_id is ADVISORY ONLY — "SGLang with tool_choice='auto' does not
# grammar-enforce argument values (verified empirically — malformed/illegal output reaches the
# wire), so the binding guard is the bot's validate-then-forfeit". No retry. So the model must
# self-select a legal id every single turn, and othello is the worst case in the pool: up to 65 ids
# over a 64-cell board whose legality depends on flip-lines the model has to compute. Under that
# reading othello's 1.8 is not a bet on the env being discriminative — it is FORFEIT INSURANCE, and
# forfeits are what actually decide othello (38% of its pairings are 10-0 sweeps, half the field
# has taken one). ⚠ NOT proven: clobber has a comparable action space and the LOWEST forfeit rate
# (7%), so raw action count cannot be the whole story. Still GPU-gate any change; the point of this
# note is that the earlier "largest slice is misallocated" reading is now contested by evidence, and
# should not be acted on as if it were settled.
# GPU-gate it: measure student head-to-head per env first, then reallocate. Do not tune this blind.
NASH_DATA_BOOST = {'liars_dice': 1.4, 'othello': 1.8, 'goofspiel': 1.5, 'gin_rummy': 1.2, 'clobber': 1.1}  # per-env data up-weight, spread across more envs to smooth the training mix. The sum of the extra weight is held constant, so the total sample budget (and thus the 2-epoch wall-clock fill of the training window) is unchanged. Board/positional envs keep a higher share (they need more examples to reproduce a deterministic policy cleanly); values selected on local student-vs-teacher evaluation.
STRATIFIED_RETENTION = 0.5
_MAX_NUM_GAMES = 25000
_TARGET_SAFETY = 1.1
_INTERCODE_MAX_VARIANTS = 20
_EPOCH_FIT_SCALE = 0.55
WINDOW_RATIO = {'gin_rummy': 22.0, 'liars_dice': 1.0, 'leduc_poker': 1.5, 'othello': 15.0, 'goofspiel': 8.0, 'clobber': 8.0}
GAME_CONFIGS = {'liars_dice': {'max_turn': 30, 'sample_by_score': False, 'score_power_1_5h': 2.0, 'score_power_3h': 2.5, 'min_periodic_save': 100}, 'gin_rummy': {'max_turn': 200, 'sample_by_score': False, 'score_power_1_5h': 1.5, 'score_power_3h': 2.0, 'min_periodic_save': 100}, 'leduc_poker': {'max_turn': 10, 'sample_by_score': False, 'score_power_1_5h': 3.0, 'score_power_3h': 3.0, 'min_periodic_save': 150}, 'intercode': {'max_turn': 10, 'sample_by_score': False, 'score_power_1_5h': 1.0, 'score_power_3h': 1.0, 'min_periodic_save': 100}, 'othello': {'max_turn': 70, 'sample_by_score': False, 'score_power_1_5h': 1.5, 'score_power_3h': 2.0, 'min_periodic_save': 100}, 'goofspiel': {'max_turn': 30, 'sample_by_score': False, 'score_power_1_5h': 1.5, 'score_power_3h': 2.0, 'min_periodic_save': 100}, 'clobber': {'max_turn': 70, 'sample_by_score': False, 'score_power_1_5h': 1.5, 'score_power_3h': 2.0, 'min_periodic_save': 100}}

def _is_long_tier(hours: float) -> bool:
    return hours == LONG_TIER_HOURS

# Continuous epoch-fit constants. The knee is the load below which no trim is needed: two envs on
# one GPU, the round that won untrimmed. The exponent is solved from the two trimmed anchors rather
# than chosen, and the floor bounds how far a very heavy cell can be cut.
_EPOCH_FIT_KNEE = 2.0
_EPOCH_FIT_CURVE = 1.4742
_EPOCH_FIT_FLOOR = 0.45

def _epoch_fit_scale_for(num_envs: int, gpu_nums: int, hours: float) -> float:
    """Trim the per-env target so the learning-rate anneal FINISHES inside the window.

    The anneal is stretched over epochs times steps-per-epoch. If the deadline arrives first, the
    saved weights sit at a high learning rate. Measured across three rounds: 4 envs on 2 GPUs
    stopped at 1.04 of 2 epochs with 62.4% of peak LR remaining and lost; 2 envs on 1 GPU reached
    1.56 and came down to 34%; 6 envs on 4 GPUs reached 1.45 and came down to 39.2%.

    Anchors that are deliberately preserved:
      - the 3.0h from-scratch tier: 1.0, its budget is loose and has never been a problem
      - 6 or more envs: _EPOCH_FIT_SCALE, the value the 6-env round won with
      - 2 or fewer envs: 1.0, that round won untrimmed

    The cell that is demonstrably broken is 3-5 envs on 2 or fewer GPUs, which the earlier gate
    missed entirely because it keyed on env count alone and was blind to GPU count. The 0.60
    default is arithmetic rather than a guess: half of the 1.04 epochs actually run, against a 0.9
    completion target, is roughly 0.58.
    """
    if _is_long_tier(hours):
        return 1.0

    # CONTINUOUS in the load, with the earlier step function kept as a floor wherever it had
    # evidence. The quantity that stretches the anneal is work per device: env variety lengthens
    # games and so lengthens sequences, while devices divide the resulting steps. Scale on that
    # directly instead of on two hand-placed cliffs.
    #
    # The exponent is SOLVED from the existing anchors, not fitted: 1.4744 from the six-env,
    # four-GPU round and 1.4739 from the four-env, two-GPU one. They agree to 0.0005 despite being
    # earned in different rounds on different GPU counts, which is the evidence that this load is
    # what the step function was crudely approximating. The knee sits at the two-env, one-GPU round,
    # which won untrimmed.
    #
    # What the step function missed, this catches: six envs on TWO GPUs used to receive the same
    # 0.55 as six on four, blind to the device count that halves throughput. It now trims to the
    # floor. Five envs on four GPUs received no trim at all and now gets 0.72.
    #
    # HONEST LIMIT: from four envs upward on a SINGLE GPU this reaches the floor, trimming harder
    # than the step function ever did. The direction is sound -- four envs on one device is heavier
    # than four on two -- but no anchor measured a 1-GPU multi-env round, so the magnitude there is
    # inference. It stands because over-trimming costs data while under-trimming costs the round,
    # and the worst case still plans over 4,000 rows. gpu_count=1 is a live path for sub-1B models,
    # so when that band first runs, read the deadline-save signature: an anneal that finishes with
    # time to spare means this floor is too aggressive and now has a real anchor to be raised on.
    #
    # And where the two disagree in the other direction, the step function WINS. Its mid-round
    # branch is the only cell backed by an observed loss, while this curve's claim that a three-env
    # round sits near the untrimmed anchor is inference. Measurement outranks inference, and the
    # asymmetry favours it: over-trimming costs data, under-trimming costs the round. So the result
    # is the smaller of the two wherever the step function had a branch.
    load = float(num_envs) / max(1.0, float(gpu_nums)) ** 0.5
    curve = 1.0 if load <= _EPOCH_FIT_KNEE else max(
        _EPOCH_FIT_FLOOR, min(1.0, (_EPOCH_FIT_KNEE / load) ** _EPOCH_FIT_CURVE))

    evidenced = 1.0
    if num_envs >= 6:
        evidenced = _EPOCH_FIT_SCALE
    elif num_envs >= 3 and gpu_nums <= 2:
        try:
            evidenced = max(0.3, min(1.0, float(os.environ.get('SFT_EPOCH_FIT_MIDROUND', '0.6'))))
        except ValueError:
            evidenced = 0.6
    return min(curve, evidenced)


def _compute_target_samples(env_name: str, num_envs: int, hours: float) -> int:
    # The old tier mapping tested `hours == 3.0` EXACTLY, but 2.5-hour production tasks do occur
    # (observed: an swe task at 2.5h via ENV_TRAINING_HOURS_SWE_INFINITE), and those read as the
    # 1.5h tier and under-provisioned the target by roughly a third of the window. Interpolate
    # LINEARLY between the two anchors (1.5, 3.0) instead. Outside that range the slope continues,
    # staying monotonic in hours, with a floor at 40% of the lower anchor so a very short window
    # can never produce a target of zero. Over-provisioning is safe by design -- generation stops
    # at target_samples or the time limit -- while under-provisioning wastes the window silently.
    targets = GAME_TARGETS.get(env_name)
    if not targets:
        targets = GAME_TARGETS['liars_dice']
    n_key = min(max(num_envs, 1), 6)
    t_lo = targets.get((REGULAR_HOURS, n_key), targets[REGULAR_HOURS, 3])
    t_hi = targets.get((LONG_TIER_HOURS, n_key), t_lo)
    frac = (hours - REGULAR_HOURS) / (LONG_TIER_HOURS - REGULAR_HOURS)
    raw = t_lo + (t_hi - t_lo) * frac
    out = int(max(raw, 0.4 * t_lo))
    if num_envs > 6:
        # Jumlah env bertambah tiap ronde; klamp bisu min(n,6) akan meng-over-provision
        # per env saat n>6. Ekstrapolasi x6/n + peringatan (spesifikasi WO-3).
        out = max(1, int(out * 6.0 / num_envs))
        print(f'[sft_env_config] target {env_name}: num_envs={num_envs} > 6 -> skala x6/{num_envs}', flush=True)
    print(f'[sft_env_config] target {env_name}: jam={hours} frac={frac:.3f} '
          f'jangkar=({t_lo},{t_hi}) n_key={n_key} -> {out}', flush=True)
    return out

def _samples_to_games(env_name: str, target_samples: int) -> int:
    ratio = WINDOW_RATIO.get(env_name, 1.0)
    if WINS_ONLY_PER_ENV.get(env_name, False):
        win_rate = WIN_RATE_ESTIMATE.get(env_name, 0.5)
        games = int(target_samples / win_rate / STRATIFIED_RETENTION / ratio * 1.1)
    else:
        games = int(target_samples / ratio * 1.1)
    return min(games, _MAX_NUM_GAMES)

_HEADROOM_W_LO = 0.6
_HEADROOM_W_HI = 1.6

def _headroom_weights(env_names: list, lo: float = _HEADROOM_W_LO, hi: float = _HEADROOM_W_HI) -> dict:
    """Per-env data weights from the validator's base-model baseline (P1.2).

    EnvStats.mean_score (core/pvp/baseline.py) = the BASE model's win-rate vs MCTS-50 in
    [0,1], present only for PvP pyspiel envs (NOT intercode, which has no in-harness agent).
    Lower mean_score = weaker base = more headroom = up-weight that env's training share.
    Weights = headroom / mean(headroom) (so mean weight ~= 1.0 → total budget preserved),
    clipped to [lo,hi] (conservative). Envs without a baseline (intercode) get no entry →
    caller defaults them to 1.0; no BASELINE_STATS (local sim) → {} → uniform (no-op)."""
    try:
        from pvp_infra.baseline_stats import load_baseline_stats, get_env_score_stats
        stats = load_baseline_stats()
        if not stats:
            return {}
        scored = {}
        probe_crashed = []
        for e in env_names:
            es = get_env_score_stats(stats, e)
            if es is None:
                # No entry at all means the env has no baseline by design, so leave it absent
                # and let the caller default it to 1.0.
                continue
            if isinstance(es, dict) and es.get('num_episodes', 0) and 'mean_score' in es:
                scored[e] = float(es['mean_score'])
            else:
                # An entry that EXISTS but is degenerate (num_episodes=0, or no mean_score) means
                # the baseline probe crashed for this env, not that the env has no baseline. Such
                # envs used to be dropped silently from `scored`, so nothing downstream could tell
                # a crashed probe from an env that never had one.
                #
                # They are recorded here and given a NEUTRAL weight, applied outside the
                # normalisation so the weights of measured envs do not shift by a single bit. The
                # weight is neutral rather than the ceiling because a crashed probe is zero evidence
                # in EITHER direction: it does not show the base model is strong here, but neither
                # does it show the opposite, and the ceiling asserts headroom nothing measured.
                # That assertion is not free -- weights compete for share, so it is taken from envs
                # whose scores WERE measured (in the six-env final it moved gin from 17.9% to 25.9%
                # of the round at their expense). An env that genuinely needs more data should get
                # it through its boost, which is an argued number, rather than through the silent
                # side effect of a failed probe.
                probe_crashed.append(e)
        fallback = {e: 1.0 for e in probe_crashed}
        if probe_crashed:
            print(f'[sft_env_config] headroom fallback (baseline probe crashed, num_episodes=0): '
                  f'{sorted(probe_crashed)} -> neutral weight 1.0', flush=True)
        if len(scored) < 2:
            out = dict(fallback)
        else:
            headroom = {e: max(0.0, 1.0 - m) for (e, m) in scored.items()}
            avg = sum(headroom.values()) / len(headroom)
            if avg <= 0:
                out = dict(fallback)
            else:
                out = {e: min(hi, max(lo, h / avg)) for (e, h) in headroom.items()}
                out.update(fallback)
        if 'intercode' in out:
            # Intercode always lands on the clip floor, because a high base score leaves little
            # headroom. Its win-loss-win pattern across rounds is driven by the learning-rate
            # schedule, not by data share, so pin it to 1.0 AFTER normalisation. Intercode stays
            # inside the average: excluding it from the mean instead shifts every other env's
            # weight (liars 1.3506 -> 1.0019), which damages the envs that are already winning.
            out['intercode'] = 1.0
        return out
    except Exception as exc:
        print(f'[sft_env_config] headroom weighting skipped: {type(exc).__name__}: {exc}', flush=True)
        return {}

def get_sft_env_config(param_nums: int) -> dict:
    if param_nums is None:
        param_nums = 7600000000
    if param_nums < 1000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['0_1_b'])
    if param_nums < 2000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['1_2_b'])
    if param_nums < 4000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['2_4_b'])
    if param_nums < 5000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['4_5_b'])
    if param_nums < 9000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['5_9_b'])
    if param_nums < 12000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['9_12_b'])
    if param_nums < 15000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['12_15_b'])
    if param_nums < 40000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['15_40_b'])
    if param_nums < 80000000000:
        return deepcopy(SFT_ENV_SIZE_CONFIG['40_80_b'])
    print(f'Model size {param_nums} unsupported, fallback 40_80_b', flush=True)
    return deepcopy(SFT_ENV_SIZE_CONFIG['40_80_b'])

def _build_run_cmd(config: dict, model_path: str, dataset_path: str) -> str:
    gpu_nums = int(config.get('gpu_nums', 1) or 1)
    if gpu_nums > 1:
        launcher = f'NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node={gpu_nums}'
    else:
        launcher = 'python'
    parts = [f'{launcher} train_sft_env.py', f'--model-name-or-path {model_path}', f'--dataset-path {dataset_path}', f"--output-dir {config['output_dir']}", f"--epochs {config['epoch_num']}", f"--learning-rate {config['learning_rate']}", f"--batch-size {config['batch_size']}", f"--grad-accum-steps {config['gradient_accumulation_steps']}", f'--warmup-steps 35', f"--min-lr-rate {config.get('min_lr_rate', 0.25)}", f"--max-seq-length {config.get('max_seq_length', 4096)}", f"--max-steps {config.get('max_steps', 100)}", f"--request-path {config['request_path']}", f'--report-to wandb']
    if config.get('use_lora', False):
        parts.extend([f"--lora-r {config.get('lora_r', 128)}", f"--lora-alpha {config.get('lora_alpha', 512)}", f"--lora-dropout {config.get('lora_dropout', 0.0)}"])
    else:
        parts.append('--no-lora')
    if config.get('gradient_checkpointing', True):
        parts.append('--gradient-checkpointing')
    if str(config.get('use_liger_kernel', '')).strip().lower() == 'true':
        parts.append('--use-liger-kernel')
    return ' '.join(parts)

def get_generation_time_ratio(param_nums) -> float:
    """Fraction of the wall-clock budget spent on generation (the rest = training). Smaller
    models train faster, so they can afford to spend more time generating. Keeps generation
    wall-clock-bounded so it can never overrun into the training budget."""
    if param_nums is None:
        param_nums = 2_000_000_000
    if param_nums < 2_000_000_000:
        return 0.35
    if param_nums < 4_000_000_000:
        return 0.25
    if param_nums < 6_000_000_000:
        return 0.20
    return 0.15

def _build_generate_cmd(env_name: str, output_path: str, num_games: int, target_samples: int, hours: float, time_limit_seconds: float = 0.0, model_architecture: str = '') -> str:
    if env_name == 'swe_infinite':
        # swe_infinite (final-round task-3, INDIVIDUAL, DECISIVE strict-win env). NOT a
        # pyspiel game and has NO env-server (miner.md: ENVIRONMENT_SERVER_URLS absent for
        # swe-only tasks) — so it must NOT fall through to generate_pvp_toolcalling below (that
        # would crash → un-noised base → draw → the incumbent baseline retains the lead). SFT-distill from the validator-
        # pre-downloaded whitelisted SWE-ZERO in MINER_DATASETS (self-gen MiniSWE fallback if
        # absent). Static dataset read → fills instantly, no time-box (can't overrun the window).
        return f'python -m envs.mastering_swe --output_path {output_path} --target_samples {target_samples}'
    if env_name == 'intercode':
        # NOTE: intercode is intentionally NOT time-boxed (no --time_limit_seconds). It is
        # self-generated in-container (shell-verified templates, no MCTS/search), so it fills
        # its target near-instantly and can never overrun the window like the pyspiel teachers.
        # Do NOT add a time-box here — it would truncate the set for no reliability gain.
        variants = max(4, min(_INTERCODE_MAX_VARIANTS, round(target_samples / (200 * 2))))
        # I3 (2026-09-19, local GPU replay): our trained model used a 2nd command in 0/20 eval
        # tasks (the untrained base model: 19/20) because --recovery-frac had defaulted to 0.0 since
        # its introduction — every self-gen submit row followed exactly one prior command.
        # Turning it on here (production invocation only; the flag's own default stays 0.0 for
        # any standalone/manual call) plus raising --refine-frac from its 0.5 default makes
        # submit rows that follow >=2 commands the MAJORITY of self-gen submit evidence, without
        # touching the trained-target rule (only CORRECT actions are ever a training target;
        # build_recovery_example's near-miss is history-only by construction). Recovery near-miss
        # observations are now REAL shell output (selfgen_bash._run_near_miss_in), never the
        # hand-written fake error strings this arm removed — see mastering_intercode.py's
        # build_recovery_example docstring. Numbers (0.65 / 0.30) sized from a measured
        # refine-eligibility rate (27/45 templates have a verbose sibling) and a measured
        # near-miss-availability rate on read-only templates so the resulting share of >=2-command
        # submit rows clears the >=55% target with margin — see experiments/i3/README.md A2 for
        # the actual measured share, this comment's numbers are the SIZING inputs, not a
        # re-measurement. GATE ON EVAL like every other self-gen recipe knob in this file.
        cmd = f'python -m envs.mastering_intercode --output_path {output_path} --task_id_min 1 --task_id_max 200 --variants {variants} --target_samples {target_samples} --recovery-frac 0.30 --refine-frac 0.65'
        # L4 (deep-dive A.2/A.5 D5-D6): eval_intercode._build_tool_messages appends a
        # `Thought N:` history line ONLY when result.content is truthy; Llama-3.2-3B-Instruct and
        # Meta-Llama-3.1-8B-Instruct DROP assistant `content` whenever `tool_calls` is present (the
        # other 4 whitelisted bases keep it), so a model SFT'd from either Llama base can never
        # emit a content the eval would treat as truthy — its eval-time history can never carry a
        # Thought line either. Gate exactly like the existing get_use_liger/disable_flash_attention
        # architecture checks (shared.model_utility.get_model_architecture, lower-cased). Every
        # non-Llama base (all Qwen in the roster) keeps 'on' — byte-identical to HEAD.
        if model_architecture.strip().lower() == 'llamaforcausallm':
            cmd += ' --history-thoughts off'
        else:
            cmd += ' --history-thoughts on'
        return cmd
    cmd = f'python -m envs.generate_pvp_toolcalling --game {env_name} --output_path {output_path} --num_games {num_games} --target_samples {target_samples}'
    _teach = TOURNAMENT_TEACHERS.get(env_name)
    if _teach:
        # Wire the validated/strong teacher into the TOURNAMENT gen path. ${VAR:-default} stays
        # overridable via env for A/B; default = the strong teacher. PvP rounds are multi-env → the
        # gen block runs via shell ('(cmd) & (cmd) & wait'), so the ${:-} + env-prefix always resolve.
        cmd = _teach[0] + '=${' + _teach[0] + ':-' + _teach[1] + '} ' + cmd
    if time_limit_seconds and time_limit_seconds > 0:
        cmd += f' --time_limit_seconds {int(time_limit_seconds)}'
    if not REFLECTION_PER_ENV.get(env_name, True):
        # E4/ARM-A1: liars_dice only (see REFLECTION_PER_ENV above). generate_pvp_toolcalling
        # already implements --no-reflection (args.no_reflection gates the reflection-row
        # extend at line ~500) — this just wires the per-env default into the command the
        # tournament actually runs, mirroring how WINS_ONLY_PER_ENV below wires --wins-only.
        cmd += ' --no-reflection'
    if WINS_ONLY_PER_ENV.get(env_name, False):
        cmd += ' --wins-only'
        if INCLUDE_DRAWS_PER_ENV.get(env_name, False):
            cmd += ' --include-draws'
    return cmd

def _peek_is_instruction_kind(path: str) -> bool:
    """Cheap PARSE-ONLY peek: does the raw snapshot at `path` look like an instruction+response
    (bigcode-shaped) dataset, WITHOUT fully loading it? `_usable_bigcode_pool_size` calls this
    before `load_miner_dataset_dir` so a 2GB messages-shaped dataset (SWE-ZERO,
    pvp-tool-calling-sft) is never fully materialized just to compute the bigcode pool size — only
    the schema (parquet) or a bounded byte-prefix (json/jsonl) is read. Never executes anything;
    a false negative here just means `load_miner_dataset_dir` (below) never runs for that dir,
    which is safe (worst case: U undercounts, never overcounts)."""
    from shared.miner_datasets import find_data_files
    try:
        files, fmt = find_data_files(path)
    except Exception:
        return False
    if not files:
        return False
    try:
        if fmt == 'parquet':
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(files[0]).schema_arrow.names)
            return {'instruction', 'response'} <= names
        # json / jsonl: read a bounded prefix (not the whole file — SWE-ZERO's jsonl is ~2GB) and
        # look for both field names textually. A false positive here just means we go on to the
        # real (correct) column check after load_miner_dataset_dir; a false negative only matters
        # if the field names are pushed past the prefix window, in which case U undercounts (safe).
        with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
            prefix = f.read(8192)
        return '"instruction"' in prefix and '"response"' in prefix
    except Exception:
        return False


def _usable_bigcode_pool_size() -> int:
    """U = size of the whitelisted bigcode pool AFTER the exact same filter+dedupe that
    combine_datasets._blend_bigcode_bash applies at blend time (both call
    envs.mastering_intercode.usable_bigcode_rows, so this count cannot drift from what actually
    gets blended). Reuses shared.miner_datasets.load_miner_dataset_dir (F1: do not re-implement
    the loader here) against the SAME MINER_DATASETS_DIR/MINER_DATASETS contract
    combine_datasets._read_miner_datasets uses. `_peek_is_instruction_kind` skips a full load for
    any mounted dir that doesn't look instruction-shaped (avoids materializing a 2GB SWE-ZERO/
    pvp-sft snapshot just to size the bigcode blend).

    The ENTIRE body is wrapped in try/except: this function runs at config-generation time inside
    get_training_json, and any exception escaping it (dataset I/O, schema surprises, a corrupt
    snapshot) would propagate up and make the WHOLE task fail closed (text_trainer ships an
    un-noised base model for every env in the round, not just intercode) — a blast radius wildly
    out of proportion to "the bigcode blend didn't work this run". Every exit path, including the
    except, returns 0 (self-gen-only fallback) and logs a line containing a quiet_mode keep-marker
    ('not found' / 'fallback') so the silent-in-quiet-mode default still leaves a trace."""
    try:
        root = os.environ.get('MINER_DATASETS_DIR')
        names = [n.strip() for n in os.environ.get('MINER_DATASETS', '').split(',') if n.strip()]
        if not root or not names:
            print('[sft_env_config] bigcode pool: MINER_DATASETS not set — not found, '
                  'falling back to self-gen-only sizing (B=0)', flush=True)
            return 0
        from shared.miner_datasets import load_miner_dataset_dir
        from envs.mastering_intercode import usable_bigcode_rows
        for name in names:
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            if not _peek_is_instruction_kind(path):
                print(f'[sft_env_config] bigcode pool: {name} not instruction-shaped by cheap '
                      f'peek — skip (no usable full-load attempted)', flush=True)
                continue
            try:
                ds = load_miner_dataset_dir(path)
            except Exception as exc:
                print(f'[sft_env_config] bigcode pool: {name} load failed: {type(exc).__name__}: '
                      f'{exc} (skip, falling back)', flush=True)
                continue
            cols = set(ds.column_names)
            if 'messages' in cols:
                continue  # pvp_sft or SWE-ZERO shape, not the instruction-kind bigcode dataset
            if {'instruction', 'response'} <= cols:
                pairs, _dup, _filtered = usable_bigcode_rows(ds)
                print(f'[sft_env_config] bigcode pool: {name} usable rows U={len(pairs)} '
                      f'(dropped {_dup} dup / {_filtered} unsafe-or-destructive-or-placeholder)', flush=True)
                return len(pairs)
        print('[sft_env_config] bigcode pool: no instruction-kind dataset mounted under '
              'MINER_DATASETS — not found, falling back to self-gen-only sizing (B=0)', flush=True)
        return 0
    except Exception as exc:
        # Belt-and-suspenders (FAIL-2): anything unexpected above must never escape and fail the
        # whole training task closed. Falling back to self-gen-only is always safe.
        print(f'[sft_env_config] bigcode pool: unexpected {type(exc).__name__}: {exc} — '
              f'falling back to self-gen-only sizing (B=0)', flush=True)
        return 0

def get_training_json(train_info: dict) -> dict:
    model_name = train_info['model_name']
    model_path = train_info['model_path']
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_sft_env_config(param_nums)
    _5_9b_bucket_arm = _is_5_9b_bucket_arm(config)
    task_id = train_info['task_id']
    _dt = train_info.get('dataset_type', {})
    _env_names = _dt.get('environment_names')
    if _env_names and len(_env_names) > 0:
        env_names = list(_env_names)
    else:
        env_names = [_dt.get('environment_name', 'liars_dice')]
    num_envs = len(env_names)
    hours = float(train_info.get('hours_to_complete', 1.5))
    # _hw_scale is calibrated against a 4-GPU reference: gpu_count must therefore express the
    # CEILING we can use, not a floor. Clamping the small buckets to 1 silently did two things at
    # once — it cut every per-env sample target to 27.5% AND dropped the torchrun launcher — which
    # is why a 3B run finished a whole epoch-pair early with the GPU idle. min() still guarantees
    # we never ask for more devices than the container actually has.
    _gpu_nums = max(1, min(get_gpu_count(), config['gpu_count']))
    _hw_scale = _gpu_nums / 4.0 * _TARGET_SAFETY
    print(f'[sft_env_config] devices={_gpu_nums} (config ceiling {config["gpu_count"]}, visible {get_gpu_count()}) → target scale {_hw_scale:.3f}', flush=True)
    combined_dataset_path = f'/workspace/scripts/datasets/sft_env_{task_id}'
    generate_cmds = []
    per_env_paths = []
    _weights = _headroom_weights(env_names)
    if _weights:
        print(f'[sft_env_config] per-env headroom weights from baseline_stats: {_weights}', flush=True)
    # Epoch-fit: the 6-env round mixes long-game envs (gin max_turn 200, othello 70) → long
    # sequences → ~8.4 s/step, so the full ~30k target only reaches ~1 epoch in the tight 1.5h
    # budget (deadline-save at ~505/948 steps). Scale the per-env target down for the 6-env
    # final's 1.5h tasks (continuation + prior-adapter base) so the targeted ~2 epochs fit. Cause is
    # env-variety (long games), NOT memory — so this is gated on num_envs only. UNTOUCHED: R1/R2
    # (num_envs<6) and the 3.0h from-scratch tier (_is_long_tier). No env is dropped — each still
    # generates, just with fewer samples (more epochs over slightly less data).
    _epoch_fit_scale = _epoch_fit_scale_for(num_envs, _gpu_nums, hours)
    if _epoch_fit_scale != 1.0:
        print(f'[sft_env_config] epoch-fit: {num_envs} env @ {hours}h {_gpu_nums} GPU -> per-env target x{_epoch_fit_scale:.3f} (the anneal schedule must finish)', flush=True)
    _gen_seconds = int(hours * 3600 * get_generation_time_ratio(param_nums))
    print(f'[sft_env_config] generation time-box: {_gen_seconds}s/env window (ratio {get_generation_time_ratio(param_nums)} of {hours}h); parallel envs each stop at min(target, window) → generation can never starve training', flush=True)
    # _env_targets: env -> computed target_samples (all the existing per-env scaling applied).
    # Filled fully BEFORE any bigcode sizing, so the B computation below (num_envs>1, intercode
    # present) always sees every env's real target, including intercode's own I_sg — no ordering
    # dependency on env_names position.
    _env_targets = {}
    for env in env_names:
        _env_targets[env] = max(1, int(_compute_target_samples(env, num_envs, hours) * _hw_scale * _weights.get(env, 1.0) * _epoch_fit_scale * NASH_DATA_BOOST.get(env, 1.0)))
    if num_envs == 1:
        # No combine_datasets step at all in the solo-env path (below), so the bigcode blend
        # never fires here regardless of which env this is — unchanged, pre-existing behaviour.
        env = env_names[0]
        target = _env_targets[env]
        num_games = _samples_to_games(env, target)
        per_env_path = combined_dataset_path
        per_env_paths.append(per_env_path)
        generate_cmds.append(_build_generate_cmd(env, per_env_path, num_games, target, hours, _gen_seconds, model_architecture))
        full_generate_cmd = generate_cmds[0]
    else:
        _bigcode_rows = 0
        if 'intercode' in env_names:
            # F2 design (replaces the old flat `_bigcode_frac = 0.15 if 'intercode' in env_names
            # else 0.0`, which added bigcode ON TOP of the round's rows and gave frac=0 in most
            # PvP pairings). Hold total rows FLAT: B is carved OUT of intercode's own self-gen
            # target, not added on top, and this branch fires in EVERY intercode pairing.
            #   I_sg  = intercode's already-computed target (_env_targets['intercode'] above,
            #           includes the existing _hw_scale/_epoch_fit_scale/NASH_DATA_BOOST/headroom
            #           scaling — B is sized off the FINAL scaled target, not a raw one).
            #   U     = usable bigcode pool size at config time (_usable_bigcode_pool_size, shared
            #           filter with combine_datasets._blend_bigcode_bash via
            #           envs.mastering_intercode.usable_bigcode_rows — cannot drift between the two).
            #   floor = _SELF_GEN_FLOOR_FRAC * I_sg, a self-gen floor so the execute->submit
            #           multi-turn anchor trajectories (selfgen_bash's recovery/submit rows) always
            #           survive the blend. No existing constant in this file implied a floor value,
            #           so 0.25 is a new, explicit, documented choice (not derived from measurement).
            #   B     = min(U, I_sg - floor)
            # v5 (2026-09): DROPPED the P=0.5 share term (was B = min(U, round(P*I_sg), I_sg-floor))
            # — measured against the real tourn-0907 R1 leaderboard, P=0.5 was leaving most of the
            # usable pool on the table for no documented reason. The top 2 intercode scorers of
            # that round trained on ~6k bigcode rows for 2 FULL epochs, not a 0.5-of-target slice:
            #   5EUHoj: 5,993 bigcode rows = 45% of a 13,341-row intercode set (2 epochs)
            #   5CRjuf: 7,682 bigcode rows = 43% of a 17,881-row intercode set (2 epochs)
            # v4's P=0.5 gave, at the real R1 cell (Llama-3.1-8B, 2xH100, I_sg=7425): B=3712 (18%
            # of the round's 20790 rows) — well below what the top scorers actually ran. Dropping P
            # lets B claim the WHOLE usable pool up to the self-gen floor: same cell now gives
            # B=min(5193, 5569)=5193 (the full pool), self-gen after=2232 (still >= the 1856 floor).
            # The self-gen floor (unchanged, 0.25) is now the ONLY thing capping B on the low side;
            # the pool size U is now the ONLY thing capping it on the high side. Second argument for
            # dropping P specifically (not raising it, or picking some other number): R1 self-gen
            # rows are only 4 / 6 / 20 paraphrase variants (capped at _INTERCODE_MAX_VARIANTS=20,
            # `variants = max(4, min(20, round(target_samples/400)))` above) over the SAME 200
            # underlying template task ids (--task_id_max 200) for 1/2/4 GPU respectively — i.e.
            # highly repetitive at this row count (a handful of paraphrasings of 200 tasks), whereas
            # each bigcode row is a DISTINCT real NL2Bash instruction/command pair (12000 originally,
            # 5193 after the usable filter). Swapping repetitive self-gen variants for unique bigcode
            # rows raises instruction diversity (200-task self-gen pool vs a 5193-row distinct pool)
            # without changing the token budget (measured mean token count, Llama tokenizer:
            # bigcode execute-only 903, self-gen execute-turn 899, self-gen submit-turn 925 — all
            # ~900, so this is a diversity trade, not a length trade). GATE ON EVAL — still an A/B
            # knob, just a different one (no P at all).
            #
            # I3 (2026-09-19): the "U rows / U pairs" unit mismatch this v5 note describes (B was
            # sized in ROW units, U in PAIR units, under the 2-rows-per-pair E10 shape) is fixed at
            # its source (combine_datasets._blend_bigcode_bash is now 1 row per pair) rather than
            # patched here — B below is a row count and a pair count simultaneously now, so this
            # comment's arithmetic examples (e.g. "B=min(5193,5569)=5193") describe the pre-I3
            # units and should not be read as current numbers. This is the ONLY change I3 makes to
            # the B/floor formula itself; the floor constant and its arithmetic (below) are
            # UNCHANGED (an earlier draft of this arm tried a `_natural_selfgen`-based floor
            # reduction here — ft-contract-guard BLOCKED it: measured against the real 216-cell
            # matrix and the real 5,193-pair pool, it collapsed self-gen's row-floor at several
            # cells [e.g. 2GPU/3env/3h: 1,444->582 rows, -63% submit-row exposure] since Change 1
            # already made self-gen the ONLY source of submit rows in this file, and it left B
            # UNCHANGED in 143/144 of that matrix's cells (zero benefit) while the only cells it did
            # move were the ones it damaged — reverted in full, see
            # experiments/i3/README.md "FIX-1"). See experiments/i3/README.md for the corrected
            # before/after B table (real 5,193-pair pool) per (bucket x gpu_count x num_envs) cell.
            _SELF_GEN_FLOOR_FRAC = 0.25
            _i_sg = _env_targets['intercode']
            _floor = max(1, round(_SELF_GEN_FLOOR_FRAC * _i_sg))
            # _pool_flag: _usable_bigcode_pool_size's ENTIRE body is already try/except-wrapped
            # (FAIL-2, see its docstring) and always returns an int, so the extra try/except here
            # only guards against something escaping that contract in the future — belt-and-
            # suspenders, not a live path today, but it is what makes 'fail' a real third state
            # rather than a value this line could never actually take.
            try:
                _u_pool = _usable_bigcode_pool_size()
                _pool_flag = 'ok' if _u_pool > 0 else 'none'
            except Exception as _exc:
                _u_pool = 0
                _pool_flag = 'fail'
                print(f'[sft_env_config] bigcode pool sizing failed: {type(_exc).__name__} — falling back to self-gen-only (B=0)', flush=True)
            _bigcode_rows = max(0, min(_u_pool, _i_sg - _floor))
            _env_targets['intercode'] = max(1, _i_sg - _bigcode_rows)
            print(f'[sft_env_config] intercode bigcode blend: I_sg={_i_sg} U={_u_pool} '
                  f'floor={_floor} -> B={_bigcode_rows}, '
                  f'self-gen target after blend={_env_targets["intercode"]}', flush=True)
            # Loud twin of the line above, FLAG-ONLY (not U/B): ft-contract-guard flagged an
            # earlier draft (`U=<n> B=<n>`) as invertible — U fingerprints the mounted dataset
            # (5193 is only ever the real intercode_bigcode_combined_12k pool after this repo's
            # exact filter) and B, combined with the published _SELF_GEN_FLOOR_FRAC=0.25 constant
            # above, recovers I_sg = B/0.75 exactly whenever B<U (the common case). 'blend=on/off'
            # only says whether any bigcode rows were carved into intercode's target this round;
            # it cannot be run backwards into I_sg, U, or B.
            print(f'[sft_env_config] !!! blend: pool={_pool_flag} blend={"on" if _bigcode_rows > 0 else "off"}', flush=True)
        for env in env_names:
            target = _env_targets[env]
            num_games = _samples_to_games(env, target)
            per_env_path = f'{combined_dataset_path}_{env}'
            per_env_paths.append(per_env_path)
            generate_cmds.append(_build_generate_cmd(env, per_env_path, num_games, target, hours, _gen_seconds, model_architecture))
        parallel_block = ' & '.join((f'({cmd})' for cmd in generate_cmds))
        parallel_block += ' & wait'
        envs_csv = ','.join(env_names)
        if 'intercode' in env_names:
            # Absolute-count path (F2): --intercode-bigcode-rows carries B, --intercode-bigcode-frac
            # is pinned to 0 so combine_datasets' legacy frac math never double-applies.
            # E14 L4 follow-up: the bigcode blend (_blend_bigcode_bash, combine_datasets.py) also
            # calls envs.mastering_intercode.build_examples, so it must carry the SAME
            # --history-thoughts gate as the self-gen generate_cmd above, on the SAME
            # model_architecture test — otherwise the blend (measured ~5,193 of ~7,400 intercode
            # rows at the R1 2-GPU cell, the MAJORITY of intercode volume there) would keep
            # re-teaching Thought lines on Llama at far higher volume than the self-gen fix removes.
            _history_thoughts = 'off' if model_architecture.strip().lower() == 'llamaforcausallm' else 'on'
            combine_cmd = f'python combine_datasets.py --task_id {task_id} --envs {envs_csv} --output {combined_dataset_path} --input-dir {os.path.dirname(combined_dataset_path)} --include-miner-datasets --intercode-bigcode-frac 0 --intercode-bigcode-rows {_bigcode_rows} --history-thoughts {_history_thoughts}'
        else:
            # No intercode in this round's env set: byte-identical to the pre-E8 command (frac 0.0,
            # no --intercode-bigcode-rows flag at all).
            combine_cmd = f'python combine_datasets.py --task_id {task_id} --envs {envs_csv} --output {combined_dataset_path} --input-dir {os.path.dirname(combined_dataset_path)} --include-miner-datasets --intercode-bigcode-frac 0.0'
        full_generate_cmd = f'{parallel_block} && {combine_cmd}'
    run_config = {'epoch_num': get_adaptive_epochs(train_info['hours_to_complete']), 'batch_size': config['batch_size'], 'learning_rate': config['lr'], 'min_lr_rate': 0.25, 'use_liger_kernel': get_use_liger(model_architecture), 'optimizer': 'paged_adamw_8bit', 'use_lora': config.get('use_lora', False), 'disable_fa': disable_flash_attention(model_architecture, model_name), 'packing': 'False', 'gpu_nums': max(1, min(get_gpu_count(), config['gpu_count'])), 'output_dir': train_info['output_dir'], 'request_path': train_info['request_path'], 'distributed': config.get('distributed', 'ddp'), 'gradient_checkpointing': 'True', 'gradient_accumulation_steps': 4, 'max_seq_length': config.get('max_seq_length', 4096), 'max_steps': -1}
    if _5_9b_bucket_arm and run_config['gpu_nums'] >= 2:
        # E9/T3: raise the per-device micro-batch for the WHOLE 5_9_b bucket (Llama-3.1-8B AND
        # Qwen2.5-7B — see MICROBATCH_5_9B above) on 2+ GPUs. Set BEFORE the swe_infinite cap below
        # so a swe_infinite round still clamps down to 2 (memory safety at max_seq_length 8192
        # wins over this raise) for either architecture.
        # _target_effective_batch (64, unchanged) then yields ga=int(64/32)=2 at 2 GPU (eff 64,
        # vs 60 today) and ga=1 at 4 GPU (eff 64, vs 48 today) — see the eff-batch note below.
        run_config['batch_size'] = MICROBATCH_5_9B
    if 'swe_infinite' in env_names:
        # Under our chat template the SWE-ZERO trajectories run to a median of 5,502 tokens and a
        # p90 of 8,009, while every bucket above caps the window at 4,096: almost every swe row was
        # being cut from the FRONT, and the part that goes first is the issue statement the whole
        # transcript is about. 8,192 covers the p90. The gate is the env set, not the bucket table,
        # so a run without swe produces exactly the command it produced before.
        # Doubling the window doubles activation memory per sample, so the micro-batch is
        # capped at 2 for this run only; effective batch is preserved the same way every other
        # env already does it a few lines down — via gradient_accumulation_steps, not by
        # touching _target_effective_batch or the per-bucket LR.
        run_config['max_seq_length'] = 8192
        run_config['batch_size'] = min(run_config['batch_size'], 2)
        # T2 (2026-09-14, Audit-config-coherence.md §4 last row / §5.3 / §7 T2): get_adaptive_epochs
        # picks epoch_num=3 for boss-t3 (hours_to_complete=2.5 -> get_adaptive_epochs above) with NO knowledge that the
        # 8192 window above just tripled the per-row token cost (median 5502 tok/row measured, vs
        # 4096-capped rows elsewhere) and that swe_infinite is the only num_envs==1 path (no
        # combine_datasets row-floor to pad an under-filled generation — see the under-fill note in
        # §5.3). Measured (n=21 converted rows, local fixture, Qwen3 tokenizer, HF_HUB_OFFLINE=1):
        # p50=4287 p90=7537 mean=4759 tok/row, none over the 8192 cap in this sample; consistent
        # order of magnitude with the corpus-wide median/p90 cited above. At the audited swe tok/s
        # (sel J, 4x H100, 23520 tok/s total) 3 epochs need 145-242 min against a ~133 min budget
        # (matrix §4 last row) -- the anneal cannot finish. Forcing epoch_num=2 here cuts the planned
        # token volume by 1/3: at the historically-REALIZED row count (10534, the measured 60%
        # under-fill) this fits inside budget with margin (need ~97-123 min, epoch reaches the full
        # 2.0 / LR floor 25%); at the INTERPOLATED target row count (17600, never actually delivered
        # in production) it is still marginal (~1.65 epoch / ~31% LR) -- flagged, not silently
        # rounded up, because closing that residual gap is a row-target change (T6-shaped), a
        # DIFFERENT technique, and is deliberately not bundled into this one-line epoch cap.
        # Gated on env_names (not the bucket table) so every non-swe cell in get_training_json is
        # byte-identical to before this arm — and on num_envs == 1, because swe_infinite only ever
        # arrives solo (boss task-3: 2.5h, 4xH100; FORCED_BOSS_ENVIRONMENT is excluded from every
        # other draw). If that validator rule is ever relaxed, a 3.0h mixed round keeps its 3 epochs.
        if num_envs == 1:
            run_config['epoch_num'] = 2
    data_per_step = run_config['batch_size'] * run_config['gpu_nums']
    # Target effective batch 64 — (1 if data_per_step>=64 else int(64/data_per_step)) AND the
    # per-model LRs paired with it (3.571e-5/4.302e-5 were Loki-tuned at eff-batch ~64). The old
    # 128 doubled eff-batch → those LRs were effectively halved → under-trained in the 1.5h
    # budget. Lower eff-batch = ~2x optimizer steps for the same wall-clock (same micro-batch count).
    # Validator now allocates GPUs per ROUND, not per model: R1 (2 envs, <=4B) gets a single
    # H100, so _hw_scale lands at 0.275 and the round is data-bound, not clock-bound — a 1.5h
    # window finishes its epochs early with the GPU idle. Effective batch is the one knob that
    # buys optimizer steps out of a FIXED dataset (same FLOPs, more frequent updates): at 1 GPU
    # this takes a 2-env round from ~267 steps to ~712.
    # Multi-GPU rounds keep 64 so the Loki-tuned per-bucket LRs, which
    # were fitted at that effective batch, stay paired with it.
    _target_effective_batch = 24 if run_config['gpu_nums'] == 1 else 64
    run_config['gradient_accumulation_steps'] = 1 if data_per_step >= _target_effective_batch else max(1, int(_target_effective_batch / data_per_step))
    # ⚠ DEAD SIGNAL — verified 2026-08-05 against validator b026da0. This never evaluates True in a
    # real tournament, so `learning_rate *= 0.5` and `lora_dropout = 0.1` below have NEVER executed.
    # Three INDEPENDENT reasons, any one of which is sufficient:
    #   1. At our model sizes use_lora=False (see the bucket table at the top of this file), so we
    #      upload FULL WEIGHTS. A previous round's repo has no adapter_config.json to find.
    #   2. /cache is mounted READ-ONLY into the trainer (runtime.py mode:"ro"), so nothing of ours
    #      could create one either.
    #   3. Even for a LoRA upload, the validator's downloader merges the adapter and deletes it
    #      before our container starts ("the adapter files are removed so downstream code sees a
    #      normal model"), and uploader.py always writes base_model_name_or_path, so the merge
    #      cannot early-return.
    # The `else` branch IS live: lora_dropout = 0.0 is what actually ships whenever use_lora is True.
    #
    # The SAME is true of `_is_prior_repo(...)` a few lines below (epoch_num = 1), for a DIFFERENT
    # reason: MODEL_PREP_ENABLED_ENV is True, so baseline_stats is non-None, so the validator passes
    # `--model <16-char salted SHA256>` instead of the repo id. model_name AND model_path are that
    # hash (train_paths' '/'->'--' mangling is a no-op on a hash), so the
    # 'gradients-io-tournaments/tournament-' substring test can never match. ⇒ NO continuation
    # tuning of any kind currently fires.
    #
    # --- original note, kept verbatim as history (E7 is the eval this note asked for) ---
    # A signal that WOULD work, if we ever want one: num_envs = len(environment_names) is 2 in R1
    # and 4/6 in R2+ (ENV_ENVS_PER_ROUND_MULTIPLIER = 2), and we already receive it.
    # ⛔ DO NOT WIRE IT WITHOUT AN EVAL. Activating this branch would switch on a 0.5x LR cut that
    # has never run in production, inside a 1.5h time-boxed window where under-training is a real
    # cost. That is an experiment, not a bug fix — gate it on a local student head-to-head first.
    # HONESTY CAVEAT: reason 3 is inferred from source about a PRE-BUILT image
    # (gradientsio/trainer-downloader:latest, not built from b026da0), and the merge only entered
    # that source on 2026-07-09, so a stale deployed image could still leave an adapter in place.
    # Reasons 1 and 2 do not depend on the downloader at all, which is why the conclusion holds.
    # --- end original note ---
    #
    # E7 NOTE ON THE HEADER ABOVE: "DEAD SIGNAL ... never evaluates True" describes the old
    # adapter_config.json PROBE only (true before E7). With E7 the structural signal below makes
    # `_is_continuation` True in R2+/boss task 1, so `lora_dropout = 0.1` now executes there; the
    # LR is still NOT halved (withdrawn).
    # E7 — WIRED via the structural signal this comment already named: num_envs =
    # len(environment_names) is 2 in R1 (DEFAULT, task_creator.py's
    # _create_environment_group_tasks, round_number==1) and 4 in R2+ / 6 in the boss round's task 1
    # (CONTINUATION: round_number>1, or _create_environment_boss_round_tasks' boss_task_configs[0]).
    # We already receive num_envs (len(env_names) above) and hours_to_complete, so no new field is
    # needed and none exists to ask for (verified against text_trainer.py's argparse: no
    # training_start_point/continuation flag is ever passed down to get_training_json's train_info).
    # The two shapes this must NOT match: boss task 2 (FROM_SCRATCH) reuses num_envs=6 but with
    # hours=3.0 (_is_long_tier, ENV_TRAINING_HOURS_BOSS_ROUND_FROM_SCRATCH), and boss task 3
    # (PREVIOUS_WINNER) forces a single env (num_envs=1, FORCED_BOSS_ENVIRONMENT=swe_infinite) — both
    # excluded by requiring num_envs>=_CONTINUATION_MIN_ENVS AND NOT _is_long_tier(hours).
    # ⛔ THIS ONLY WIRES DETECTION, NOT THE LR HALVING — that branch was already withdrawn above
    # (Ibrahim et al. 2024) and stays withdrawn; only `lora_dropout` (0.1 vs 0.0) reacts to
    # `_is_continuation` below. Read the "DO NOT WIRE IT WITHOUT AN EVAL" history above as still
    # binding for any FUTURE branch added under this flag (e.g. a revived LR cut) — this patch is
    # explicitly the eval-enabling arm (E7 in experiments/e7/README.md), not a claim that
    # lora_dropout=0.1 has been validated yet.
    #
    # The old adapter_config.json probe is kept as a secondary OR-branch, harmless and evidenced
    # dead (reasons 1-3 above) under the current downloader; if a future image ever leaves an
    # adapter in place, it fires too instead of silently doing nothing.
    _CONTINUATION_MIN_ENVS = 4
    _is_continuation = (num_envs >= _CONTINUATION_MIN_ENVS and not _is_long_tier(hours)) or (
        os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'adapter_config.json')))
    run_config['learning_rate'] *= train_info.get('reg_ratio', 1.0)
    if _is_continuation:
        # The 0.5x halving is WITHDRAWN, following Ibrahim et al. 2024 (arXiv:2403.08763). The
        # continual-pretraining recipe that won there is a FULL re-warm to eta_max followed by a
        # re-decay, not a trimmed peak. Their finding: "higher values of eta_max lead to more
        # forgetting and more adaptation while the opposite is true for lower values". Here we WANT
        # adaptation, since each round changes the env set, and forgetting is not punished because
        # the eval only scores the env of the round being played.
        # E7: this branch is now LIVE for R2+/boss task 1 via the structural signal above (it was
        # dead before E7: the adapter probe never fires). The only effect is lora_dropout 0.1 vs
        # 0.0 — the LR is NOT halved. This arm exists to be A/B-evaluated (experiments/e7); it is
        # not yet validated, per the "DO NOT WIRE IT WITHOUT AN EVAL" note above.
        run_config['lora_dropout'] = 0.1
    else:
        run_config['lora_dropout'] = 0.0
    if _is_prior_repo(model_name) or _is_prior_repo(model_path):
        run_config['epoch_num'] = 1
    if run_config['use_lora']:
        run_config['lora_r'] = 128
        run_config['lora_alpha'] = 512
    run_cmd = _build_run_cmd(run_config, model_path, combined_dataset_path)
    _first_env_save = GAME_CONFIGS.get(env_names[0], GAME_CONFIGS['liars_dice'])['min_periodic_save']
    train_request = deepcopy(train_info)
    train_request['dataset_path'] = combined_dataset_path
    train_request['save_before_remaining_time'] = 8
    train_request['adjust_batch_size'] = False
    train_request['periodic_save_steps'] = _first_env_save
    train_request['checking_step'] = 70
    train_request['min_steps'] = max(int(train_info['hours_to_complete'] * 70), train_info.get('min_steps', 100))
    return {'train_request': train_request, 'run_cmd': run_cmd, 'generate_cmd': full_generate_cmd}
