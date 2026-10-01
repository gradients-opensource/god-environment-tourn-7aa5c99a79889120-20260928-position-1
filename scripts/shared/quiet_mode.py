"""Quiet mode: keep failures loud, keep the recipe quiet.

Import as early as possible in an entry point: several controls below are environment defaults the
heavy libraries read once, at their own import time.

THE ARCHITECTURE THIS RELIES ON
    Child processes are launched by `text_trainer.run_cmd_with_log` with `stderr=subprocess.STDOUT`,
    so a child's stderr is merged into its stdout, relayed through the parent's `print`, and
    therefore passes through this filter. That is what makes it safe to leave the logging module
    alone: a child's log lines are filtered one level up. Only the top-level process's own stderr
    reaches the outside unfiltered, which is where the crash guard below matters.

WHAT SURVIVES
    Failures, always. A line is kept when it matches one of the markers below, and kept lines are
    routed to stderr rather than dropped. Many failures in this tree only print and carry on without
    raising -- an env skipped, a model that never reached the submission directory, an adapter merge
    quietly passed over -- so a crash guard alone would never see them.

    The learning-rate line is kept deliberately: the LR curve is the only honest evidence that the
    anneal ran to completion, which is the thing several rounds were lost on.

WHAT GETS REDACTED RATHER THAN PASSED
    Our own tagged lines. They read like diagnostics but they spell out the recipe -- which
    environments, which datasets, how many samples, what budget. A word like "skipped" or "0 failed"
    inside one used to promote the whole line into the log by accident. When such a line does signal
    a problem, only the tag and the triggering word are emitted: the alarm survives, the content does
    not. A line that genuinely must be read in full carries the loud marker.

WHY IT DEFAULTS ON
    TRAIN_QUIET=1 means on, and UNSET means on too. That default is the whole design: the tournament
    passes no environment variables to a miner, so an unset variable IS the tournament case and it
    has to be the quiet one. A lever that needed setting to switch on would be dead precisely where
    it matters. TRAIN_QUIET=0, or TRAIN_VERBOSE=1, turns it off for a local run.

THREE THINGS THAT MAKE IT SAFE TO ENABLE -- do not remove any one of them
    * The crash guard writes tracebacks to stderr AND to the original stdout, so a harness that
      captures only one of the two still sees the failure. An early version of this feature was
      abandoned precisely because it hid a tournament traceback.
    * There is no global `logging.disable()`, and the root logger is left alone. Our own loggers stay
      alive, and a caller that configures logging later is not silently overridden.
    * Warning-shaped lines are REDIRECTED, not discarded.
"""

import builtins
import faulthandler
import functools
import logging
import os
import sys
import traceback
import warnings

_TRUTHY = ('1', 'true', 'yes', 'on')
_FALSY = ('0', 'false', 'no', 'off')

_real_print = builtins.print
_STDOUT_AT_IMPORT = sys.stdout   # the stream the harness captures; see _quiet_print

# Redirected to stderr, not dropped. Matched case-insensitively against the first argument.
#   learning_rate / save the model at step -- the LR curve is the only legitimate evidence that the
#     anneal ran. Note the current path does not actually reach here: with `disable_tqdm` at its
#     default of False, HF uses ProgressCallback, whose `on_log` writes via `tqdm.write()` straight
#     to stdout without passing through this shim. The markers are kept for the day `disable_tqdm`
#     is set, which switches HF to PrinterCallback and its plain `print`.
#   failure words -- most failures in this tree print and continue rather than raise, so nothing
#     else would catch them.
_KEEP_MARKERS = (
    'learning_rate', 'save the model at step',
    'warn', 'fail', 'error', 'fatal', 'traceback', 'exception', 'abort', 'critical',
    'skip', 'fallback', 'falling back', 'retry', 'retrying', '!!!',
    'oom', 'out of memory', 'cannot', 'could not', 'unable', 'missing', 'invalid',
    'corrupt', 'denied', 'refused', 'timeout', 'timed out', 'not found', 'no such',
    'unsupported', 'no usable', 'creating dummy', 'crash', 'killed',
    # Traceback continuation frames. Without this only the header and the final "XxxError: ..."
    # line survive a relayed child crash, and the stack in between -- the part that says WHERE --
    # is stripped from both the console and the log file.
    'file "',
)

# Our own tagged lines. Kept, but redacted -- see the module docstring.
_OWN_TAGS = (
    '[baseline_stats]', '[parse_warn]', '[check]', '[combine]', '[downloader]', '[eval]',
    '[eval-json]', '[eval-json-opponent]', '[intercode]', '[pvp-tc]', '[sft_env_config]',
    '[swe]', '[text_trainer]',
)

# Put this in a line that must reach the log intact despite carrying one of our tags.
_LOUD = '!!!'


def _redact(text: str, marker: str) -> str:
    """Reduce one of our tagged lines to its tag and the word that triggered the keep."""
    tag = text[:text.index(']') + 1] if ']' in text[:40] else '[?]'
    return f'{tag} {marker}'


def _marker_for(text: str):
    low = text.lower()
    return next((m for m in _KEEP_MARKERS if m in low), None)


def _is_own_tag(text: str) -> bool:
    return text.lstrip().lower().startswith(_OWN_TAGS)


def _quiet_print(*args, **kwargs):
    if kwargs.get('file', None) is sys.stderr:
        return _real_print(*args, **kwargs)
    # Only filter what is headed for the process's real stdout. A print aimed at any other stream
    # -- an explicit file=, or a sys.stdout someone swapped in (contextlib.redirect_stdout,
    # dill.temp.capture) -- is that caller capturing output for itself, and must receive it intact.
    # dill.detect.nestedglobals captures dis.dis(func) this way to find a function's globals: when
    # this shim dropped those lines the capture came back empty, and the lines that matched a marker
    # ("ValueError", "skip") were rerouted to stderr, publishing a disassembly of our code in the log.
    dest = kwargs.get('file', None) or sys.stdout
    if dest is not sys.__stdout__ and dest is not _STDOUT_AT_IMPORT:
        return _real_print(*args, **kwargs)
    try:
        if args:
            head = str(args[0])
            marker = _marker_for(head)
            if marker is None:
                return None
            kwargs['file'] = sys.stderr
            if _LOUD not in head and _is_own_tag(head):
                return _real_print(_redact(head.lstrip(), marker), **kwargs)
            return _real_print(*args, **kwargs)
    except Exception:
        pass
    return None


# numba's `@infer_global(print)` asserts that
# `getattr(sys.modules[print.__module__], print.__name__) is print`. Without this disguise numba --
# and therefore axolotl -- fails to import.
try:
    functools.update_wrapper(_quiet_print, _real_print)
except Exception:
    pass


def _is_quiet() -> bool:
    """TRAIN_QUIET=1 means quiet is on. Unset means the same thing.

    One variable, read the obvious way round, because the previous three-variable version needed a
    double negative to state its default and that is how a lever ends up doing the opposite of what
    someone assumed. The DEFAULT is what matters: the tournament passes no environment variables to
    a miner, so an unset variable is the tournament case and it must be the quiet one. Setting it in
    the dockerfile changes nothing functionally -- it makes the intent visible to whoever reads the
    image next.

    TRAIN_VERBOSE=1 is kept as a convenience for turning quiet off in a shell without remembering
    the sense of the other variable.
    """
    if (os.environ.get('TRAIN_VERBOSE') or '').strip().lower() in _TRUTHY:
        return False
    return (os.environ.get('TRAIN_QUIET') or '1').strip().lower() in _TRUTHY


def _silence_libraries() -> None:
    # setdefault so an operator's own export still wins.
    # TRANSFORMERS_VERBOSITY and DATASETS_VERBOSITY are deliberately NOT forced to error: that also
    # discards the training metric lines, which are the evidence the run is progressing at all.
    env_defaults = {
        'TQDM_DISABLE': '1',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'DATASETS_DISABLE_PROGRESS_BARS': '1',
        'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'BITSANDBYTES_NOWELCOME': '1',
        'VLLM_LOGGING_LEVEL': 'ERROR',
        # A warning prints the SOURCE FILE PATH alongside its message, so an otherwise quiet run
        # publishes its directory layout one DeprecationWarning at a time. Inherited by
        # subprocesses, which the in-process filter cannot reach.
        'PYTHONWARNINGS': 'ignore',
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

    # In-process too, because a library calling resetwarnings() would undo the environment default.
    try:
        warnings.simplefilter('ignore')
    except Exception:
        pass

    # No global logging.disable(), and the root logger is left at whatever the caller chose. Only
    # named third-party loggers are lowered.
    for name in ('datasets', 'accelerate', 'deepspeed', 'peft', 'vllm', 'axolotl', 'torch',
                 'huggingface_hub', 'filelock', 'urllib3', 'open_spiel'):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass


def _install_crash_guard() -> None:
    """Make hard failures impossible to lose, whichever stream the harness captures."""
    try:
        faulthandler.enable()
    except Exception:
        pass

    _prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        text = ''.join(traceback.format_exception(exc_type, exc, tb))
        # _real_print bypasses the shim. Write to BOTH streams so a harness capturing only one of
        # them still gets the traceback.
        for stream in (sys.stderr, sys.__stdout__):
            try:
                _real_print(text, file=stream, flush=True)
            except Exception:
                pass
        try:
            _prev_hook(exc_type, exc, tb)
        except Exception:
            pass

    try:
        sys.excepthook = _hook
    except Exception:
        pass


def _silence_print() -> None:
    builtins.print = _quiet_print


def enable() -> bool:
    """Apply quiet mode. Returns whether it was applied. Safe to call more than once."""
    if not _is_quiet():
        return False
    for step in (_silence_libraries, _install_crash_guard, _silence_print):
        try:
            step()
        except Exception:
            pass
    return True


def is_enabled() -> bool:
    return builtins.print is _quiet_print


def relay(line: str):
    """What a relayed child line should become in the log, or None to drop it.

    The subprocess relay writes each child line to both the container stdout and a local log file.
    Routing both through this keeps one definition rather than two that can drift, and stops the
    file from holding verbatim what the filter had just removed -- the file outlives the run.
    """
    if not is_enabled():
        return line
    marker = _marker_for(line)
    if marker is None:
        return None
    stripped = line.lstrip()
    if _LOUD not in stripped and _is_own_tag(stripped):
        return _redact(stripped, marker) + '\n'
    return line


# Applied on import so an entry point only has to import this module. Every step is guarded, so a
# failure here degrades to a noisy run rather than a dead one: losing logs must never lose a run.
enable()
