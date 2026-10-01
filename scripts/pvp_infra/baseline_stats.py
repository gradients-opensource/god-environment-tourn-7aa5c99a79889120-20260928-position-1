import json
import logging
import os
from typing import Any
logger = logging.getLogger(__name__)

def load_baseline_stats() -> 'dict[str, Any] | None':
    path = os.environ.get('BASELINE_STATS_PATH', '').strip()
    if path:
        try:
            with open(path) as f:
                stats = json.load(f)
            logger.info('BASELINE_STATS loaded from file %s: task_type=%s, keys=%s', path, stats.get('task_type'), list(stats.keys()))
            return stats
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning('BASELINE_STATS_PATH=%s unreadable/invalid: %s', path, exc)
    raw = os.environ.get('BASELINE_STATS', '')
    if not raw:
        return None
    try:
        stats = json.loads(raw)
        logger.info('BASELINE_STATS loaded from env var: task_type=%s, keys=%s', stats.get('task_type'), list(stats.keys()))
        return stats
    except json.JSONDecodeError as exc:
        logger.warning('BASELINE_STATS env var invalid JSON: %s', exc)
        return None

def get_init_loss(stats: 'dict | None') -> 'float | None':
    if not stats:
        return None
    return stats.get('training', {}).get('init_loss')

def get_env_score_stats(stats: 'dict | None', environment_name: str) -> 'dict | None':
    if not stats:
        return None
    return stats.get('env_stats', {}).get(environment_name)
