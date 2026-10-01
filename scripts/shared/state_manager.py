import json
import os
import redis
STATE_KEY = 'state'

# Redis is wrapped so it can NEVER be fatal.
# run_text_trainer.sh uses `set -e` with `redis-server --daemonize yes`, and --daemonize returns 0
# IMMEDIATELY, so a daemon that fails to bind a moment later is not caught by `set -e`. get_state()
# at text_trainer.py:379 then raises ConnectionError and the process dies with a non-zero exit
# BEFORE a single second of training. A non-zero exit means the validator never uploads, which
# means a score of 0.
# Di jalur EnvTask, Redis nol nilai fungsional: hasil get_state() langsung ditimpa `state = {}`
# di baris berikutnya, dan konsumen state di customized_trainer digerbangi
# checking_mode 'first_time'/'second_time' sementara EnvTask memaksa 'none'.
# Jadi degradasi ke dict kosong = perilaku identik, tanpa risiko fatal.


def _get_redis_client() -> redis.Redis:
    host = os.getenv('REDIS_HOST', 'localhost')
    port = int(os.getenv('REDIS_PORT', 6379))
    password = os.getenv('REDIS_PASSWORD', None)
    db = int(os.getenv('REDIS_DB', 0))
    return redis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)


def get_state() -> dict:
    try:
        client = _get_redis_client()
        value = client.get(STATE_KEY)
    except Exception:
        return {}
    if value is None:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def set_state(state: dict) -> None:
    try:
        _get_redis_client().set(STATE_KEY, json.dumps(state))
    except Exception:
        pass
