#!/bin/bash
set -e

# stdout only: a failure still reports on stderr, and `set -e` still stops the run.
redis-server --daemonize yes > /dev/null
sleep 10

echo "*****Running text trainer"
source /workspace/.grpo_env/bin/activate
python3 /workspace/scripts/text_trainer.py "$@"
deactivate