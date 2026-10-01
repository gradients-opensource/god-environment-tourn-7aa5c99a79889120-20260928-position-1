# Compliant with the env-tournament rules (docs/miner.md): NO external dataset is
# bundled. InterCode training data is SELF-GENERATED at train-time inside the
# container (scripts/envs/selfgen_bash.py) and verified against a real shell —
# never the NL2Bash benchmark instances, never a dataset file baked into the image.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    vim \
    zip \
    tmux \
    iotop \
    nvtop \
    bmon \
    wget \
    nano \
    zsh \
    htop \
    redis-server \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts

# Puts scripts/sitecustomize.py in reach of the interpreter's startup, which is what applies quiet
# mode to EVERY process in the image -- the generator subprocesses and the torchrun workers included,
# none of which import it themselves. The working directory alone is not enough: `site` runs before
# the working directory joins sys.path, so this must be PYTHONPATH.
ENV PYTHONPATH=/workspace/scripts

# Quiet mode is already the code default, so this changes nothing functionally. It is here so
# the image states its own intent: whoever reads this file can see the run is quiet without
# having to go and find the default in Python. Set to 0 for a noisy local build.
ENV TRAIN_QUIET=1

COPY scripts/grpo_requirements.txt /workspace/scripts/grpo_requirements.txt
RUN python -m venv /workspace/.grpo_env
RUN bash -c "source /workspace/.grpo_env/bin/activate && \
    pip install uv && \
    uv pip install --no-build-isolation -r /workspace/scripts/grpo_requirements.txt && \
    uv pip install --no-build-isolation flash-attn==2.8.3 && \
    deactivate"

COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_text_trainer.sh

ENTRYPOINT ["./run_text_trainer.sh"]
