FROM python:3.10-slim

WORKDIR /workspace

# torch + peft required for _detect_and_merge_lora (R2+ continuation chain).
# Validator's trainer_downloader has merge logic; miner must mirror for chain
# to work end-to-end. Without merge, R2 would crash loading adapter as base.
RUN pip install --no-cache-dir \
    huggingface_hub \
    aiohttp \
    pydantic \
    transformers \
    torch \
    peft

COPY scripts/ scripts/

ENV PYTHONPATH=/workspace/scripts

# Quiet mode is already the code default, so this changes nothing functionally. It is here so
# the image states its own intent: whoever reads this file can see the run is quiet without
# having to go and find the default in Python. Set to 0 for a noisy local build.
ENV TRAIN_QUIET=1

ENTRYPOINT ["python", "scripts/trainer_downloader.py"]