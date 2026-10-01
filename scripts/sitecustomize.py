"""Applies quiet mode to every Python process in the image.

The interpreter imports `sitecustomize` automatically at startup when it is importable, so placing
this on PYTHONPATH covers processes that never import it themselves. That is what makes it the right
mechanism here: the training run is not one process. The config emits five different command shapes
-- three generator subprocesses (`python -m envs.generate_pvp_toolcalling`, `envs.mastering_intercode`,
`envs.mastering_swe`), the dataset combiner, and the trainer under torchrun -- and the generators run
for the entire generation window. Wiring quiet mode into each one by hand would mean repeating the
same few lines in every entry point, for no gain over doing it once here.

The interpreter reaches this file only when its directory is on PYTHONPATH at startup, not merely
when it is the working directory: `site` runs before the working directory joins `sys.path`. Both
dockerfiles set PYTHONPATH accordingly. If that ever stops being true, quiet mode simply does not
apply and the run is noisy -- never broken.

Everything here is guarded. A failure in this file would otherwise fail interpreter startup itself,
taking down every process in the image; a submission must never be lost over log formatting.
"""

try:
    import shared.quiet_mode  # noqa: F401
except Exception:
    pass
