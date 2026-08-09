# syntax=docker/dockerfile:1
#
# Production image for the eMPI FastAPI service (src/api/). Two stages:
#   builder — installs dependencies into a throwaway venv with build tools.
#   runtime — copies just that venv + src/ onto a minimal base, non-root.
#
# `data/`, `models/`, `logs/` are runtime state (see .gitignore — mostly
# gitignored on purpose), not baked into the image: mount volumes over them.
# `data/empi.db` and `data/local_index/` in particular need a persistent
# volume, or every container restart starts from an empty resolved output.
#
# Multi-worker note: `src/api/jobs.py`'s in-flight run/score registries are
# in-process dicts — fine for a single replica, but `--workers > 1` (or
# multiple container replicas behind a load balancer) would need a shared
# store instead (e.g. Redis) for run-status polling to work correctly. Not
# addressed here; single-replica deployment is the supported target today.

# ---- Builder ----
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY requirements.txt .

# `postal` (libpostal's Python binding) needs the native libpostal library
# plus ~2GB of data files to build/run — not a fit for a lean image. The
# cleaning pipeline already tolerates its absence (Address_normalized stays
# NaN — see src/preprocessing/transformations.py's try/except around the
# import), so it's excluded here rather than installed.
RUN grep -vx postal requirements.txt > requirements.docker.txt \
    && uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r requirements.docker.txt

# ---- Runtime ----
FROM python:3.13-slim AS runtime

# libgomp1 is the OpenMP runtime LightGBM's native lib dlopen()s at import.
# The slim base omits it, and the wheel doesn't vendor it — without this,
# `import lightgbm` raises OSError: libgomp.so.1, which surfaces as a failed
# run the first time Stage 4.25/4.5 unpickles its model (registry
# load_model_artifact -> joblib.load), not at build or startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system empi \
    && useradd --system --gid empi --create-home --home-dir /home/empi empi

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=empi:empi src ./src
# scripts/init_db.py in particular is a required ops step, not optional --
# see this image's docker-compose.yml usage comment ("docker compose exec
# empi-service python scripts/init_db.py"). scripts/research/ is excluded
# via .dockerignore (ad-hoc analysis, not part of the ops surface).
COPY --chown=empi:empi scripts ./scripts

# Runtime-populated directories, empty at build time — see module docstring.
# Mirrors src/config.py's Settings.ensure_dirs() (the actual runtime source
# of truth, called from main.py's lifespan and pipeline.py -- this list is
# just so the image isn't misleading documentation, not load-bearing).
RUN mkdir -p \
        data/raw data/processed data/blocking data/auto_merge data/non_matches \
        data/no_match data/clusters data/runs data/evaluations data/fs_output \
        data/gate_output data/ml_output data/local_index \
        models/fs models/nonmatch_gate models/ml logs \
    && chown -R empi:empi /app

USER empi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
