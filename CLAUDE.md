# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FastAPI server that serves biomedical NER models. Two model families behind one
`/annotate/` endpoint:

- **token_classification** — a NeMo `TokenClassificationModel` (`.nemo` checkpoint) that tags
  entities and returns PubAnnotator-format JSON (`{text, denotations[]}`).
- **sapbert** — a HuggingFace SapBERT encoder that embeds the query text and does vector
  nearest-neighbour lookup against a prebuilt index of biomedical concept embeddings,
  returning `{score, name, curie, category}` hits. This is entity *linking*, not tagging.

Deployed at RENCI (`med-nemo.apps.renci.org`, `med-nemo-sapbert.apps.renci.org`) on
Kubernetes, CPU-only in practice.

## Commands

```bash
# Serve (CONFIG_PATH defaults to ./config.yaml)
CONFIG_PATH=./config.yaml uvicorn src.server:app --port 8080

# Tests. pytest.ini puts `.` and `src` on sys.path — unit tests import `ModelSingleton`
# unqualified, server code imports `src.ModelSingleton`. Both work only via pytest.
pytest tests/TokenClassificationModelWrapper       # unit, no model download
pytest -s tests/integration                         # hits LIVE deployments, see below
pytest tests/integration/test_api.py::test_models   # single test

# Offline batch annotation over a JSONL file (each line needs a "text" key)
python src/cli.py -c config.yaml annotate -i in.jsonl -o out.jsonl -m token_classification

# Legacy single-shot index build from .npy + csv (superseded by build-db/, see below)
python src/cli.py -c config.yaml index
```

Integration tests point at production by default. Override with `NEMOSERVE_URL`,
`NEMOSAPBERT_URL`, `MODELS_EXPECTED` (pipe-separated) before running them against a local
instance.

Serving image builds from `Dockerfile` on `nvcr.io/nvidia/nemo` — it `git clone`s the repo at
`--build-arg VERSION=<tag>` rather than copying the working tree, so local edits are not in
the image. Released on GitHub release via `.github/workflows/docker-image.yml`.

## Architecture

**Config drives model loading.** `config.yaml` maps a model name to a `path` and a `class`
string. `init_models()` looks that string up in `ModelFactory.model_classes` — a new wrapper
must be registered in that dict or startup raises. Wrappers subclass `ModelWrapper` and
implement `async __call__(query_text, count, **kwargs)`; the factory treats instances as
callables. `init_models()` special-cases the literal model name `"sapbert"` to pass
`connection_config`/`backend` as extra params, so a SapBERT model under a different config key
will load without its storage client.

**Storage backends are duck-typed, not an interface.** `SAPQdrant` (`src/utils/SAPQdrant.py`)
and `RedisMemory` (`src/utils/SAPRedis.py`) both expose
`search / create_index / delete_index / populate_index / refresh_index`; `config.yaml`'s
`storage:` key picks one in both `ModelSingleton` and `src/index.py`. Qdrant is the one in use.

**Index building lives in `build-db/`, separate from the server.** `build-db/sapbert-builder-2.py`
is a standalone script (its own `requirements.txt`, `Dockerfile`, and `k8s-job.yaml`) that
streams Babel embedding dumps (`<folder>/embedding/*.npy` + `<folder>/metadata/name_ids*` +
`metadata/id_types.csv`) into Qdrant. It has its **own** `SAPQdrant` class, unrelated to the
serving one. Key behaviours: resumable via `--progress-file` JSON, SIGTERM-safe, point ids are
sequential integers so `--id-offset` must clear existing ids or upserts silently overwrite
(there is a guard that refuses to start otherwise), `--exclude-prefixes` drops curies by
namespace. `src/index.py` is the older in-repo path for the same job and is not what production
collections were built with.

## Perf invariants — do not undo these

The most recent work was a measured perf overhaul; comments in the code carry the numbers.
Changes that look like cleanup here are regressions:

- `_cap_torch_threads()` in `ModelSingleton.py` reads the cgroup CPU quota because
  `os.cpu_count()` reports host cores. 48 OpenMP threads on a 4-CPU pod = 880ms/forward pass
  vs 56ms. `OMP_NUM_THREADS` in the pod spec is the primary fix; this is the backstop.
- `SapbertModelWrapper._embed_slots` bounds starlette's 40-worker threadpool to
  `cpus / torch_threads`. Unbounded concurrency re-creates the same oversubscription.
- Qdrant client uses **gRPC** (`prefer_grpc=True`): REST spends ~4.7ms/hit in pydantic
  deserialisation (53ms vs 4.6ms for the same search). The `GRPC_OPTIONS` keepalive dict plus
  `_retrying()` exist because a ClusterIP reaps the idle HTTP/2 connection and the first
  request after a quiet period fails.
- `SEARCH_PARAMS` rescoring recovers recall@10 from 94.7% to 96.7% under int8 quantization for
  no measurable latency.
- In `build-db` `create_index`, `HnswConfigDiff(on_disk=False)` is deliberate: storage is NetApp
  NFS, and on-disk HNSW makes every graph hop a network round trip (137.9ms vs 3.0ms p50).

## Known rough edges

- Serving `SAPQdrant.search()` reads `payload["categories"]`, but serving `populate_index()`
  writes `payload["category"]` and `refresh_index()` indexes `category`. Collections built by
  `build-db/` write `categories` and work; collections built by `src/index.py` will KeyError on
  search. The `@TODO` in the file marks it.
- `TokenClassificationModelWrapper._token_chunks` references an undefined `failing_text` in its
  regex loop — that branch (token count >= window_size) raises `NameError`.
- `.bak` files sit next to several edited sources; they are scratch, not a pattern to follow.
