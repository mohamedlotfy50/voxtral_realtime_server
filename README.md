# Voxtral Realtime Server

Realtime **speech-to-text** (Voxtral via vLLM) + **speaker diarization** as two
independent HTTP services, built with a clean hexagonal / feature-modular
architecture in pure OOP (one class per file, no global functions, no global
variables).

---

## Overview

This repository runs **two services** that can be launched together or
independently:

| Service | Entry point | Default port | Description |
|---------|-------------|:------------:|-------------|
| **STT** | `serve_voxtral.py` | `8000` | vLLM-hosted Voxtral-Mini-4B-Realtime realtime transcription via WebSocket. Patches vLLM's `RealtimeConnection` to emit `token_ids` in delta events. |
| **Diarization** | `serve_diarization.py` | `8001` | REST API for per-sentence speaker labeling using pyannote.audio on a GPU-batched cross-session inference engine. |

Both services share a common configuration layer (`ConfigService`) that reads
`config.yaml` (vLLM/STT) and `app/shared/config/diarization.yaml` (diarization).

---

## Project structure

```
app/
├── server.py                      # AppFactory — builds the diarization FastAPI app + lifespan
├── shared/
│   ├── config/
│   │   ├── app_config.py           # AppConfig (aggregate)
│   │   ├── diarization_settings.py # DiarizationSettings (frozen dataclass)
│   │   ├── server_settings.py      # ServerSettings (frozen dataclass)
│   │   ├── voxtral_settings.py     # VoxtralSettings (frozen dataclass + build_argv)
│   │   ├── loader.py               # ConfigService (singleton, yaml loading, overrides)
│   │   └── diarization.yaml        # diarization defaults
│   └── infrastructure/
│       ├── embedder.py             # SpeechEmbedder (ECAPA)
│       ├── embedder_provider.py    # EmbedderProvider (singleton)
│       ├── embedding_batcher.py    # EmbeddingBatcher (continuous-batch worker)
│       ├── batcher_request.py      # BatcherRequest
│       ├── batcher_provider.py     # BatcherProvider (singleton + shutdown)
│       ├── diarization_engine.py   # DiarizationEngine (cross-session worker pool)
│       ├── diarization_engine_provider.py # DiarizationEngineProvider (singleton)
│       ├── engine_request.py       # EngineRequest (queued pipeline run)
│       ├── pyannote_pipeline_provider.py  # PyannotePipelineProvider (per-worker pool)
│       ├── audio_decoder.py        # AudioDecoder (base64 PCM -> float)
│       ├── tf32_patch.py           # Tf32Patch (re-enable TF32 for throughput)
│       └── warning_filter.py       # PyannoteWarningFilter (benign-warning silencer)
├── diarization/
│   ├── domain/                     # Pure logic — no FastAPI; numpy + stdlib (+ torch for audio buffers)
│   │   ├── entities.py             # Speaker
│   │   ├── embedding_store.py      # EmbeddingStore
│   │   ├── reclusterer.py          # Reclusterer (agglomerative)
│   │   ├── incremental_clusterer.py # IncrementalClusterer (stick/switch/recency/new)
│   │   ├── pyannote_labeler.py     # PyannoteLabeler (per-session pyannote-backed labeler)
│   │   ├── segment_registry.py     # SegmentRegistry / SegmentRecord (relabel protocol)
│   │   └── session_audio_buffer.py # SessionAudioBuffer (full-timeline audio buffer)
│   ├── repositories/
│   │   ├── session.py              # Session
│   │   ├── session_not_found.py    # SessionNotFound
│   │   └── session_repository.py   # SessionRepository (thread-safe in-memory)
│   ├── services/
│   │   ├── session_service.py     # SessionService
│   │   ├── label_service.py       # LabelService (decode -> label -> respond)
│   │   └── batch_service.py       # BatchService (offline one-shot diarization)
│   └── controllers/
│       ├── diarization_controller.py # DiarizationController (bound-method routes)
│       ├── label_req.py            # LabelReq schema
│       ├── batch_req.py            # BatchReq schema
│       ├── batch_item.py           # BatchItem schema
│       ├── batch_resp.py           # BatchItemResp schema
│       ├── batch_service_dep.py    # BatchServiceDep (FastAPI dependency)
│       ├── create_session_resp.py  # CreateSessionResp schema
│       ├── delete_session_resp.py   # DeleteSessionResp schema
│       ├── session_service_dep.py  # SessionServiceDep (FastAPI dependency)
│       └── label_service_dep.py    # LabelServiceDep (FastAPI dependency)
└── voxtral/
    └── realtime_patch.py          # RealtimeMonkyPatch (vLLM patch)

config.example.yaml                # vLLM / Voxtral serve config template (committed)
config.yaml                        # your local config (gitignored — copy from example)
serve_voxtral.py                   # Entrypoint: STT server
serve_diarization.py               # Entrypoint: diarization server
requirements.txt
```

### Architecture rules

- **One class per file** — enforced across the entire codebase.
- **No global functions** — every function is a class/instance/staticmethod.
- **No global variables** — constants and loggers are class-level attributes.
  The sole exception is `app = AppFactory.create()` in `server.py`, required by
  uvicorn's `uvicorn app.server:app` interface.
- **Layering**: `controllers → services → domain`; `repositories` is the
  data-access seam; `infrastructure` holds external adapters (torch,
  speechbrain, pyannote). The domain layer is mostly numpy + stdlib
  (`SessionAudioBuffer` uses torch for waveform tensors, and
  `PyannoteLabeler` submits pipeline runs via `DiarizationEngineProvider`).
- **Dependencies** wired via FastAPI `Depends()` with class-based providers.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **torch**: install the build matching your CUDA version, e.g.
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu130
> ```

### 2. Download model weights

The `llm_model/` directory is **not** included in the repo due to size. It is
gitignored — you must download the weights separately:

- **Voxtral-Mini-4B-Realtime-2602** — download from
  [huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
  and place it under `llm_model/Voxtral-Mini-4B-Realtime-2602/` (or any path
  you set in `config.yaml`).
- **speechbrain/spkrec-ecapa-voxceleb** — downloaded automatically on first run
  (cached in `hf_cache/`, also gitignored).

```bash
# Example: download via huggingface-cli
huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --local-dir llm_model/Voxtral-Mini-4B-Realtime-2602
```

### 3. Configure

Copy the example config and fill in your values:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` for the STT server:

```yaml
model_path: <path/to/Voxtral-Mini-4B-Realtime-2602>   # local model directory
served_model_name: Voxtral-Mini-4B-Realtime-2602
api_key: <your-api-key>                                # any secret string for auth
host: 0.0.0.0
port: "8000"
trust_remote_code: true
# Use compilation_config OR enforce_eager (not both). Recipe recommends:
# compilation_config: '{"cudagraph_mode": "PIECEWISE"}'
enforce_eager: true
tensor_parallel_size: "1"
max_model_len: "3584"
max_num_batched_tokens: "4096"
max_num_seqs: "4"
gpu_memory_utilization: "0.90"
attention_backend: FLASH_ATTN
```

Edit `app/shared/config/diarization.yaml` for the diarization server:

```yaml
server:
  host: 0.0.0.0
  port: 8001
  log_level: info

diarization:
  device: cuda:1
  hf_token: null            # prefer the HF_TOKEN env var; never commit a real token
  pyannote_model: pyannote/speaker-diarization-community-1
  sample_rate: 16000
  segmentation_batch_size: 32
  embedding_batch_size: 32
  engine_workers: 2
  tf32: true
  pipeline_every_n: 2
  pipeline_min_audio_s: 2.0
  max_speakers: null
  session_ttl_s: 3600

  # Server-wide pipeline hyperparameter defaults (per-request `params`
  # in /batch items override these):
  pipeline_params:
    clustering:
      threshold: 0.65
```

> The pyannote model requires a HuggingFace token with accepted terms —
> export `HF_TOKEN` (or set `hf_token` in the yaml) before starting the
> diarization server.

### 4. Run

**Both servers** (STT on `:8000`, diarization on `:8001`):

```bash
python serve_voxtral.py      # STT (vLLM Voxtral)
python serve_diarization.py  # diarization (in a separate terminal)
```

**Diarization only** (without vLLM):

```bash
python serve_diarization.py
# or directly via uvicorn:
uvicorn app.server:app --host 0.0.0.0 --port 8001
```

---

## Diarization API

Base URL: `http://localhost:8001`

### Create session

```
POST /v1/diarization/sessions
```

```json
{ "session_id": "ca8e192ce3724359be6ff3935d1f9479" }
```

### Label a segment

```
POST /v1/diarization/sessions/{session_id}/label
```

Request body:

```json
{
  "audio": "<base64-encoded int16 PCM @16kHz mono>",
  "start_s": 0.0,
  "end_s": 2.5,
  "prev_speaker": null
}
```

Response:

```json
{
  "speaker_idx": 0,
  "speaker": "SPEAKER_00",
  "sim": 0.87,
  "num_speakers": 1,
  "method": "stick",
  "consolidated": false,
  "relabel": null
}
```

When `consolidated` is `true`, `relabel` describes label corrections. It has
two parts:

* `mapping` — bulk rename `{old: new}` for confirmed speaker labels that
  moved as a whole (merges/majority-vote target changes). Safe to apply to
  every stored segment: the no-split matching rule guarantees each old
  label maps to exactly one new label.
* `segments` — per-segment corrections for newly confirmed segments
  (matched by `(start_s, end_s)`). A bulk rename cannot express these —
  the same borrowed temporary label may resolve to different real speakers
  once a pipeline pass confirms them.

```json
{
  "relabel": {
    "mapping": { "SPEAKER_02": "SPEAKER_00" },
    "segments": [
      { "start_s": 12.0, "end_s": 14.0, "speaker": "SPEAKER_01" }
    ],
    "num_speakers": 2,
    "consolidation": 16
  }
}
```

Speaker IDs are stable across pipeline passes (no per-pass renumbering):
matched speakers keep their ID, brand-new speakers get fresh IDs.

### Session summary

```
GET /v1/diarization/sessions/{session_id}
```

```json
{
  "session_id": "ca8e...",
  "speakers": [
    { "id": "SPEAKER_00", "talk_time_s": 12.5, "utterances": 8 },
    { "id": "SPEAKER_01", "talk_time_s": 7.2,  "utterances": 5 }
  ],
  "num_speakers": 2
}
```

### Force consolidation

```
POST /v1/diarization/sessions/{session_id}/consolidate
```

Runs a final agglomerative merge of all pending embeddings and returns the
relabel mapping (use at end of stream).

### Delete session

```
DELETE /v1/diarization/sessions/{session_id}
```

### Batch (offline, one-shot)

```
POST /v1/diarization/batch
```

Diarizes multiple full audios in one request. Items are executed on the same
GPU engine as realtime sessions but at **lower priority**, so live sessions
are never delayed by batch jobs. Per-item failures are isolated into that
item's `error` field.

Request body:

```json
{
  "items": [
    { "audio": "<base64-encoded int16 PCM @16kHz mono>", "max_speakers": 2 },
    { "audio": "<base64-encoded int16 PCM @16kHz mono>" }
  ]
}
```

Response:

```json
{
  "results": [
    {
      "turns": [
        { "start_s": 0.5, "end_s": 3.2, "speaker": "SPEAKER_00" },
        { "start_s": 3.8, "end_s": 6.1, "speaker": "SPEAKER_01" }
      ],
      "num_speakers": 2,
      "error": null
    }
  ]
}
```

### Health

```
GET /v1/diarization/health
```

```json
{
  "sessions": 3,
  "engine": {
    "workers": 2,
    "runs": 42,
    "failures": 0,
    "avg_run_s": 1.8,
    "last_run_s": 1.2,
    "last_run_ago_s": 0.4,
    "realtime_queue_depth": 0,
    "offline_queue_depth": 0
  },
  "config": { "device": "cuda:1", "segmentation_batch_size": 32, "..." : "..." }
}
```

---

## How diarization works

1. **pyannote pipeline**: Each session accumulates its full audio timeline
   (speech + silence) in a per-session buffer. Every `pipeline_every_n`
   segments (min `pipeline_min_audio_s` audio), the accumulated waveform is
   run through `pyannote/speaker-diarization-3.1` and utterances are
   relabeled by their dominant speaker turn. `consolidate` forces a final
   pass to absorb the tail.
2. **GPU batching**: The pipeline's native batch knobs
   (`segmentation_batch_size` / `embedding_batch_size`) process segmentation
   windows and speaker-diarization crops in batches per forward pass.
3. **Cross-session engine**: A worker pool (`engine_workers` pipelines,
   one per thread) executes pipeline runs from all sessions and from
   offline `/batch` requests. Realtime requests are drained before offline
   ones; parallel workers overlap CUDA kernels to keep the GPU saturated.
4. **Label stability**: pyannote speaker labels are remapped to
   first-appearance order and matched to prior internal labels via majority
   vote, so speaker indices stay stable across incremental pipeline runs; a
   `relabel` mapping lets clients rewrite previously-printed labels.

---

## Configuration reference

> **Note:** `config.yaml` is gitignored (it contains local paths + secrets).
> Copy `config.example.yaml` to `config.yaml` and fill in your values.

### `config.yaml` (root — vLLM/STT)

| Key | Description |
|-----|-------------|
| `model_path` | Path to the downloaded [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) model directory |
| `served_model_name` | Model name exposed via the API |
| `api_key` | Secret string for authenticated requests (set your own) |
| `host` / `port` | Bind address for the vLLM server |
| `enforce_eager` | Disable CUDA graphs (recommended for realtime) |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `max_model_len` | Maximum model context length |
| `gpu_memory_utilization` | Fraction of GPU memory to use |

### `app/shared/config/diarization.yaml`

| Key | Default | Description |
|-----|:-------:|-------------|
| `server.host` | `0.0.0.0` | Diarization server bind address |
| `server.port` | `8001` | Diarization server port |
| `server.log_level` | `info` | Uvicorn log level |
| `diarization.device` | `cuda:1` | Device for pyannote inference |
| `diarization.hf_token` | `null` | HuggingFace token (prefer the `HF_TOKEN` env var — never commit a real token) |
| `diarization.pyannote_model` | `pyannote/speaker-diarization-3.1` | HF pipeline id (requires accepted terms + token; the shipped yaml uses `pyannote/speaker-diarization-community-1`) |
| `diarization.sample_rate` | `16000` | PCM sample rate clients must send |
| `diarization.segmentation_batch_size` | `32` | Segmentation windows per GPU forward pass |
| `diarization.embedding_batch_size` | `32` | Speaker crops per embedding forward pass |
| `diarization.engine_workers` | `2` | Parallel pipeline workers (each loads its own pipeline) |
| `diarization.tf32` | `true` | Re-enable TF32 matmuls for throughput (set `false` for pyannote's accuracy-first default) |
| `diarization.pipeline_every_n` | `2` | Segments between incremental pipeline runs |
| `diarization.pipeline_min_audio_s` | `2.0` | Min accumulated audio before a pipeline run |
| `diarization.max_speakers` | `null` | Cap speaker count if known |
| `diarization.pipeline_params` | `null` | Server-wide pyannote hyperparameter overrides, e.g. `clustering: {threshold: 0.65}` (per-request `params` in `/batch` items take precedence) |
| `diarization.session_ttl_s` | `3600` | Idle session expiry (seconds) |

### Runtime overrides

Override any setting programmatically before first config access:

```python
from app.shared.config.loader import ConfigService

ConfigService.get().configure(stick_threshold=0.65, port="9000")
```

Or via environment variables pointing at alternative yaml files:

```bash
export DIARIZATION_CONFIG=/path/to/my-diarization.yaml
export VOXTRAL_CONFIG=/path/to/my-config.yaml
```

---

## Tech stack

- **vLLM** — high-throughput inference engine for Voxtral realtime STT
- **SpeechBrain** — ECAPA-TDNN speaker embedding encoder
- **scikit-learn** — agglomerative clustering for consolidation
- **FastAPI + Uvicorn** — diarization REST API server
- **PyTorch** — tensor operations / GPU inference

---

## License

See the model licenses for [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
and [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
