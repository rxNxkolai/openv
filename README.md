# Patron

Agentic shopper-behavior intelligence on existing store cameras.

Patron reads a store's existing security camera feeds, turns shopper behavior into a
structured event stream, and runs an agent over that stream that diagnoses what is going
wrong at the shelf and drafts the fix: planogram changes, packaging briefs, ad variants.

The cameras are the sensor. The agent is the product.

## Status

**M0: detection and tracking.** Video in, people detected, stable track IDs out.

## Quickstart

```bash
uv sync
```

Grab the grocery store sample footage:

```bash
uv run patron fetch-sample grocery-store
```

Run on it:

```bash
uv run patron track data/grocery-store.mp4 --out out/tracked.mp4
```

Run on a webcam:

```bash
uv run patron track webcam:0 --show
```

### Useful flags

| Flag | What it does |
|---|---|
| `--out PATH` | write an annotated mp4 |
| `--show` | live preview window, `q` to quit |
| `--max-frames N` | stop after N frames, good for a quick check |
| `--device cuda\|cpu` | defaults to cuda when available |
| `--conf FLOAT` | detection confidence threshold, default 0.4 |
| `--resolution PX` | detector input size, default 896, must be divisible by 32 |
| `--slice PX` | tiled inference, for wide or high camera mounts |
| `--tracker NAME` | `bytetrack` (default), `botsort`, `ocsort`, `sort` |
| `--no-trace` | turn off the path trails |
| `--no-half` | disable fp16 |

## Measured on an RTX 3070 (8GB)

Two things dominate whether a shopper gets seen at all, and both are tunable.

**Detector input resolution.** A mid-aisle shopper in a 4K frame, whole-frame inference:

| Resolution | Result | Latency |
|---|---|---|
| 576 (RF-DETR default) | 0.42 confidence, plus a false positive | 79ms |
| 896 (our default) | 0.80 confidence, clean | 107ms |
| 1280 | 0.83 confidence, clean | 203ms |

**Tiling.** An overhead fixed camera over a crowd, shoppers ~20px tall after downscaling:

| Mode | People found | Median confidence | Latency |
|---|---|---|---|
| Whole frame | 4 | low | 0.15s |
| `--slice 896` | 126 | 0.86 | 1.32s |
| `--slice 640` | 138 | 0.88 | 2.35s |

Tiling is the difference between missing a crowd entirely and resolving all of it. It costs
roughly 15x the compute, so it is opt-in per camera. Note that behavior analytics does not
need 30fps: dwell, path and conversion are all fine at 2 to 5fps, which is what makes the
tiled mode practical on real hardware.

fp16 was verified to match fp32 confidences to within 0.001, so it is on by default on CUDA.

## Stack

| Layer | Library | License |
|---|---|---|
| Detection | RF-DETR (Nano/Small/Medium/Large only) | Apache 2.0 |
| Tracking | roboflow/trackers (ByteTrack, BoT-SORT, OC-SORT, SORT) | Apache 2.0 |
| Tiling, annotation | supervision | MIT |
| Video IO | OpenCV | Apache 2.0 |

RF-DETR's XLarge and 2XLarge checkpoints ship under Roboflow's PML 1.0, not Apache 2.0,
and are deliberately not exposed.

No AGPL dependencies. See [CLAUDE.md](CLAUDE.md) for the full constraint list, including
the privacy posture that keeps this out of BIPA and EU AI Act exposure.
