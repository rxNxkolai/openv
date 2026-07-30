# Patron

Agentic shopper-behavior intelligence on existing store cameras.

Patron reads a store's existing security camera feeds, turns shopper behavior into a
structured event stream, and runs an agent over that stream that diagnoses what is going
wrong at the shelf and drafts the fix: planogram changes, packaging briefs, ad variants.

The cameras are the sensor. The agent is the product.

## Status

**M0: detection and tracking.** Video in, people detected, stable track IDs out.
**M1: zones and events.** Named store areas, visit records with dwell, per-zone funnel.

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

## Recording your own footage

Fixed-camera footage is what this is built for. The bundled samples cannot validate
dwell: one is 8 seconds long, the other has a moving camera.

```bash
uv run patron record --out data/my-store.mp4 --seconds 600
```

The camera must not move once recording starts.

Webcams routinely ignore the framerate you ask for. The recorder measures the rate
actually delivered during warmup and writes the file with that, because a file that
claims 30fps while the camera delivered 7 plays at 4x speed and divides every dwell
time by four. USB bandwidth is usually the limit: a camera that manages 30fps at
640x480 often drops to 7fps at 1280x960. Both are fine for dwell and path.

## Zones and events (M1)

Draw the areas you want numbers for. Click points, `n` to name and close a zone,
`s` to save.

```bash
uv run patron zones data/grocery-store.mp4 --out data/store.zones.json
```

Then track with zones enabled. Every shopper's stay in every zone becomes a visit
record in SQLite, and the funnel prints at the end.

```bash
uv run patron track data/grocery-store.mp4 --zones data/store.zones.json --db out/patron.db
```

Re-read the numbers any time:

```bash
uv run patron report --db out/patron.db
uv run patron report --db out/patron.db --all-sessions
```

### What a visit is

One continuous stay by one shopper in one zone. Enter time, exit time and dwell are
all derivable from it, so the visit is the durable record.

Two details make the dwell numbers trustworthy:

- **Debounce.** A shopper standing on a zone edge flickers in and out. A visit only
  opens after `--enter-seconds` continuously inside and only closes after
  `--exit-seconds` continuously outside, so one shopper at one shelf is one visit,
  not forty.
- **Backdating.** Timestamps are rolled back to when the streak actually started,
  otherwise every dwell reads short by the debounce window.

Shoppers who walk out of frame while inside a zone get their visit closed after
`--lost-seconds` rather than hanging open and vanishing from the numbers.

Zones may overlap on purpose. An end-cap sits inside an aisle, and a shopper at it
should count for both.

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
