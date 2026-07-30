"""Patron CLI."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "out"


def _cmd_track(args: argparse.Namespace) -> int:
    import cv2

    from patron.detectors import RFDETRPersonDetector
    from patron.pipeline import Pipeline
    from patron.render import Renderer
    from patron.sources import VideoSource

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"source   {args.source}")
    print(f"device   {device}")
    slicing = f", tiled @ {args.slice}px" if args.slice else ""
    print(f"model    RF-DETR {args.variant} @ {args.resolution}px{slicing} (Apache 2.0)")
    print(f"tracker  {args.tracker} (Apache 2.0)")
    print(f"conf     {args.conf}")

    with VideoSource(args.source) as source:
        info = source.info
        print(
            f"stream   {info.width}x{info.height} @ {info.fps:.1f}fps"
            + (f", {info.frame_count} frames" if info.frame_count else ", live")
        )

        detector = RFDETRPersonDetector(
            variant=args.variant,
            confidence=args.conf,
            device=device,
            half=not args.no_half,
            resolution=args.resolution,
            slice_size=args.slice,
        )
        pipeline = Pipeline(detector, tracker=args.tracker)
        renderer = Renderer(
            resolution_wh=(info.width, info.height), draw_traces=not args.no_trace
        )

        writer = None
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                info.fps,
                (info.width, info.height),
            )
            if not writer.isOpened():
                print(f"error: could not open {out_path} for writing", file=sys.stderr)
                return 1

        seen_ids: set[int] = set()
        frames_done = 0
        started = time.perf_counter()

        try:
            for frame, result in pipeline.run_with_frames(
                source, max_frames=args.max_frames
            ):
                seen_ids.update(p.track_id for p in result.people)
                frames_done += 1

                if writer is not None or args.show:
                    canvas = renderer.annotate(frame, result)
                    if writer is not None:
                        writer.write(canvas)
                    if args.show:
                        cv2.imshow("patron", canvas)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if frames_done % 30 == 0:
                    rate = frames_done / (time.perf_counter() - started)
                    print(
                        f"  frame {result.frame_index:>6}  "
                        f"people {result.count:>2}  "
                        f"unique {len(seen_ids):>3}  "
                        f"{rate:5.1f} fps",
                        end="\r",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            if writer is not None:
                writer.release()
            if args.show:
                cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    rate = frames_done / elapsed if elapsed > 0 else 0.0

    print("\n")
    print(f"frames processed   {frames_done}")
    print(f"unique track ids   {len(seen_ids)}")
    print(f"processing rate    {rate:.1f} fps  ({elapsed:.1f}s wall)")
    if args.out:
        print(f"written            {args.out}")

    return 0


def _cmd_fetch_sample(args: argparse.Namespace) -> int:
    from supervision.assets import VideoAssets, download_assets

    names = sorted(a.name for a in VideoAssets)

    if args.name is None:
        print("available sample videos:\n")
        for name in names:
            print(f"  {name.lower().replace('_', '-')}")
        print("\npick one, e.g.  patron fetch-sample people-walking")
        return 0

    wanted = args.name.upper().replace("-", "_")
    if wanted not in names:
        print(f"error: no sample named {args.name!r}", file=sys.stderr)
        print(f"available: {', '.join(n.lower().replace('_', '-') for n in names)}")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    asset = VideoAssets[wanted]

    downloaded = Path(download_assets(asset))
    target = DATA_DIR / downloaded.name
    if downloaded.resolve() != target.resolve():
        shutil.move(str(downloaded), str(target))

    print(f"saved  {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patron", description="Shopper-behavior intelligence on store cameras."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    track = sub.add_parser("track", help="detect and track people in a video or camera")
    track.add_argument("source", help="path to a video file, or webcam:0")
    track.add_argument("--out", help="write an annotated mp4 here")
    track.add_argument("--show", action="store_true", help="live preview, q to quit")
    track.add_argument("--max-frames", type=int, help="stop after N frames")
    track.add_argument("--device", choices=["cuda", "cpu"], help="default: auto")
    track.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    track.add_argument(
        "--resolution",
        type=int,
        default=896,
        help="detector input size, must be divisible by 32. "
        "Higher finds smaller/further shoppers at a speed cost (default: 896)",
    )
    track.add_argument(
        "--variant",
        default="medium",
        choices=["nano", "small", "medium", "large"],
        help="RF-DETR size (Apache 2.0 checkpoints only)",
    )
    track.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "botsort", "ocsort", "sort"],
        help="association algorithm (default: bytetrack)",
    )
    track.add_argument(
        "--slice",
        type=int,
        metavar="PX",
        help="tiled inference at this tile size, e.g. 896. Finds far/small shoppers "
        "a wide or high camera mount would otherwise miss. Costs ~15x compute",
    )
    track.add_argument("--no-trace", action="store_true", help="hide path trails")
    track.add_argument(
        "--no-half", action="store_true", help="disable fp16 inference on cuda"
    )
    track.set_defaults(func=_cmd_track)

    fetch = sub.add_parser("fetch-sample", help="download a sample video into data/")
    fetch.add_argument("name", nargs="?", help="asset name, omit to list")
    fetch.set_defaults(func=_cmd_fetch_sample)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
