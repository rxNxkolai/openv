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

        zones = None
        visit_tracker = None
        store = None
        session_id = None
        visits_written = 0

        if args.zones:
            from patron.events import VisitTracker
            from patron.store import EventStore
            from patron.zones import ZoneSet

            zones = ZoneSet.load(args.zones)
            print(f"zones    {len(zones)} loaded: {', '.join(zones.names)}")

            # Debounce windows are specified in seconds and converted, so the
            # behaviour is the same on a 10fps camera and a 60fps one.
            visit_tracker = VisitTracker(
                zones=zones,
                fps=info.fps,
                min_frames_inside=max(1, round(args.enter_seconds * info.fps)),
                min_frames_outside=max(1, round(args.exit_seconds * info.fps)),
                track_timeout_frames=max(1, round(args.lost_seconds * info.fps)),
            )
            store = EventStore(args.db)
            session_id = store.start_session(
                source=args.source, fps=info.fps, width=info.width, height=info.height
            )
            print(f"db       {args.db} (session {session_id})")

        renderer = Renderer(
            resolution_wh=(info.width, info.height),
            draw_traces=not args.no_trace,
            zones=zones,
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

                if visit_tracker is not None and store is not None:
                    completed = visit_tracker.update(result)
                    if completed:
                        visits_written += store.add_visits(session_id, completed)

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
            # Anyone still standing in a zone when the stream ends has an open
            # visit. Without this flush they never reach the numbers.
            if visit_tracker is not None and store is not None:
                visits_written += store.add_visits(session_id, visit_tracker.flush())
                store.close()
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
    if args.zones:
        print(f"zone visits        {visits_written}")
    if args.out:
        print(f"written            {args.out}")

    if args.zones:
        print()
        _print_summary(args.db, session_id)

    return 0


def _print_summary(db_path: str, session_id: int | None) -> None:
    from patron.store import EventStore

    with EventStore(db_path) as store:
        rows = store.zone_summary(session_id)

    if not rows:
        print("no zone visits recorded")
        return

    header = f"{'zone':<24}{'shoppers':>9}{'visits':>8}{'stopped':>9}{'stop%':>7}{'mean dwell':>12}{'max':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        stop_pct = (r["stopped"] / r["visits"] * 100) if r["visits"] else 0.0
        print(
            f"{r['zone']:<24}{r['shoppers']:>9}{r['visits']:>8}{r['stopped']:>9}"
            f"{stop_pct:>6.0f}%{r['mean_dwell_s']:>11.2f}s{r['max_dwell_s']:>7.1f}s"
        )
    print("\nstopped = dwell >= 2s. Until M2 adds reach detection, that is the")
    print("stand-in for passed-by vs actually engaged.")


def _cmd_report(args: argparse.Namespace) -> int:
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    with EventStore(args.db) as store:
        session_id = None if args.all_sessions else store.latest_session_id()

    scope = "all sessions" if args.all_sessions else f"session {session_id}"
    print(f"{args.db}  ({scope})\n")
    _print_summary(args.db, session_id)
    return 0


def _cmd_zones(args: argparse.Namespace) -> int:
    """Draw zone polygons on a still frame from the source."""
    import cv2
    import numpy as np

    from patron.zones import Zone, ZoneSet

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"error: could not open {args.source}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"error: could not read frame {args.frame}", file=sys.stderr)
        return 1

    height, width = frame.shape[:2]
    scale = min(1.0, 1600 / max(width, height))
    view = cv2.resize(frame, (int(width * scale), int(height * scale)))

    points: list[tuple[int, int]] = []
    zones: list[Zone] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    window = "patron zones"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, view.shape[1], view.shape[0])
    cv2.setMouseCallback(window, on_mouse)

    print("click to add points  |  n = finish zone  |  u = undo point")
    print("r = remove last zone |  s = save and quit  |  q = quit without saving")

    while True:
        canvas = view.copy()
        for i, zone in enumerate(zones):
            pts = (np.array(zone.polygon) * scale).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (255, 128, 0), 2)
            cv2.putText(
                canvas, zone.name, tuple(pts.min(axis=0) + [4, -6]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2, cv2.LINE_AA,
            )
        if points:
            cv2.polylines(canvas, [np.array(points, np.int32)], False, (0, 255, 255), 2)
            for p in points:
                cv2.circle(canvas, p, 4, (0, 255, 255), -1)

        cv2.putText(
            canvas, f"zones: {len(zones)}  points: {len(points)}", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and points:
            points.pop()
        elif key == ord("r") and zones:
            removed = zones.pop()
            print(f"removed {removed.name}")
        elif key == ord("n"):
            if len(points) < 3:
                print("need at least 3 points")
                continue
            name = input("zone name: ").strip()
            if not name:
                print("skipped, no name given")
                continue
            zones.append(
                Zone(
                    name=name,
                    polygon=tuple((x / scale, y / scale) for x, y in points),
                )
            )
            points.clear()
            print(f"added {name}")
        elif key == ord("s"):
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("quit without saving")
            return 0

    cv2.destroyAllWindows()

    if not zones:
        print("no zones defined, nothing saved")
        return 1

    out_path = Path(args.out)
    ZoneSet(zones=tuple(zones)).save(out_path)
    print(f"saved {len(zones)} zones to {out_path}")
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
    track.add_argument("--zones", help="zones.json, enables visit capture")
    track.add_argument(
        "--db", default="out/patron.db", help="event store path (default: out/patron.db)"
    )
    track.add_argument(
        "--enter-seconds",
        type=float,
        default=0.2,
        help="time inside a zone before a visit opens, debounces boundary flicker",
    )
    track.add_argument(
        "--exit-seconds",
        type=float,
        default=0.5,
        help="time outside a zone before a visit closes",
    )
    track.add_argument(
        "--lost-seconds",
        type=float,
        default=1.5,
        help="time a track can be missing before its visits are closed",
    )
    track.set_defaults(func=_cmd_track)

    zones = sub.add_parser("zones", help="draw zone polygons on a frame")
    zones.add_argument("source", help="video file to draw zones on")
    zones.add_argument("--out", default="zones.json", help="where to save")
    zones.add_argument("--frame", type=int, default=0, help="frame to draw on")
    zones.set_defaults(func=_cmd_zones)

    report = sub.add_parser("report", help="per-zone funnel numbers from the event store")
    report.add_argument(
        "--db", default="out/patron.db", help="event store path (default: out/patron.db)"
    )
    report.add_argument(
        "--all-sessions", action="store_true", help="aggregate across every session"
    )
    report.set_defaults(func=_cmd_report)

    fetch = sub.add_parser("fetch-sample", help="download a sample video into data/")
    fetch.add_argument("name", nargs="?", help="asset name, omit to list")
    fetch.set_defaults(func=_cmd_fetch_sample)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
