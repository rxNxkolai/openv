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
        pose_estimator = None
        if args.pose:
            from patron.pose import PoseEstimator

            pose_estimator = PoseEstimator()
            print("pose     MediaPipe (Apache 2.0), reach detection on")

        pipeline = Pipeline(detector, tracker=args.tracker, pose=pose_estimator)

        for warning in _inert_flag_warnings(args):
            print(f"note     {warning}")

        zones = None
        visit_tracker = None
        reach_tracker = None
        position_recorder = None
        store = None
        session_id = None
        visits_written = 0
        reaches_written = 0
        positions_written = 0

        if args.floor:
            from patron.floor import FloorMap, PositionRecorder

            floor_map = FloorMap.load(args.floor)
            position_recorder = PositionRecorder(
                floor_map, min_interval_s=args.position_interval
            )
            checked = (
                f"error {floor_map.reprojection_error:.3f}{floor_map.units}"
                if floor_map.is_verifiable
                else "UNVERIFIABLE, only 4 points"
            )
            print(f"floor    {args.floor} ({checked})")

        if args.zones:
            from patron.events import ReachTracker, VisitTracker
            from patron.zones import ZoneSet

            zones = ZoneSet.load(args.zones)
            print(
                f"zones    {len(zones.floor)} floor, {len(zones.shelf)} shelf: "
                f"{', '.join(zones.names)}"
            )
            if len(zones.shelf) and not args.pose:
                print("         note: shelf zones present but --pose is off, no reaches")
            # The complementary case needs the zones loaded before it can be told.
            for warning in _inert_flag_warnings(args, shelf_zone_count=len(zones.shelf)):
                if "kind 'shelf'" in warning:
                    print(f"note     {warning}")

            # Debounce windows are specified in seconds and converted, so the
            # behaviour is the same on a 10fps camera and a 60fps one.
            visit_tracker = VisitTracker(
                zones=zones,
                fps=info.fps,
                min_frames_inside=max(1, round(args.enter_seconds * info.fps)),
                min_frames_outside=max(1, round(args.exit_seconds * info.fps)),
                track_timeout_frames=max(1, round(args.lost_seconds * info.fps)),
            )
            if pose_estimator is not None and len(zones.shelf):
                reach_tracker = ReachTracker(
                    zones=zones,
                    fps=info.fps,
                    min_arm_extension=args.min_arm_extension,
                    frame_size=(info.width, info.height),
                )

        # The store is opened for zones or for floor positions. Opening it only
        # for zones would make `--floor` alone record nothing at all, silently.
        if args.zones or args.floor:
            from patron.store import EventStore

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

                if reach_tracker is not None and store is not None:
                    done = reach_tracker.update(result, dict(result.poses))
                    if done:
                        reaches_written += store.add_reaches(session_id, done)

                if position_recorder is not None and store is not None:
                    sampled = position_recorder.update(result)
                    if sampled:
                        positions_written += store.add_positions(session_id, sampled)

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
                if reach_tracker is not None:
                    reaches_written += store.add_reaches(
                        session_id, reach_tracker.flush()
                    )
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
        if reach_tracker is not None:
            print(f"shelf reaches      {reaches_written}")
    if position_recorder is not None:
        print(f"floor positions    {positions_written}")
    if args.out:
        print(f"written            {args.out}")

    if args.zones:
        print()
        _print_summary(args.db, session_id)

    return 0


def _cmd_bench_trackers(args: argparse.Namespace) -> int:
    """Compare trackers on identical detections.

    Tracker choice is the open accuracy question: a shopper who is occluded and
    reacquired gets a fresh id, so COUNT(DISTINCT track_id) overcounts people.
    Re-running detection per tracker would make the comparison unfair and slow,
    so detection runs once and every tracker replays the same cached frames.
    """
    import time

    import cv2

    from patron.detectors import RFDETRPersonDetector
    from patron.sources import VideoSource
    from patron.tracking import ALGORITHMS, PersonTracker

    algorithms = [a.strip() for a in args.trackers.split(",") if a.strip()]
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        print(f"error: unknown tracker(s) {unknown}. Known: {sorted(ALGORITHMS)}", file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    with VideoSource(args.source) as source:
        info = source.info
        print(f"source   {args.source}  {info.width}x{info.height} @ {info.fps:.1f}fps")
        print(f"frames   {args.frames}")
        print(f"detect   RF-DETR {args.variant} @ {args.resolution}px"
              + (f", tiled @ {args.slice}px" if args.slice else ""))

        detector = RFDETRPersonDetector(
            variant=args.variant,
            confidence=args.conf,
            device=device,
            resolution=args.resolution,
            slice_size=args.slice,
        )

        print("\ndetecting once (shared by every tracker)...")
        cached: list = []
        started = time.perf_counter()
        for index, _ts, frame in source.frames():
            if index >= args.frames:
                break
            detections = detector.detect(frame)
            # Keep a small frame: BoT-SORT uses it for camera motion compensation,
            # and motion is scale-relative so a downscale is still valid.
            small = cv2.resize(frame, (640, int(640 * info.height / info.width)))
            cached.append((detections, small))
            if (index + 1) % 10 == 0:
                rate = (index + 1) / (time.perf_counter() - started)
                print(f"  {index + 1}/{args.frames}  {rate:.1f} fps", end="\r", flush=True)

    if not cached:
        print("\nerror: no frames decoded", file=sys.stderr)
        return 1

    per_frame = [len(d) for d, _ in cached]
    mean_people = sum(per_frame) / len(per_frame)
    print(f"\n\ncached {len(cached)} frames, {mean_people:.1f} detections/frame mean\n")

    header = f"{'tracker':<16}{'unique ids':>11}{'mean/frame':>12}{'churn':>8}{'time':>9}"
    print(header)
    print("-" * len(header))

    lost_values = [float(v) for v in args.lost_seconds.split(",") if v.strip()]

    results = []
    for algorithm in algorithms:
      for lost in lost_values:
        tracker = PersonTracker(fps=info.fps, algorithm=algorithm, lost_seconds=lost)
        seen: set[int] = set()
        counts: list[int] = []
        last_frame_of: dict[int, int] = {}
        reacquired = 0
        started = time.perf_counter()
        for index, (detections, frame) in enumerate(cached):
            people = tracker.update(detections, frame)
            for person in people:
                # A track that reappears after a gap is one the tracker carried
                # through an occlusion instead of retiring. Fewer ids is only good
                # news if this rises with it; fewer ids while this stays flat means
                # identities are being merged, not preserved.
                previous = last_frame_of.get(person.track_id)
                if previous is not None and index - previous > 1:
                    reacquired += 1
                last_frame_of[person.track_id] = index
            seen.update(p.track_id for p in people)
            counts.append(len(people))
        elapsed = time.perf_counter() - started

        mean_tracked = sum(counts) / len(counts) if counts else 0.0
        # Churn: unique ids per person visible at any instant. A tracker that
        # never loses anyone approaches the number of people who genuinely
        # entered and left; one that fragments climbs above it.
        churn = len(seen) / mean_tracked if mean_tracked else float("inf")
        label = f"{algorithm}@{lost:g}s"
        results.append((label, len(seen), mean_tracked, churn, elapsed))
        print(
            f"{label:<16}{len(seen):>11}{mean_tracked:>12.1f}"
            f"{churn:>8.2f}{elapsed:>8.1f}s"
        )

    best = min(results, key=lambda r: r[3])
    print(f"\nlowest churn: {best[0]} ({best[3]:.2f} ids per visible person)")
    print("Lower is better. Same detections for every tracker, so the difference")
    print("is association only, not detection.")
    return 0


def _load_analysis(args: argparse.Namespace):
    from patron.analysis import analyze
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return None, None, None

    store = EventStore(args.db)
    session_id = None if args.all_sessions else store.latest_session_id()
    return store, session_id, analyze(store, session_id, stop_threshold_s=args.stop_seconds)


def _print_findings(analysis) -> None:
    from patron.analysis import MIN_SHOPPERS_FOR_CONFIDENCE

    if not analysis.findings:
        print("no shelf zones with reach data yet.")
        print("Draw a zone of kind 'shelf' and run `track --pose` to get a funnel.")
        return

    median = analysis.median_reach_rate
    print(f"shoppers observed   {analysis.total_shoppers}  (upper bound, see note)")
    print(
        "store median reach  "
        + ("n/a" if median is None else f"{median * 100:.0f}%")
    )
    print()

    marks = {"high": "!!", "medium": " !", "low": "  ", "none": " ?"}
    for f in analysis.findings:
        fn = f.funnel
        print(f"{marks[f.severity]} {f.headline}")
        if fn.passed:
            print(
                f"     passed {fn.passed}"
                f" -> stopped {fn.stopped}"
                f" -> reached {fn.reached}"
                f"   (aisle: {fn.floor_zone or 'unpaired'}, "
                f"mean dwell {fn.mean_dwell_s:.1f}s)"
            )
        print()

    print(f"!! high   ! medium   ? below {MIN_SHOPPERS_FOR_CONFIDENCE} shoppers, not a rate yet")


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Deterministic funnel analysis. No model, no API key."""
    store, session_id, analysis = _load_analysis(args)
    if store is None:
        return 1

    scope = "all sessions" if args.all_sessions else f"session {session_id}"
    print(f"{args.db}  ({scope})\n")
    _print_findings(analysis)
    store.close()
    return 0


def _cmd_advise(args: argparse.Namespace) -> int:
    """Agent recommendations over the computed findings."""
    from patron.agent import AdvisorUnavailable, advise

    store, session_id, analysis = _load_analysis(args)
    if store is None:
        return 1

    scope = "all sessions" if args.all_sessions else f"session {session_id}"
    print(f"{args.db}  ({scope})\n")
    _print_findings(analysis)

    if not analysis.actionable:
        print("\nnothing actionable to advise on.")
        store.close()
        return 0

    print(f"\nasking {args.model} about {len(analysis.actionable)} finding(s)...\n")
    try:
        recommendations = advise(analysis, model=args.model, effort=args.effort)
    except AdvisorUnavailable as exc:
        print(f"advisor unavailable: {exc}", file=sys.stderr)
        print(
            "\nThe funnel above is unaffected: it is computed locally and needs no"
            " model.\nRun `patron analyze` for it without this step.",
            file=sys.stderr,
        )
        store.close()
        return 1

    for i, r in enumerate(recommendations, 1):
        print(f"--- {i}. {r.zone}  [{r.confidence} confidence] " + "-" * 24)
        print(f"diagnosis   {r.diagnosis}")
        print(f"action      {r.action}")
        print(f"because     {r.rationale}")
        print(f"expect      {r.expected_effect}")
        print(f"\ndraft:\n{r.drafted_change}\n")

    if session_id is not None:
        written = store.add_recommendations(session_id, recommendations)
        print(f"stored {written} recommendation(s) with status 'proposed'.")
        print("Nothing has been applied. Approval is a human decision.")
    store.close()
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    """Live console: camera in, overlay and running numbers in the browser."""
    import threading
    import webbrowser

    import uvicorn

    from patron.web import LiveEngine, create_app

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    engine = LiveEngine(
        source=args.source,
        width=args.width,
        height=args.height,
        zones_path=args.zones,
        db_path=None if args.no_db else args.db,
        conf=args.conf,
        resolution=args.resolution,
        variant=args.variant,
        device=device,
        pose=args.pose,
        loop=not args.no_loop,
        min_arm_extension=args.min_arm_extension,
        floor_path=args.floor,
        position_interval=args.position_interval,
    )
    app = create_app(engine)

    url = f"http://127.0.0.1:{args.port}"
    print(f"source   {args.source}")
    print(f"model    RF-DETR {args.variant} @ {args.resolution}px on {device}")
    print(f"pose     {'on' if args.pose else 'off'}")
    print(f"zones    {args.zones}")
    print(f"db       {'disabled' if args.no_db else args.db}")

    replaying_file = not args.source.startswith("webcam:")
    if replaying_file and not args.no_loop and not args.no_db:
        # Every loop re-measures the same shoppers, so a persisted funnel from a
        # looping replay counts them once per pass. Fine for a demo, meaningless
        # as a measurement, and silence here would produce confident nonsense.
        print(
            "\nwarning  looping a file into the event store double-counts shoppers"
            "\n         on every pass. Use --no-loop to measure, or --no-db to demo."
        )

    print(f"\nopen     {url}\n")
    print("draw zones by clicking on the video. ctrl-c here to stop.\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def _probe_cameras(limit: int = 5) -> list[int]:
    import cv2

    found = []
    for index in range(limit):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:
            found.append(index)
        cap.release()
    return found


def _cmd_record(args: argparse.Namespace) -> int:
    """Capture fixed-camera footage to a file.

    Webcams routinely ignore a requested framerate and deliver something else. If
    the file is written claiming 30fps while the camera actually delivered 15, the
    video plays at double speed and every dwell time downstream is halved. So the
    real rate is measured during warmup and the file is written with that.
    """
    import cv2

    index = args.camera
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        available = _probe_cameras()
        print(f"error: could not open camera {index}", file=sys.stderr)
        print(
            f"available cameras: {available}" if available else "no cameras found",
            file=sys.stderr,
        )
        return 1

    # MJPG before anything else. OpenCV's DirectShow default is uncompressed YUY2,
    # which saturates USB bandwidth and silently drops a 1280x960 webcam to ~7fps.
    # Asking for MJPG lets the camera compress on-board and deliver its rated rate.
    if not args.raw_capture:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera   {index}")
    print(f"capture  {width}x{height} (asked for {args.width}x{args.height})")

    # Warm up and measure the rate the camera actually delivers.
    for _ in range(10):
        cap.read()
    warmup_frames = 30
    started = time.perf_counter()
    for _ in range(warmup_frames):
        if not cap.read()[0]:
            print("error: camera stopped during warmup", file=sys.stderr)
            cap.release()
            return 1
    measured_fps = warmup_frames / (time.perf_counter() - started)
    print(f"fps      {measured_fps:.1f} measured (asked for {args.fps})")

    if measured_fps < args.fps * 0.6:
        print(
            f"         note: USB bandwidth caps this camera at {measured_fps:.0f}fps"
            f" at {width}x{height}."
        )
        print(
            "         Fine for dwell and path. Drop to --width 640 --height 480"
            " for ~30fps if tracking looks choppy."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), measured_fps, (width, height)
    )
    if not writer.isOpened():
        print(f"error: could not open {out_path} for writing", file=sys.stderr)
        cap.release()
        return 1

    for remaining in range(args.countdown, 0, -1):
        print(f"starting in {remaining}...", end="\r", flush=True)
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            cap.read()

    print(f"RECORDING {args.seconds}s into {out_path}   (ctrl-c to stop early)")

    frames = 0
    started = time.perf_counter()
    try:
        while (elapsed := time.perf_counter() - started) < args.seconds:
            ok, frame = cap.read()
            if not ok:
                print("\ncamera stopped early")
                break
            writer.write(frame)
            frames += 1
            if frames % 15 == 0:
                print(
                    f"  {elapsed:6.1f}s / {args.seconds}s   {frames} frames",
                    end="\r",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        writer.release()
        cap.release()

    elapsed = time.perf_counter() - started
    size_mb = out_path.stat().st_size / 1_000_000 if out_path.exists() else 0.0
    print(f"\n\nrecorded  {frames} frames, {elapsed:.1f}s, {size_mb:.1f} MB")
    print(f"saved     {out_path}")
    print("\nnext:")
    print(f"  uv run patron zones {out_path} --out data/my.zones.json")
    print(f"  uv run patron track {out_path} --zones data/my.zones.json --db out/patron.db")
    return 0


def _print_summary(db_path: str, session_id: int | None) -> None:
    from patron.store import EventStore

    with EventStore(db_path) as store:
        rows = store.zone_summary(session_id)
        reaches = store.reach_summary(session_id)

    if not rows and not reaches:
        print("no zone visits recorded")
        return

    if rows:
        header = (
            f"{'floor zone':<24}{'shoppers':>9}{'visits':>8}{'stopped':>9}"
            f"{'stop%':>7}{'mean dwell':>12}{'max':>8}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            stop_pct = (r["stopped"] / r["visits"] * 100) if r["visits"] else 0.0
            print(
                f"{r['zone']:<24}{r['shoppers']:>9}{r['visits']:>8}{r['stopped']:>9}"
                f"{stop_pct:>6.0f}%{r['mean_dwell_s']:>11.2f}s{r['max_dwell_s']:>7.1f}s"
            )
        print("\nstopped = dwell >= 2s, a proxy for lingering rather than passing.")

    if reaches:
        header = (
            f"{'shelf zone':<24}{'shoppers':>9}{'reaches':>9}"
            f"{'mean hold':>12}{'max':>8}"
        )
        print()
        print(header)
        print("-" * len(header))
        for r in reaches:
            print(
                f"{r['zone']:<24}{r['shoppers']:>9}{r['reaches']:>9}"
                f"{r['mean_hold_s']:>11.2f}s{r['max_hold_s']:>7.1f}s"
            )
        print("\na reach is a hand inside a shelf zone: engagement, not just presence.")
    elif rows:
        # The store holds visits and reaches, not the zone set, so zero reach rows
        # cannot distinguish "no shelf zone drawn" from "no hand entered one".
        # Claiming the first is how you get sent off to redraw zones that are fine.
        print("\nno reaches recorded. Either no zone of kind 'shelf' was drawn over")
        print("the shelf face, or no hand entered one. Shelf zones also need --pose.")


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


# Flag defaults, kept here so an inert setting can be told from a deliberate one.
# argparse cannot report whether a value was typed or defaulted.
_TRACK_DEFAULTS = {
    "min_arm_extension": 2.5,
    "position_interval": 1.0,
    "enter_seconds": 0.2,
    "exit_seconds": 0.5,
    "lost_seconds": 1.5,
    "db": "out/patron.db",
}


def _inert_flag_warnings(args, shelf_zone_count: int | None = None) -> list[str]:
    """Flags that were set but cannot do anything in this configuration.

    Twice now this codebase has shipped a setting that looked like it worked and
    silently did nothing: shelf zones were uncreatable from either drawing tool,
    and `--floor` alone never opened the event store. Both were invisible at
    runtime. A flag that cannot take effect should say so rather than let someone
    conclude the feature is broken, or worse, trust an empty result.
    """
    def changed(name: str) -> bool:
        return getattr(args, name, None) != _TRACK_DEFAULTS[name]

    out: list[str] = []

    if args.pose and not args.zones:
        out.append(
            "--pose without --zones: pose runs and costs ~14ms per person per "
            "frame, and nothing consumes it. Reaches need a shelf zone."
        )
    if changed("min_arm_extension") and not args.pose:
        out.append("--min-arm-extension without --pose: no reaches are detected at all.")
    if changed("position_interval") and not args.floor:
        out.append("--position-interval without --floor: no positions are recorded.")
    for name in ("enter_seconds", "exit_seconds", "lost_seconds"):
        if changed(name) and not args.zones:
            out.append(f"--{name.replace('_', '-')} without --zones: no visits are tracked.")
    if changed("db") and not (args.zones or args.floor):
        out.append(
            "--db without --zones or --floor: nothing is written, so the database "
            "is created empty."
        )
    if args.no_trace and not (args.out or args.show):
        out.append("--no-trace without --out or --show: nothing is being rendered.")
    if shelf_zone_count == 0 and args.pose:
        out.append(
            "--pose but no zone of kind 'shelf': reaches are tested against "
            "wrists in shelf zones, and there are none."
        )

    return out


def _cmd_ask(args: argparse.Namespace) -> int:
    """Ask a question about the store, answered through the tool layer."""
    import json

    from patron.chat import ChatSession, ChatUnavailable
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    with EventStore(args.db) as store:
        session_id = None if args.all_sessions else store.latest_session_id()
        chat = ChatSession(store, session_id)

        if args.thread is not None:
            if not store.conversation(args.thread) and not any(
                t["id"] == args.thread for t in store.conversations()
            ):
                print(f"error: no thread {args.thread}", file=sys.stderr)
                return 1
            chat.resume(args.thread)
            print(f"thread   {args.thread} (resumed)\n")
        elif args.save is not None:
            chat.conversation_id = store.start_conversation(args.save, session_id)
            print(f"thread   {chat.conversation_id} ({args.save})\n")

        try:
            answer = chat.ask(args.question)
        except ChatUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            # analyze needs no model at all, and is the floor of the product.
            print("\n`patron analyze` needs no credentials and still works.", file=sys.stderr)
            return 1

    print(answer.text)

    if answer.citations and args.show_calls:
        print("\n--- what was looked up ---")
        for citation in answer.citations:
            params = ", ".join(f"{k}={v!r}" for k, v in citation.params.items())
            print(f"\n{citation.tool}({params})")
            print(json.dumps(citation.result, indent=2, default=str))
    elif answer.citations:
        calls = ", ".join(
            f"{c.tool}({', '.join(f'{k}={v!r}' for k, v in c.params.items())})"
            for c in answer.citations
        )
        # Every number above came from these. Naming them by default, because an
        # answer whose provenance is hidden behind a flag is an answer people
        # stop checking.
        print(f"\nlooked up: {calls}")
        print("re-run with --show-calls to see what each returned.")

    return 1 if answer.truncated else 0


def _cmd_digest(args: argparse.Namespace) -> int:
    """What is worth telling someone about this session. Needs no model.

    Exit code 0 means there is something to send and 2 means there is not, so a
    scheduled job can branch on it without parsing the output. An automation
    that posts "nothing to report" every morning teaches people to mute the
    channel before anything matters.
    """
    import json

    from patron.digest import build_digest
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    with EventStore(args.db) as store:
        session_id = None if args.all_sessions else store.latest_session_id()

        compare_with = args.compare_with
        if compare_with is None and args.compare:
            recent = [s["id"] for s in store.sessions()]
            previous = [s for s in recent if s != session_id]
            compare_with = previous[0] if previous else None

        digest = build_digest(store, session_id, compare_with=compare_with)

    if args.json:
        print(json.dumps(digest.as_dict(), indent=2))
    else:
        print(digest.render())

    if args.webhook or args.dry_run:
        from patron.deliver import build_payload, send

        payload = build_payload(digest, args.format)
        if args.dry_run:
            print("\n--- would post exactly this ---")
            print(json.dumps(payload, indent=2))
            if not digest.worth_sending:
                print("\n(and would not actually post it: nothing worth sending)")
            return 0 if digest.worth_sending else 2

        result = send(digest, args.webhook, fmt=args.format)
        print(f"\nwebhook: {result.detail}")
        if not result.ok:
            # A digest that silently fails to post is worse than one never
            # scheduled: the numbers look attended to and nobody is reading them.
            return 1

    return 0 if digest.worth_sending else 2


def _cmd_sessions(args: argparse.Namespace) -> int:
    """List what has been recorded, so session ids stop being magic numbers."""
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    with EventStore(args.db) as store:
        rows = store.sessions()

    if not rows:
        print("no sessions recorded yet")
        return 0

    # ISO-8601 with an offset is 25 characters, so a narrower column silently
    # pushes every other one out of alignment.
    header = (
        f"{'id':>4}  {'started':<25}  {'visits':>7}{'reaches':>8}"
        f"{'positions':>10}  source"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['id']:>4}  {row['started_at']:<25}  {row['visits']:>7}"
            f"{row['reaches']:>8}{row['positions']:>10}  {row['source']}"
        )
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    """Did a change to a zone work? Needs no model."""
    from patron.analysis import measure_change
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    with EventStore(args.db) as store:
        before, after = args.before, args.after

        # Nobody remembers session ids. Defaulting to the two most recent runs
        # that actually contain this zone is what makes the command usable, and
        # it names what it picked so the choice is never silent.
        if before is None or after is None:
            candidates = store.sessions_with_reaches(args.zone)
            if len(candidates) < 2:
                print(
                    f"{args.zone} has reach data in "
                    f"{len(candidates)} session{'' if len(candidates) == 1 else 's'}. "
                    f"Two are needed to measure a change.",
                    file=sys.stderr,
                )
                print("`patron sessions` lists what has been recorded.", file=sys.stderr)
                return 1
            after = after if after is not None else candidates[0]
            before = before if before is not None else candidates[1]
            print(f"comparing session {before} (before) with {after} (after)\n")

        change = measure_change(store, args.zone, before, after)

    if change is None:
        print(
            f"no reach data for {args.zone} in session {before} or "
            f"{after}, so there is nothing to compare"
        )
        return 1

    mark = {
        "improved": "++",
        "worsened": "--",
        "indistinguishable": "  ",
        "not_enough_data": " ?",
    }[change.verdict]

    print(f"{mark} {args.zone}: {change.verdict.replace('_', ' ')}")
    print(f"     {change.reason}\n")

    header = f"{'':<10}{'passed':>9}{'reached':>9}{'reach rate':>13}"
    print(header)
    print("-" * len(header))
    for label, funnel in (("before", change.before), ("after", change.after)):
        rate = (
            f"{funnel.reach_rate * 100:.1f}%"
            if funnel.has_confidence
            else "not enough"
        )
        print(f"{label:<10}{funnel.passed:>9}{funnel.reached:>9}{rate:>13}")

    if change.delta is not None and change.before.has_confidence and change.after.has_confidence:
        print(f"\nchange     {change.delta * 100:+.1f} percentage points")
    if not change.conclusive:
        # Said plainly, because a difference that is not a result reads as one
        # to anybody skimming a table of numbers.
        print("\nThis is not a result. Do not plan against it.")
    return 0


def _cmd_threads(args: argparse.Namespace) -> int:
    """List saved conversations."""
    from patron.store import EventStore

    if not Path(args.db).exists():
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    with EventStore(args.db) as store:
        threads = store.conversations()
        if args.thread is not None:
            messages = store.conversation(args.thread)
            if not messages:
                print(f"no messages in thread {args.thread}")
                return 0
            for message in messages:
                who = "you" if message["role"] == "user" else "patron"
                print(f"\n[{who}]  {message['created_at']}")
                print(message["text"])
                if message["citations"]:
                    calls = ", ".join(c["tool"] for c in message["citations"])
                    print(f"  (looked up: {calls})")
                if message["truncated"]:
                    print("  (this answer was cut off before it settled)")
            return 0

    if not threads:
        print("no saved threads. `patron ask ... --save \"a title\"` starts one.")
        return 0

    print(f"{'id':>4}  {'msgs':>4}  {'started':<20}  title")
    print("-" * 60)
    for thread in threads:
        print(
            f"{thread['id']:>4}  {thread['message_count']:>4}  "
            f"{thread['created_at']:<20}  {thread['title']}"
        )
    return 0


def _calibration_status(floor_map, point_count: int, units: str) -> tuple[str, bool]:
    """HUD line for the calibrator, and whether it should read as a warning.

    Extracted so the guarantee is testable: a four-point calibration must never
    present as verified. It fits exactly by construction, so reporting its
    reprojection error would show a reassuring 0.000 for a mapping nothing has
    checked, including one clicked entirely on shelf edges.
    """
    from patron.floor import MIN_CORRESPONDENCES

    if point_count < MIN_CORRESPONDENCES:
        return f"{point_count}/{MIN_CORRESPONDENCES} points, need more", True
    if floor_map is None:
        return "points are degenerate, spread them out", True
    if not floor_map.is_verifiable:
        return f"{MIN_CORRESPONDENCES} points: exact by construction, UNVERIFIABLE. add a 5th", True

    error = floor_map.reprojection_error
    return f"{point_count} points, error {error:.3f}{units}", error >= 0.25


def _cmd_floorplan(args: argparse.Namespace) -> int:
    """Calibrate the mapping from camera pixels to store floor coordinates.

    Click a feature on the floor, type where it sits on the store plan, repeat.

    Two things this deliberately makes hard to skip. Four points fit a homography
    exactly whatever they are, so a four-point calibration reports no error and
    cannot be checked: the HUD says so rather than showing a reassuring zero. And
    the rectified preview is the manual check, because a calibration clicked on
    shelf edges instead of the floor looks fine in every number and obvious the
    moment the floor is warped flat.
    """
    import cv2
    import numpy as np

    from patron.floor import MIN_CORRESPONDENCES, FloorMap
    from patron.floorview import rectify

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

    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    pending: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and not pending:
            pending.append((x, y))

    window = "patron floorplan"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, view.shape[1], view.shape[0])
    cv2.setMouseCallback(window, on_mouse)

    print("click a point ON THE FLOOR, then type its plan coordinates")
    print("floor features only: tile corners, floor markings, door thresholds.")
    print("a shelf edge or the top of a display is not on the ground plane.\n")
    print("u = undo last  |  w = rectified preview  |  s = save and quit  |  q = quit")

    floor_map: FloorMap | None = None

    def rebuild() -> FloorMap | None:
        if len(pairs) < MIN_CORRESPONDENCES:
            return None
        try:
            return FloorMap(tuple(pairs), units=args.units)
        except ValueError as exc:
            print(f"  cannot solve: {exc}")
            return None

    while True:
        canvas = view.copy()

        for i, (image_point, floor_point) in enumerate(pairs):
            px = (int(image_point[0] * scale), int(image_point[1] * scale))
            cv2.drawMarker(canvas, px, (0, 220, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                canvas,
                f"{i}: {floor_point[0]:g},{floor_point[1]:g}",
                (px[0] + 8, px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA,
            )

        status, warn = _calibration_status(floor_map, len(pairs), args.units)
        colour = (0, 165, 255) if warn else (0, 255, 128)

        cv2.putText(
            canvas, status, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if pending:
            x, y = pending.pop()
            source_point = (x / scale, y / scale)
            print(f"\nclicked ({source_point[0]:.0f}, {source_point[1]:.0f})")
            answer = input(f"  plan coordinates in {args.units}, as 'x y' or 'x,y': ")
            try:
                fx, fy = (float(v) for v in answer.replace(",", " ").split())
            except ValueError:
                print("  skipped, need two numbers")
                continue
            pairs.append((source_point, (fx, fy)))
            floor_map = rebuild()
            if floor_map is not None and floor_map.is_verifiable:
                print(f"  error now {floor_map.reprojection_error:.3f}{args.units}")

        elif key == ord("u") and pairs:
            removed = pairs.pop()
            floor_map = rebuild()
            print(f"removed point at {removed[1]}")

        elif key == ord("w"):
            if floor_map is None:
                print("need a solvable calibration first")
                continue
            # Extent from the plan points themselves, padded, so the preview
            # frames whatever area was actually calibrated.
            xs = [p[1][0] for p in pairs]
            ys = [p[1][1] for p in pairs]
            pad = 0.35 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            extent = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
            ppu = 900 / max(extent[2] - extent[0], extent[3] - extent[1])
            cv2.imshow("rectified floor", rectify(frame, floor_map, extent, ppu))
            print("preview: the floor's own tiles and joints should come out")
            print("square and parallel. if they fan out, points were off the floor.")

        elif key == ord("s"):
            if floor_map is None:
                print(f"need at least {MIN_CORRESPONDENCES} workable points to save")
                continue
            break

        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("quit without saving")
            return 0

    cv2.destroyAllWindows()  # closes the rectified preview too, if it was opened

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    floor_map.save(out_path)

    print(f"\nsaved {len(pairs)} correspondences to {out_path}")
    if floor_map.is_verifiable:
        print(f"reprojection error {floor_map.reprojection_error:.3f}{args.units}")
    else:
        print(
            "WARNING: 4 points fit exactly whatever they are, so this calibration "
            "has not been checked by anything. Add a fifth point to get a residual."
        )
    return 0


def _cmd_zones(args: argparse.Namespace) -> int:
    """Draw zone polygons on a still frame from the source."""
    import cv2
    import numpy as np

    from patron.zones import FLOOR_KIND, SHELF_KIND, Zone, ZoneSet

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
            # Shelf and floor zones are drawn in different colours because the
            # two are indistinguishable by shape and mixing them up is silent.
            colour = (0, 200, 255) if zone.is_shelf else (255, 128, 0)
            cv2.polylines(canvas, [pts], True, colour, 2)
            cv2.putText(
                canvas, f"{zone.name} ({zone.kind})", tuple(pts.min(axis=0) + [4, -6]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
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
            # Kind has to be chosen at draw time. It decides whether the polygon
            # is tested against wrists or foot points, and getting it wrong is
            # not visible later: the zone simply reports nothing.
            answer = input(f"kind, blank for {FLOOR_KIND}, s for {SHELF_KIND}: ")
            kind = SHELF_KIND if answer.strip().lower() in ("s", SHELF_KIND) else FLOOR_KIND
            zones.append(
                Zone(
                    name=name,
                    polygon=tuple((x / scale, y / scale) for x, y in points),
                    kind=kind,
                )
            )
            points.clear()
            print(f"added {name} ({kind})")
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
    track.add_argument(
        "--pose",
        action="store_true",
        help="run pose estimation, needed for reach detection on shelf zones",
    )
    # Spelled out rather than imported from events.DEFAULT_MIN_ARM_EXTENSION:
    # patron.events costs ~670ms to import and that would land on every
    # `patron --help`. The CLI always passes this through explicitly, so the
    # constructor default only applies to direct API use.
    track.add_argument(
        "--min-arm-extension",
        type=float,
        default=2.5,
        help="wrist-to-shoulder distance, in shoulder widths, before a hand in a "
        "shelf zone counts as a reach. 0 disables the check (default: 2.5)",
    )
    track.add_argument("--zones", help="zones.json, enables visit capture")
    track.add_argument(
        "--floor",
        help="floor.json from `patron floorplan`, records where shoppers stood "
        "in store floor coordinates",
    )
    track.add_argument(
        "--position-interval",
        type=float,
        default=1.0,
        help="seconds between floor position samples per shopper. Paths and "
        "dwell are fine at 1Hz and per-frame rows buy nothing (default: 1.0)",
    )
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

    live = sub.add_parser("live", help="live console in the browser")
    live.add_argument(
        "--source",
        default="webcam:0",
        help="webcam:N, or a video file to replay (looped, paced to real time)",
    )
    live.add_argument("--port", type=int, default=8000)
    live.add_argument(
        "--pose", action="store_true", help="enable reach detection on shelf zones"
    )
    live.add_argument(
        "--min-arm-extension",
        type=float,
        default=2.5,
        help="wrist-to-shoulder distance, in shoulder widths, before a hand in a "
        "shelf zone counts as a reach. 0 disables the check (default: 2.5)",
    )
    live.add_argument(
        "--floor",
        help="floor.json from `patron floorplan`, enables the plan view and "
        "records where shoppers stood",
    )
    live.add_argument(
        "--position-interval",
        type=float,
        default=1.0,
        help="seconds between floor position samples per shopper (default: 1.0)",
    )
    live.add_argument(
        "--db", default="out/patron.db", help="event store for the live session"
    )
    live.add_argument(
        "--no-db", action="store_true", help="do not persist, numbers vanish on exit"
    )
    live.add_argument(
        "--no-loop", action="store_true", help="stop at the end of a video file"
    )
    live.add_argument(
        "--zones", default="data/live.zones.json", help="zone file, created if missing"
    )
    live.add_argument("--width", type=int, default=1280)
    live.add_argument("--height", type=int, default=960)
    live.add_argument("--conf", type=float, default=0.4)
    live.add_argument("--resolution", type=int, default=896)
    live.add_argument(
        "--variant", default="medium", choices=["nano", "small", "medium", "large"]
    )
    live.add_argument("--device", choices=["cuda", "cpu"], help="default: auto")
    live.add_argument("--no-browser", action="store_true", help="do not auto-open")
    live.set_defaults(func=_cmd_live)

    record = sub.add_parser("record", help="capture fixed-camera footage to a file")
    record.add_argument(
        "--out", default="data/my-store.mp4", help="where to save the recording"
    )
    record.add_argument("--seconds", type=int, default=600, help="how long to record")
    record.add_argument("--camera", type=int, default=0, help="camera index")
    record.add_argument(
        "--width", type=int, default=1280, help="640 usually buys a much higher fps"
    )
    record.add_argument("--height", type=int, default=960)
    record.add_argument("--fps", type=int, default=30)
    record.add_argument(
        "--countdown", type=int, default=5, help="seconds before recording starts"
    )
    record.add_argument(
        "--raw-capture",
        action="store_true",
        help="skip the MJPG request, only if the camera misbehaves with it",
    )
    record.set_defaults(func=_cmd_record)

    ask = sub.add_parser("ask", help="ask a question about the store")
    ask.add_argument("question", help="what you want to know, in plain language")
    ask.add_argument("--db", default="out/patron.db", help="event store to answer from")
    ask.add_argument(
        "--all-sessions", action="store_true", help="answer across every session"
    )
    ask.add_argument(
        "--show-calls",
        action="store_true",
        help="print what each lookup returned, not just which ones ran",
    )
    ask.add_argument(
        "--save", metavar="TITLE", help="start a saved thread with this title"
    )
    ask.add_argument(
        "--thread", type=int, help="continue a saved thread, keeping its history"
    )
    ask.set_defaults(func=_cmd_ask)

    measure = sub.add_parser(
        "measure", help="did a change to a zone work? compares two sessions"
    )
    measure.add_argument("zone", help="the shelf zone that was changed")
    measure.add_argument(
        "--before",
        type=int,
        help="session id before the change (default: second most recent with this zone)",
    )
    measure.add_argument(
        "--after",
        type=int,
        help="session id after the change (default: most recent with this zone)",
    )
    measure.add_argument("--db", default="out/patron.db", help="event store to read")
    measure.set_defaults(func=_cmd_measure)

    digest = sub.add_parser(
        "digest", help="what is worth telling someone, or nothing at all"
    )
    digest.add_argument("--db", default="out/patron.db", help="event store to read")
    digest.add_argument(
        "--all-sessions", action="store_true", help="digest every session together"
    )
    digest.add_argument(
        "--compare",
        action="store_true",
        help="measure worrying zones against the previous session",
    )
    digest.add_argument(
        "--compare-with", type=int, metavar="SESSION", help="compare against this session"
    )
    digest.add_argument("--json", action="store_true", help="emit JSON for a connector")
    digest.add_argument(
        "--webhook",
        metavar="URL",
        help="POST the digest here. Slack, Discord, Teams and Google Chat "
        "incoming webhooks all accept the default text format",
    )
    digest.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="text posts {\"text\": ...}, json posts the full structure",
    )
    digest.add_argument(
        "--dry-run",
        action="store_true",
        help="print exactly what would be posted, and post nothing",
    )
    digest.set_defaults(func=_cmd_digest)

    sessions = sub.add_parser("sessions", help="list recorded sessions")
    sessions.add_argument("--db", default="out/patron.db", help="event store to read")
    sessions.set_defaults(func=_cmd_sessions)

    threads = sub.add_parser("threads", help="list or read saved conversations")
    threads.add_argument("--db", default="out/patron.db", help="event store to read")
    threads.add_argument("--thread", type=int, help="print this thread's messages")
    threads.set_defaults(func=_cmd_threads)

    floorplan = sub.add_parser(
        "floorplan", help="calibrate camera pixels to store floor coordinates"
    )
    floorplan.add_argument("source", help="video file or webcam:0 to calibrate on")
    floorplan.add_argument(
        "--out", default="data/store.floor.json", help="where to save the calibration"
    )
    floorplan.add_argument("--frame", type=int, default=0, help="frame to calibrate on")
    floorplan.add_argument(
        "--units", default="m", help="units of the plan coordinates (default: m)"
    )
    floorplan.set_defaults(func=_cmd_floorplan)

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

    bench = sub.add_parser(
        "bench-trackers", help="compare trackers on identical cached detections"
    )
    bench.add_argument("source", help="video file")
    bench.add_argument("--frames", type=int, default=120)
    bench.add_argument(
        "--trackers", default="bytetrack,botsort,ocsort,sort", help="comma separated"
    )
    bench.add_argument(
        "--lost-seconds",
        default="2.0",
        help="comma separated occlusion tolerances to sweep, e.g. 0.5,2,5",
    )
    bench.add_argument("--slice", type=int, metavar="PX", help="tiled inference")
    bench.add_argument("--resolution", type=int, default=896)
    bench.add_argument("--conf", type=float, default=0.4)
    bench.add_argument(
        "--variant", default="medium", choices=["nano", "small", "medium", "large"]
    )
    bench.add_argument("--device", choices=["cuda", "cpu"])
    bench.set_defaults(func=_cmd_bench_trackers)

    analyze_p = sub.add_parser(
        "analyze", help="funnel analysis and ranked findings (no model needed)"
    )
    analyze_p.add_argument("--db", default="out/patron.db")
    analyze_p.add_argument("--all-sessions", action="store_true")
    analyze_p.add_argument(
        "--stop-seconds",
        type=float,
        default=2.0,
        help="dwell above which a shopper counts as stopping, not passing",
    )
    analyze_p.set_defaults(func=_cmd_analyze)

    advise_p = sub.add_parser(
        "advise", help="agent recommendations over the findings (needs credentials)"
    )
    advise_p.add_argument("--db", default="out/patron.db")
    advise_p.add_argument("--all-sessions", action="store_true")
    advise_p.add_argument("--stop-seconds", type=float, default=2.0)
    advise_p.add_argument("--model", default="claude-opus-5")
    advise_p.add_argument(
        "--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"]
    )
    advise_p.set_defaults(func=_cmd_advise)

    fetch = sub.add_parser("fetch-sample", help="download a sample video into data/")
    fetch.add_argument("name", nargs="?", help="asset name, omit to list")
    fetch.set_defaults(func=_cmd_fetch_sample)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
