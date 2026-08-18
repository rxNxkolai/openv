"""The privacy posture, enforced rather than described.

CLAUDE.md constraint 2 is what keeps this product out of BIPA and EU AI Act
exposure. Note the precise version, because the imprecise one is a liability in a
deposition: *Arnold v. Target* and *Jankowski v. Home Depot* are facial-recognition
loss-prevention cases, not behaviour analytics. They are the category we avoid.
What they establish is that plaintiffs plead on information and belief and survive
dismissal anyway, and that vendors are named directly. So these tests are not only
a design guardrail, they are the evidence that answers the complaint early.

Right now that posture is honoured by construction: the code happens not to store
faces because nobody wrote code that stores faces.

That is not the same as a guarantee. Someone adds frame dumping to debug a
tracker, or a face landmark to improve pose, and nothing anywhere objects. These
tests object.

They are deliberately about structure rather than behaviour. A behavioural test
can only show that the posture held for the inputs it tried.

**Verifying these have teeth.** Copying the tree, introducing a violation and
re-running does not work for the checks that import openv: the package is
installed editable, so an import inside the copy still resolves to the original
source and the test passes against unmodified code. Those have to be checked in
process by patching the module attribute. The file-scanning checks read via
`__file__` and do see a copy. Every check here has been confirmed to fail on a
real violation by one method or the other; if you add one, confirm it the same
way, because a privacy guarantee nobody has watched fail is not a guarantee.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "openv"


def sources() -> dict[str, str]:
    return {
        str(p.relative_to(SRC)).replace("\\", "/"): p.read_text(encoding="utf-8")
        for p in SRC.rglob("*.py")
    }


# --------------------------------------------------------------------------
# Nothing biometric is ever computed
# --------------------------------------------------------------------------

def test_pose_collects_no_face_landmark():
    """MediaPipe landmarks 0 to 10 are the face: nose, eyes, ears, mouth.

    Collecting one would make this a face-processing system, which is the
    single line that must not be crossed. Patron needs to know whether a hand
    went toward a shelf and nothing else about the body.
    """
    from openv.pose import JOINTS

    assert JOINTS, "the joint list vanished, which is not a pass"
    for name, index in JOINTS.items():
        assert index >= 11, f"{name} is a face landmark (index {index})"


def test_pose_landmarks_are_only_ever_read_through_the_joint_map():
    """`JOINTS` is the gate, so nothing may reach around it.

    `test_pose_collects_no_face_landmark` proves the map holds no face index.
    That is worth nothing if some other line does `landmarks[3]` directly, so
    this is the check that makes the first one load-bearing.

    The claim this defends is narrower than "we never compute anything facial",
    and the narrow one is the true one. MediaPipe computes all 33 landmarks
    including the face; what Patron can prove is that **no facial landmark value
    is ever read out of the model**. That is exactly this test.
    """
    text = (SRC / "pose.py").read_text(encoding="utf-8")

    subscripts = re.findall(r"\blandmarks\s*\[([^\]]+)\]", text)
    assert subscripts, "pose.py stopped indexing landmarks, so this check went blind"

    for expression in subscripts:
        assert expression.strip() == "index", (
            f"pose.py reads landmarks[{expression.strip()}] directly. Every landmark "
            f"must come from the JOINTS mapping so the face indices stay unreachable."
        )


def test_no_module_mentions_face_or_embedding_machinery():
    forbidden = ("face_mesh", "face_detection", "FaceMesh", "face_encoding", "embedding")
    offenders = []
    for path, text in sources().items():
        for word in forbidden:
            # Prose in a docstring saying we do not do this is fine; a call is not.
            for match in re.finditer(rf"\b{re.escape(word)}\b", text):
                line = text[: match.start()].count("\n") + 1
                snippet = text.splitlines()[line - 1].strip()
                if snippet.startswith("#") or snippet.startswith('"'):
                    continue
                if "never" in snippet.lower() or "no " in snippet.lower():
                    continue
                offenders.append(f"{path}:{line} {snippet}")

    assert offenders == [], f"biometric machinery appeared: {offenders}"


# --------------------------------------------------------------------------
# Nothing visual is persisted
# --------------------------------------------------------------------------

#: Every place allowed to write pixels to disk, and why. Each is something a
#: user explicitly asked for on the command line, not a side effect of running.
FRAME_WRITERS = {
    "cli.py": "track --out writes an annotated mp4, and record captures footage",
}


def test_only_explicitly_requested_code_writes_pixels():
    """A frame written as a side effect is a retained raw frame.

    The posture allows a processing buffer and an output the operator asked
    for. It does not allow a debug dump that survives the session.
    """
    writers = re.compile(r"cv2\.imwrite|cv2\.VideoWriter\(")
    offenders = [
        path
        for path, text in sources().items()
        if writers.search(text) and path not in FRAME_WRITERS
    ]

    assert offenders == [], (
        f"these modules write pixels without being asked to: {offenders}. "
        f"If that is deliberate, add it to FRAME_WRITERS with the reason."
    )


def test_the_event_store_has_no_column_that_could_hold_an_image():
    from openv.store import SCHEMA

    lowered = SCHEMA.lower()
    for forbidden in (" blob", "image", "frame_data", "thumbnail", "embedding", "jpeg"):
        assert forbidden not in lowered, f"the schema can hold {forbidden.strip()}"


def test_the_live_engine_keeps_one_frame_not_a_history():
    """A buffer is a buffer. A list of buffers is a recording."""
    from openv.web.engine import LiveEngine

    text = Path(LiveEngine.__module__.replace(".", "/") + ".py")
    source = (SRC.parent.parent / "src" / text).read_text(encoding="utf-8")

    # The single slot the MJPEG stream reads from.
    assert "self._jpeg = buffer.tobytes()" in source
    assert "self._jpeg.append" not in source
    assert "_jpeg_history" not in source


# --------------------------------------------------------------------------
# Identity dies with the session
# --------------------------------------------------------------------------

def test_no_table_links_a_person_across_sessions():
    from openv.store import SCHEMA

    lowered = SCHEMA.lower()
    for forbidden in ("person", "identity", "visitor", "customer_id", "shopper_id"):
        assert forbidden not in lowered, (
            f"a '{forbidden}' table or column would outlive the session that "
            f"created it, which is re-identification"
        )


def test_every_track_id_is_qualified_by_a_session(tmp_path):
    """The same track id in two sessions is two different people.

    Nothing may join them, and the schema should make joining them wrong rather
    than merely discouraged.
    """
    from openv.events import ZoneVisit
    from openv.store import EventStore

    def visit(track_id):
        return ZoneVisit(
            track_id=track_id, zone="aisle", entered_frame=0, entered_s=0.0,
            exited_frame=30, exited_s=1.0,
        )

    with EventStore(tmp_path / "e.db") as store:
        first = store.start_session("monday.mp4", fps=30.0, width=100, height=100)
        second = store.start_session("tuesday.mp4", fps=30.0, width=100, height=100)
        store.add_visits(first, [visit(7)])
        store.add_visits(second, [visit(7)])

        # Two shoppers, not one who came back.
        assert store.total_shoppers(first) == 1
        assert store.total_shoppers(second) == 1
        rows = {r["zone"]: r for r in store.zone_summary(first)}
        assert rows["aisle"]["shoppers"] == 1


def test_positions_are_scoped_the_same_way(tmp_path):
    from openv.floor import FloorPosition
    from openv.store import EventStore

    with EventStore(tmp_path / "e.db") as store:
        first = store.start_session("a.mp4", fps=30.0, width=100, height=100)
        second = store.start_session("b.mp4", fps=30.0, width=100, height=100)
        for session, x in ((first, 1.0), (second, 9.0)):
            store.add_positions(
                session,
                [
                    FloorPosition(track_id=3, frame=0, t_s=0.0, x=x, y=0.0),
                    FloorPosition(track_id=3, frame=30, t_s=1.0, x=x, y=1.0),
                ],
            )

        assert store.paths(first)[3][0][1] == 1.0
        assert store.paths(second)[3][0][1] == 9.0


# --------------------------------------------------------------------------
# What actually leaves the edge
# --------------------------------------------------------------------------

def test_no_emotion_or_demographic_inference_anywhere():
    """Scientifically contested and legally radioactive under EU AI Act
    Recital 44 and Art. 5, so it is not a feature to be added carefully."""
    forbidden = ("emotion", "sentiment", "age_estimate", "gender", "ethnicity")
    offenders = []
    for path, text in sources().items():
        for word in forbidden:
            for match in re.finditer(rf"\b{re.escape(word)}\w*", text, re.IGNORECASE):
                line_no = text[: match.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1]
                stripped = line.strip()
                if stripped.startswith("#") or "No " in line or "no " in line:
                    continue
                offenders.append(f"{path}:{line_no} {stripped}")

    assert offenders == [], f"prohibited inference appeared: {offenders}"


@pytest.mark.parametrize("module", ["digest", "deliver", "tools"])
def test_the_outbound_modules_carry_no_track_identifier(module):
    """Anything that leaves the machine is aggregate.

    A zone name and a count describe a place. A track id describes a person,
    even a session-scoped one, and it has no business on a wire.
    """
    text = (SRC / f"{module}.py").read_text(encoding="utf-8")

    for line_no, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        assert '"track_id"' not in line, f"{module}.py:{line_no} emits a track id"
