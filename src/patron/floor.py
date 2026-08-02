"""Ground-plane mapping: camera pixels to store floor coordinates.

A camera sees the store obliquely. Two shoppers ten pixels apart at the far end
of an aisle may be metres apart on the floor, while ten pixels in the foreground
is a few centimetres. Every spatial question worth asking (how far did they walk,
how fast, how much floor does this display actually command, where are they on
the store plan) needs image pixels turned into real floor positions first.

A single 3x3 homography does that, because the floor is a plane. Four
correspondences between points in the image and the same points on the store's
floor plan are enough to solve for it, and stores already have floor plans for
planogram and fire-code reasons, so nothing has to be reconstructed from pixels.

Two consequences worth stating plainly:

**Calibration points must be on the floor.** A homography maps one plane to
another. Clicking a shelf edge or the top of a display puts a point off the
ground plane and silently skews the entire mapping. Tile corners, floor markings
and door thresholds are the things to click.

**Only ground-plane points can be projected.** `Box.foot_point` is where a
shopper meets the floor, which is why zone membership already uses it. Projecting
a box centre or a wrist would be asking where a floating point lands on the
ground, which has no meaningful answer.

Floor coordinates are in real units, metres by default, which is what makes
several cameras describable in one shared space and what makes a distance mean
something. See CLAUDE.md: nothing here is biometric, and a floor position is not
an identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from patron.types import FrameResult

MIN_CORRESPONDENCES = 4

# Below this the perspective divide is at or behind the horizon, where the
# ground plane does not project to a finite point.
_MIN_HOMOGENEOUS_W = 1e-9


class FloorMap:
    """Maps image points on the ground plane to store floor coordinates.

    Built from correspondences rather than a raw matrix, because the
    correspondences are reviewable and a matrix is not. Someone looking at a
    saved calibration six months later can see which floor features were used
    and re-judge them; nine opaque floats tell them nothing.
    """

    def __init__(
        self,
        correspondences: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
        units: str = "m",
    ) -> None:
        if len(correspondences) < MIN_CORRESPONDENCES:
            raise ValueError(
                f"need at least {MIN_CORRESPONDENCES} correspondences to solve a "
                f"homography, got {len(correspondences)}"
            )

        self.correspondences = tuple(
            ((float(ix), float(iy)), (float(fx), float(fy)))
            for (ix, iy), (fx, fy) in correspondences
        )
        self.units = units
        self._matrix = self._solve()

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def _solve(self) -> np.ndarray:
        import cv2

        image = np.array([c[0] for c in self.correspondences], dtype=np.float64)
        floor = np.array([c[1] for c in self.correspondences], dtype=np.float64)

        # Plain least squares over every point, not RANSAC. These are a handful
        # of deliberately placed clicks, so discarding one as an outlier would
        # hide a bad click instead of surfacing it in the reprojection error.
        matrix, _ = cv2.findHomography(image, floor, method=0)
        if matrix is None:
            raise ValueError(
                "could not solve a homography from these points. Three or more "
                "collinear points, or duplicates, make the system degenerate"
            )
        return matrix

    def project(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """Image point on the ground plane to floor coordinates.

        Returns None for points at or above the horizon, where the ground plane
        does not project to a finite position. Returning a number there would be
        worse than returning nothing: it would look like a shopper standing
        somewhere impossible.
        """
        x, y, w = self._matrix @ np.array([point[0], point[1], 1.0])
        if w <= _MIN_HOMOGENEOUS_W:
            return None
        return (float(x / w), float(y / w))

    def project_many(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float] | None]:
        return [self.project(p) for p in points]

    @property
    def reprojection_error(self) -> float | None:
        """RMS distance, in floor units, between clicked and reprojected points.

        **None when there are exactly four correspondences, and that is the
        important case.** A homography has eight degrees of freedom and four
        point pairs supply exactly eight equations, so the fit is exact by
        construction. Four points clicked on the ceiling would reproject with
        zero error just as happily as four clicked on the floor.

        Reporting 0.000 there would be a confident number standing in for no
        evidence at all. The fifth point is what buys a residual, and with it the
        only automatic check that the calibration is on the ground plane.
        """
        if len(self.correspondences) <= MIN_CORRESPONDENCES:
            return None

        errors = []
        for image_point, floor_point in self.correspondences:
            projected = self.project(image_point)
            if projected is None:
                return float("inf")
            errors.append(
                (projected[0] - floor_point[0]) ** 2
                + (projected[1] - floor_point[1]) ** 2
            )
        return float(np.sqrt(np.mean(errors)))

    @property
    def is_verifiable(self) -> bool:
        """Are there enough points for the fit to be checkable at all?"""
        return len(self.correspondences) > MIN_CORRESPONDENCES

    def project_people(self, result: FrameResult) -> dict[int, tuple[float, float]]:
        """Floor positions by track id for everyone in a frame.

        Foot points only. A box centre floats somewhere up the body and asking
        where it lands on the ground has no meaningful answer, which is the same
        reason zone membership already uses `foot_point`.

        Anyone whose foot point falls above the horizon is left out rather than
        given a position. That happens with a bad calibration or a detection
        clipped at the top of the frame, and an absent shopper is easier to
        notice than one standing through a wall.
        """
        out: dict[int, tuple[float, float]] = {}
        for person in result.people:
            projected = self.project(person.box.foot_point)
            if projected is not None:
                out[person.track_id] = projected
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "units": self.units,
                    "correspondences": [
                        {"image": list(image), "floor": list(floor)}
                        for image, floor in self.correspondences
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> FloorMap:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            correspondences=tuple(
                (tuple(c["image"]), tuple(c["floor"])) for c in raw["correspondences"]
            ),
            units=raw.get("units", "m"),
        )

    def __repr__(self) -> str:
        error = self.reprojection_error
        quality = "unverifiable" if error is None else f"error {error:.3f}{self.units}"
        return f"FloorMap({len(self.correspondences)} points, {quality})"
