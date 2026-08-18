"""Store zones: named polygons on the floor plane.

A zone is an area of the store you want numbers for, e.g. "aisle-6-endcap",
"checkout-queue", "entrance". Membership is tested against a person's foot point,
never their box center, because a shopper leaning over a shelf has a box center
that drifts into the neighbouring aisle.

Zones are per-camera and live in a JSON file next to the footage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SHELF_KIND = "shelf"
FLOOR_KIND = "floor"


@dataclass(frozen=True)
class Zone:
    """A named area of the camera's view.

    `kind` decides which body point membership is tested against, which is the
    whole difference between a visit and a reach:

    - `shelf` zones are drawn on the shelf face and tested against **wrists**. A
      hand entering one is a reach.
    - every other kind is floor, tested against **foot points**. A shopper standing
      in one is a visit.
    """

    name: str
    polygon: tuple[tuple[float, float], ...]
    kind: str = FLOOR_KIND

    @property
    def is_shelf(self) -> bool:
        return self.kind == SHELF_KIND

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"zone {self.name!r} needs at least 3 points, got {len(self.polygon)}"
            )

    @property
    def contour(self) -> np.ndarray:
        """OpenCV contour form: (N, 1, 2) float32."""
        return np.array(self.polygon, dtype=np.float32).reshape(-1, 1, 2)

    def contains(self, point: tuple[float, float]) -> bool:
        """True when the point is inside or exactly on the boundary."""
        return cv2.pointPolygonTest(self.contour, (float(point[0]), float(point[1])), False) >= 0


@dataclass(frozen=True)
class ZoneSet:
    """The zones for one camera."""

    zones: tuple[Zone, ...]

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(z.name for z in self.zones)

    @property
    def floor(self) -> ZoneSet:
        """Zones tested against foot points."""
        return ZoneSet(zones=tuple(z for z in self.zones if not z.is_shelf))

    @property
    def shelf(self) -> ZoneSet:
        """Zones tested against wrists."""
        return ZoneSet(zones=tuple(z for z in self.zones if z.is_shelf))

    def containing(self, point: tuple[float, float]) -> tuple[str, ...]:
        """Names of every zone containing the point.

        Zones may overlap on purpose: an end-cap sits inside an aisle, and a
        shopper standing at it should count for both.
        """
        return tuple(z.name for z in self.zones if z.contains(point))

    @classmethod
    def load(cls, path: str | Path) -> ZoneSet:
        # Every failure below names the file. A raw JSONDecodeError reaching a
        # user reads as a broken install when what actually happened is that
        # they edited a zones file by hand and dropped a comma.
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read {path}: {exc}") from exc

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}"
            ) from exc

        if not isinstance(raw, dict) or "zones" not in raw:
            raise ValueError(f"{path} has no 'zones' key")

        # A zone with no declared kind is a floor zone. Defaulting the other way
        # turns every unlabelled polygon into a wrist-tested one, which silently
        # reports no visits at all and no reaches either unless --pose is on.
        try:
            zones = tuple(
                Zone(
                    name=z["name"],
                    polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
                    kind=z.get("kind", FLOOR_KIND),
                )
                for z in raw["zones"]
            )
        except KeyError as exc:
            raise ValueError(f"{path}: a zone is missing {exc}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "at least 3 points" in str(exc):
                raise ValueError(f"{path}: {exc}") from exc
            raise ValueError(f"{path}: a zone polygon is malformed: {exc}") from exc
        names = [z.name for z in zones]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate zone names in {path}: {sorted(duplicates)}")
        return cls(zones=zones)

    def save(self, path: str | Path) -> None:
        payload = {
            "zones": [
                {
                    "name": z.name,
                    "kind": z.kind,
                    "polygon": [[x, y] for x, y in z.polygon],
                }
                for z in self.zones
            ]
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
