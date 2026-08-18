"""Top-down rendering of shoppers on the store floor.

Like `render.py`, this is a demo and debugging surface, not part of the product
data path. Nothing here writes to the event store.

Two separate things live here and they answer different questions:

- `rectify` warps the camera frame onto the floor plane. It exists to check a
  calibration, because a homography built from four clicked points always
  solves, and a wrong one looks exactly like a right one in the numbers. Warp
  the frame and the answer is visible: if the floor's own tiles, joints or
  markings come out square and parallel, the calibration is on the ground plane.
  If they come out as a fan, it is not.

- `FloorView` draws shoppers as positions on a plan. That is the bird's-eye the
  product is heading toward, and it consumes floor coordinates rather than
  pixels, so several cameras can eventually draw into one of these.
"""

from __future__ import annotations

import cv2
import numpy as np

from openv.floor import FloorMap

_BACKGROUND = (24, 24, 28)
_GRID = (44, 44, 52)
_AXIS = (80, 80, 92)
_PERSON = (0, 200, 255)
_TRAIL = (120, 90, 40)


def rectify(
    frame: np.ndarray,
    floor_map: FloorMap,
    extent: tuple[float, float, float, float],
    pixels_per_unit: float = 40.0,
) -> np.ndarray:
    """Warp a camera frame onto the floor plane, for checking a calibration.

    `extent` is (min_x, min_y, max_x, max_y) in floor units. Only the part of
    the floor inside it is drawn, because a homography maps the region near the
    horizon to enormous distances and rendering that would be mostly stretched
    noise.
    """
    min_x, min_y, max_x, max_y = extent
    width = max(1, int(round((max_x - min_x) * pixels_per_unit)))
    height = max(1, int(round((max_y - min_y) * pixels_per_unit)))

    # Floor units to output pixels, with y flipped so the plan reads the way a
    # floor plan does rather than upside down.
    to_pixels = np.array(
        [
            [pixels_per_unit, 0.0, -min_x * pixels_per_unit],
            [0.0, -pixels_per_unit, max_y * pixels_per_unit],
            [0.0, 0.0, 1.0],
        ]
    )
    return cv2.warpPerspective(
        frame, to_pixels @ floor_map.matrix, (width, height), flags=cv2.INTER_LINEAR
    )


class FloorView:
    """A plan-view canvas in floor coordinates."""

    def __init__(
        self,
        extent: tuple[float, float, float, float],
        pixels_per_unit: float = 40.0,
        grid_step: float = 1.0,
        units: str = "m",
    ) -> None:
        self.extent = extent
        self.pixels_per_unit = pixels_per_unit
        self.grid_step = grid_step
        self.units = units

        min_x, min_y, max_x, max_y = extent
        self.width = max(1, int(round((max_x - min_x) * pixels_per_unit)))
        self.height = max(1, int(round((max_y - min_y) * pixels_per_unit)))

    def to_pixels(self, point: tuple[float, float]) -> tuple[int, int]:
        min_x, _min_y, _max_x, max_y = self.extent
        return (
            int(round((point[0] - min_x) * self.pixels_per_unit)),
            int(round((max_y - point[1]) * self.pixels_per_unit)),
        )

    def _blank(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), _BACKGROUND, dtype=np.uint8)
        min_x, min_y, max_x, max_y = self.extent

        # A grid in real units, so distance on the plan is readable by eye. This
        # is the whole reason floor coordinates are metric rather than arbitrary.
        step = self.grid_step
        x = np.ceil(min_x / step) * step
        while x <= max_x:
            px = self.to_pixels((x, 0))[0]
            cv2.line(canvas, (px, 0), (px, self.height), _GRID, 1)
            x += step
        y = np.ceil(min_y / step) * step
        while y <= max_y:
            py = self.to_pixels((0, y))[1]
            cv2.line(canvas, (0, py), (self.width, py), _GRID, 1)
            y += step

        if min_x <= 0 <= max_x:
            px = self.to_pixels((0, 0))[0]
            cv2.line(canvas, (px, 0), (px, self.height), _AXIS, 1)
        if min_y <= 0 <= max_y:
            py = self.to_pixels((0, 0))[1]
            cv2.line(canvas, (0, py), (self.width, py), _AXIS, 1)

        return canvas

    def render(
        self,
        positions: dict[int, tuple[float, float]],
        trails: dict[int, list[tuple[float, float]]] | None = None,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw shoppers, and optionally their recent paths, on the plan.

        `background` accepts a rectified frame so the real floor can sit under
        the markers, which is what makes a calibration error obvious at a glance
        instead of only in the numbers.
        """
        if background is not None:
            canvas = cv2.resize(background, (self.width, self.height))
            canvas = cv2.addWeighted(
                canvas, 0.55, np.full_like(canvas, _BACKGROUND, dtype=np.uint8), 0.45, 0
            )
        else:
            canvas = self._blank()

        for track_id, path in (trails or {}).items():
            points = [self.to_pixels(p) for p in path]
            if len(points) > 1:
                cv2.polylines(
                    canvas, [np.array(points, np.int32)], False, _TRAIL, 2, cv2.LINE_AA
                )

        radius = max(3, int(0.18 * self.pixels_per_unit))
        for track_id, position in positions.items():
            px, py = self.to_pixels(position)
            cv2.circle(canvas, (px, py), radius, _PERSON, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), radius + 2, (10, 10, 12), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(track_id),
                (px + radius + 3, py - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                _PERSON,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"{len(positions)} on floor  |  grid {self.grid_step:g}{self.units}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )
        return canvas
