"""The store on one page: fixtures on the floor plan, coloured by what happened.

A table of zones answers "which shelf". A plan answers "where in the store",
which is the question a person walking the floor actually has. This draws the
store map's fixtures in plan coordinates, colours each by its finding, and puts
foot traffic underneath when the session recorded floor positions.

It is an SVG, written as text. No pixels from any camera are involved, and the
traffic layer is a count of position samples per floor cell with no track
attached to it, so the drawing holds nothing about any individual.

The plan is drawn y-up, the way `floorview.py` draws it and the way a floor
plan reads.
"""

from __future__ import annotations

from collections import Counter
from xml.sax.saxutils import escape

from openv.products import ProductAnalysis, Shelf

# The same tokens as web/static/theme.css, written out because an SVG that is
# handed to somebody has no stylesheet to read them from.
_BG = "#07090b"
_PANEL = "#0c0f13"
_LINE = "#2a323d"
_FG = "#dfe5ec"
_DIM = "#8492a3"
_ACCENT = "#6fd6ff"
_FILL = {
    "high": "#ff5a5f",
    "medium": "#ffb224",
    "low": "#3ddc97",
    "none": "#6b7785",
}
_UNWATCHED = "#39424d"
_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
_WIDTH = 960.0
_MARGIN = 28.0
_GRID = 48  # traffic cells across the longer side


def _extent(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _label(shelf: Shelf | None) -> str:
    if shelf is None:
        return "not measured this session"
    funnel = shelf.finding.funnel
    if not funnel.has_confidence:
        return f"{funnel.passed} passed, not a rate yet"
    return f"{funnel.reached} of {funnel.passed} reached ({funnel.reach_rate * 100:.0f}%)"


def render_plan(
    products: ProductAnalysis,
    paths: dict[int, list[tuple[float, float, float]]] | None = None,
) -> str:
    """An SVG of the store plan. Raises ValueError when nothing has a plan shape."""
    store_map = products.store_map
    drawn = [f for f in store_map.fixtures if f.plan is not None]
    if not drawn and store_map.outline is None:
        raise ValueError(
            "the store map has no 'outline' and no fixture has a 'plan' polygon, "
            "so there is nothing to draw. Add plan coordinates to the map"
        )

    samples = [(x, y) for track in (paths or {}).values() for _, x, y in track]
    points = [p for f in drawn for p in f.plan]
    if store_map.outline is not None:
        points += list(store_map.outline)
    else:
        # With no outline the traffic decides how much floor is shown.
        points += samples
    min_x, min_y, max_x, max_y = _extent(points)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = (_WIDTH - 2 * _MARGIN) / span_x
    height = span_y * scale + 2 * _MARGIN

    def px(point: tuple[float, float]) -> tuple[float, float]:
        return (
            _MARGIN + (point[0] - min_x) * scale,
            _MARGIN + (max_y - point[1]) * scale,
        )

    def path(polygon) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in map(px, polygon))

    by_zone = {s.zone: s for s in products.shelves}
    legend_y = height + 18
    total_height = height + 64
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH:.0f} '
        f'{total_height:.0f}" font-family="{_FONT}" font-size="11">',
        f'<rect width="{_WIDTH:.0f}" height="{total_height:.0f}" fill="{_BG}"/>',
    ]

    if store_map.outline is not None:
        out.append(
            f'<polygon points="{path(store_map.outline)}" fill="{_PANEL}" '
            f'stroke="{_FG}" stroke-opacity="0.6" stroke-width="1.25"/>'
        )

    # Traffic: position samples per floor cell. Positions are sampled at a
    # fixed rate per shopper, so a cell's count is seconds of presence there.
    if samples:
        cell = max(span_x, span_y) / _GRID
        cells = Counter(
            (int((x - min_x) // cell), int((y - min_y) // cell))
            for x, y in samples
            if min_x <= x <= max_x and min_y <= y <= max_y
        )
        peak = max(cells.values(), default=0)
        for (cx, cy), count in sorted(cells.items()):
            left, top = px((min_x + cx * cell, min_y + (cy + 1) * cell))
            out.append(
                f'<rect x="{left:.1f}" y="{top:.1f}" width="{cell * scale:.1f}" '
                f'height="{cell * scale:.1f}" fill="{_ACCENT}" '
                f'fill-opacity="{0.05 + 0.5 * count / peak:.2f}"/>'
            )

    for fixture in drawn:
        shelf = by_zone.get(fixture.zone)
        fill = _FILL[shelf.finding.severity] if shelf is not None else _UNWATCHED
        dash = "" if shelf is not None else ' stroke-dasharray="5 4"'
        out.append(
            f'<polygon points="{path(fixture.plan)}" fill="{fill}" fill-opacity="0.3" '
            f'stroke="{fill}" stroke-width="1.25"{dash}>'
            f"<title>{escape(fixture.label or fixture.zone)}: {escape(_label(shelf))}. "
            f"{escape(', '.join(fixture.brands) or 'nothing mapped')}</title></polygon>"
        )
        fx0, fy0, fx1, fy1 = _extent(list(map(px, fixture.plan)))
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        # Long thin fixtures get their label turned to run along them.
        turn = f' transform="rotate(-90 {cx:.1f} {cy:.1f})"' if fy1 - fy0 > 1.6 * (fx1 - fx0) else ""
        out.append(
            f'<text x="{cx:.1f}" y="{cy - 3:.1f}" text-anchor="middle" '
            f'letter-spacing="0.8" fill="{_FG}"{turn}>'
            f"{escape((fixture.label or fixture.zone).upper())}</text>"
        )
        out.append(
            f'<text x="{cx:.1f}" y="{cy + 12:.1f}" text-anchor="middle" '
            f'fill="{fill if shelf is not None else _DIM}"{turn}>{escape(_label(shelf))}</text>'
        )

    x = _MARGIN
    for text, colour in (
        ("high", _FILL["high"]),
        ("medium", _FILL["medium"]),
        ("at or above store median", _FILL["low"]),
        ("not a rate yet", _FILL["none"]),
        ("not measured", _UNWATCHED),
    ):
        out.append(f'<rect x="{x:.0f}" y="{legend_y:.0f}" width="12" height="12" fill="{colour}"/>')
        out.append(f'<text x="{x + 17:.0f}" y="{legend_y + 10:.0f}" fill="{_DIM}">{text}</text>')
        x += 32 + 6.8 * len(text)
    if samples:
        out.append(f'<rect x="{x:.0f}" y="{legend_y:.0f}" width="12" height="12" fill="{_ACCENT}" fill-opacity="0.55"/>')
        out.append(f'<text x="{x + 17:.0f}" y="{legend_y + 10:.0f}" fill="{_DIM}">foot traffic</text>')

    site = store_map.name or store_map.site
    out.append(
        f'<text x="{_MARGIN:.0f}" y="{legend_y + 34:.0f}" fill="{_DIM}">'
        f"{escape(site)}, session {products.session_id}, "
        f"{products.analysis.total_shoppers} shoppers observed. Plan in "
        f"{escape(store_map.units)}.</text>"
    )
    out.append("</svg>")
    return "\n".join(out)
