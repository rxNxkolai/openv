"""HTTP surface for the live console.

MJPEG for the video because it works in every browser with no client library and
no negotiation. Stats are polled as JSON rather than pushed, which is a deliberate
simplification: at one update a second there is nothing a socket would buy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from openv.web.engine import LiveEngine

STATIC = Path(__file__).parent / "static"

_STATIC_TYPES = {
    ".css": "text/css",
    ".js": "text/javascript",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
}


def mount_static(app: FastAPI) -> None:
    """Serve the theme, the map renderer and, when present, a local brand layer.

    `brand.css` and `fonts/` are not part of the open-source release. When a
    deployment drops them in beside `theme.css` they are served and loaded
    after it, so a brand can restyle every page by overriding tokens and
    nothing here has to know what it looks like.
    """
    from fastapi.responses import Response

    def serve(path: Path) -> Response:
        # Resolved and checked against the static directory, so a name with
        # dots in it cannot walk out to the rest of the disk.
        resolved = path.resolve()
        kind = _STATIC_TYPES.get(resolved.suffix)
        inside = STATIC.resolve() in resolved.parents
        if kind is None or not inside or not resolved.is_file():
            return Response(status_code=404)
        return Response(resolved.read_bytes(), media_type=kind)

    @app.get("/static/{name}")
    def static_file(name: str) -> Response:
        return serve(STATIC / name)

    @app.get("/static/fonts/{name}")
    def static_font(name: str) -> Response:
        return serve(STATIC / "fonts" / name)


def console_page() -> str:
    """The live console, with the shared theme linked ahead of its own rules."""
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    links = '<link rel="stylesheet" href="/static/theme.css">'
    if (STATIC / "brand.css").exists():
        links += '\n<link rel="stylesheet" href="/static/brand.css">'
    return page.replace("<!--THEME-->", links)


def create_app(engine: LiveEngine) -> FastAPI:
    app = FastAPI(title="OpenV live", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _start() -> None:
        engine.start()

    @app.on_event("shutdown")
    def _stop() -> None:
        engine.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return console_page()

    @app.get("/stream")
    def stream() -> StreamingResponse:
        def frames():
            blank_waits = 0
            while True:
                jpeg = engine.latest_jpeg()
                if jpeg is None:
                    # Camera warmup or a load failure. Back off rather than spin.
                    blank_waits += 1
                    time.sleep(0.1)
                    if blank_waits > 600:
                        break
                    continue
                blank_waits = 0
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                time.sleep(1 / 30)

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/stats")
    def stats() -> JSONResponse:
        return JSONResponse(engine.stats())

    @app.get("/api/zones")
    def get_zones() -> JSONResponse:
        return JSONResponse(engine.zones_payload())

    @app.get("/api/floor")
    def get_floor() -> JSONResponse:
        # Separate from /api/stats because it is polled at a different rate and
        # is empty on any camera without a calibration.
        return JSONResponse(engine.floor_payload())

    @app.get("/api/scene")
    def get_scene() -> JSONResponse:
        # The store this camera is in, for the 3D map: fixtures from the store
        # map and this camera's patch of floor. Static for the life of a run.
        return JSONResponse(engine.scene_payload())

    @app.post("/api/zones")
    def post_zones(payload: dict[str, Any]) -> JSONResponse:
        zones = payload.get("zones", [])
        try:
            engine.set_zones(zones)
        except (KeyError, ValueError, TypeError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "count": len(zones)})

    mount_static(app)
    return app
