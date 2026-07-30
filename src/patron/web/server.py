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

from patron.web.engine import LiveEngine

STATIC = Path(__file__).parent / "static"


def create_app(engine: LiveEngine) -> FastAPI:
    app = FastAPI(title="Patron live", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _start() -> None:
        engine.start()

    @app.on_event("shutdown")
    def _stop() -> None:
        engine.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

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

    @app.post("/api/zones")
    def post_zones(payload: dict[str, Any]) -> JSONResponse:
        zones = payload.get("zones", [])
        try:
            engine.set_zones(zones)
        except (KeyError, ValueError, TypeError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "count": len(zones)})

    return app
