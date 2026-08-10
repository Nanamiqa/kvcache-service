"""Command line entry point."""

from __future__ import annotations

import math

import uvicorn

from .app import create_app
from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        proxy_headers=True,
        timeout_graceful_shutdown=math.ceil(settings.shutdown_grace_seconds),
    )


if __name__ == "__main__":
    main()
