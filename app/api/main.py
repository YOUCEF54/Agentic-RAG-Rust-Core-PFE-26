"""Modular API entrypoint for the new project structure.

Current scope:
- Includes migrated `health` and `documents` routes.
- Additional endpoints will be moved incrementally from root `main.py`.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.hardware import router as hardware_router
from app.api.routes.health import router as health_router
from app.api.routes.index import router as index_router
from app.api.routes.query import router as query_router
from app.api.routes.stream import router as stream_router
from app.core import config

app = FastAPI(title=config.APP_NAME)

app.add_middleware(CORSMiddleware, **config.get_cors_settings())

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(hardware_router)
app.include_router(index_router)
app.include_router(query_router)
app.include_router(stream_router)

__all__ = ["app"]
