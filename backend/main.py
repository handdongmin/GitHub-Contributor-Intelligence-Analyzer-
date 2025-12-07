from fastapi import FastAPI
from backend.core.config import get_settings
from backend.routers import analyze, fetch, ml

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="GitHub Contributor Intelligence Analyzer", version="0.1.0")
    app.state.settings = settings
    app.include_router(fetch.router)
    app.include_router(analyze.router)
    app.include_router(ml.router)
    return app

app = create_app()