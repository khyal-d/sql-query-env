# Multi-mode deployment entry point — re-exports the main FastAPI app.
# Allows running: uvicorn server.app:app
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, main  # noqa: F401
