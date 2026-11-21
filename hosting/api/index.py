"""Vercel Serverless Function Entrypoint for FastAPI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root (hosting folder) to sys.path so modules and models can be found
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import app

# Vercel natively serves ASGI 'app'
# Mangum handler provided for compatibility with standard serverless lambda runtimes
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = app
