"""Launch the FastAPI backend for the React frontend."""
import argparse
import uvicorn

parser = argparse.ArgumentParser(description="AI Photo Enhancer API Server")
parser.add_argument("--mock", action="store_true", help="Run without AI model")
parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")
parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
args = parser.parse_args()

# Import engine only if not mock
engine = None
upscale_engine = None
restore_engine = None
if not args.mock:
    from backend.engine import ReflectionRemovalEngine
    engine = ReflectionRemovalEngine()
    from backend.upscale_engine import UpscaleEngine
    upscale_engine = UpscaleEngine()
    from backend.restore_engine import RestoreEngine
    restore_engine = RestoreEngine()
    from backend.skin_retouch_engine import SkinRetouchEngine
    skin_retouch_engine = SkinRetouchEngine()
else:
    skin_retouch_engine = None

# Configure the FastAPI app
from backend.api import app, configure
configure(engine=engine, mock_mode=args.mock, upscale_engine=upscale_engine, restore_engine=restore_engine, skin_retouch_engine=skin_retouch_engine)

# Serve static frontend from web/dist if it exists (for single-port deployment)
import os
from pathlib import Path
dist_dir = Path(__file__).parent / "web" / "dist"
if dist_dir.is_dir():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi import HTTPException

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA - static files or fallback to index.html."""
        # Don't intercept API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_dir / "index.html")

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
