"""
Live Training Dashboard Server.

Runs on port 3000 and provides:
- Real-time training metrics via Server-Sent Events
- REST API for current state
- Static dashboard UI

Usage:
    uv run python dashboard/server.py &
    uv run python scripts/train.py --config configs/a40_full.yaml --dashboard
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import threading

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


# Global state for metrics (thread-safe)
class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            "status": "waiting",
            "epoch": 0,
            "total_epochs": 100,
            "step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "final_loss": 0.0,
            "val_loss": None,
            "val_accuracy": None,
            "attn_entropy": 0.0,
            "lr": 0.0,
            "samples_per_sec": 0.0,
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "n_recursions": 0,
            "memory_gb": 0.0,
            "history": {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
            }
        }
        self._subscribers = []
    
    def update(self, **kwargs):
        with self._lock:
            self._metrics.update(kwargs)
            # Notify subscribers
            for queue in self._subscribers:
                try:
                    queue.put_nowait(self._metrics.copy())
                except:
                    pass
    
    def get(self):
        with self._lock:
            return self._metrics.copy()
    
    def add_epoch_to_history(self, epoch: int, train_loss: float, val_loss: Optional[float], val_accuracy: Optional[float]):
        with self._lock:
            self._metrics["history"]["epochs"].append(epoch)
            self._metrics["history"]["train_loss"].append(train_loss)
            self._metrics["history"]["val_loss"].append(val_loss)
            self._metrics["history"]["val_accuracy"].append(val_accuracy)
    
    def subscribe(self):
        queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue):
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


# Global store
metrics_store = MetricsStore()

# FastAPI app
app = FastAPI(title="TRM Training Dashboard")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics."""
    return metrics_store.get()


@app.post("/api/metrics")
async def update_metrics(request: Request):
    """Update metrics (called by training script)."""
    data = await request.json()
    metrics_store.update(**data)
    return {"status": "ok"}


@app.post("/api/epoch")
async def log_epoch(request: Request):
    """Log completed epoch."""
    data = await request.json()
    metrics_store.add_epoch_to_history(
        epoch=data["epoch"],
        train_loss=data["train_loss"],
        val_loss=data.get("val_loss"),
        val_accuracy=data.get("val_accuracy")
    )
    return {"status": "ok"}


@app.get("/api/stream")
async def stream_metrics():
    """Server-Sent Events stream for live updates."""
    async def event_generator():
        queue = metrics_store.subscribe()
        try:
            # Send initial state
            yield f"data: {json.dumps(metrics_store.get())}\n\n"
            
            while True:
                try:
                    # Wait for updates with timeout
                    metrics = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(metrics)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f": heartbeat\n\n"
        finally:
            metrics_store.unsubscribe(queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


def run_server(host: str = "0.0.0.0", port: int = 3000):
    """Run the dashboard server."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_server_background(host: str = "0.0.0.0", port: int = 3000):
    """Run the dashboard server in a background thread."""
    thread = threading.Thread(target=run_server, args=(host, port), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    print("Starting TRM Dashboard on http://0.0.0.0:3000")
    run_server()
