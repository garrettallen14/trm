"""
Dashboard client for sending metrics from training.

Can work in two modes:
1. In-process: Direct updates to metrics_store (when server runs in same process)
2. HTTP: Send metrics via HTTP to separate dashboard server

Usage:
    from dashboard.client import DashboardClient
    
    client = DashboardClient(mode="inprocess")  # or mode="http", url="http://localhost:3000"
    client.update(loss=0.5, epoch=1)
    client.log_epoch(epoch=1, train_loss=0.5, val_accuracy=0.35)
"""

import time
from typing import Optional
import threading


class DashboardClient:
    """Client for sending metrics to the dashboard."""
    
    def __init__(self, mode: str = "inprocess", url: str = "http://localhost:3000"):
        self.mode = mode
        self.url = url
        self.start_time = time.time()
        self._last_epoch_time = time.time()
        
        if mode == "inprocess":
            from .server import metrics_store
            self.metrics_store = metrics_store
        elif mode == "http":
            import urllib.request
            import json
            self._urllib = urllib
            self._json = json
    
    def update(self, **kwargs):
        """Update current metrics."""
        # Add elapsed time
        kwargs["elapsed_seconds"] = time.time() - self.start_time
        
        if self.mode == "inprocess":
            self.metrics_store.update(**kwargs)
        else:
            self._http_post("/api/metrics", kwargs)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, val_accuracy: Optional[float] = None):
        """Log completed epoch to history."""
        # Calculate ETA
        epoch_time = time.time() - self._last_epoch_time
        self._last_epoch_time = time.time()
        
        current_metrics = self.get_current()
        remaining_epochs = current_metrics.get("total_epochs", 100) - epoch
        eta = epoch_time * remaining_epochs
        
        # Update current state
        self.update(
            epoch=epoch,
            loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            eta_seconds=eta
        )
        
        # Add to history
        if self.mode == "inprocess":
            self.metrics_store.add_epoch_to_history(epoch, train_loss, val_loss, val_accuracy)
        else:
            self._http_post("/api/epoch", {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
    
    def get_current(self) -> dict:
        """Get current metrics."""
        if self.mode == "inprocess":
            return self.metrics_store.get()
        else:
            return {}  # HTTP mode doesn't support this yet
    
    def _http_post(self, endpoint: str, data: dict):
        """Send HTTP POST request."""
        try:
            url = self.url + endpoint
            req = self._urllib.request.Request(
                url,
                data=self._json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            self._urllib.request.urlopen(req, timeout=1)
        except Exception as e:
            pass  # Silently ignore dashboard errors


def create_dashboard_callback(client: DashboardClient):
    """Create a callback function for the training loop."""
    def callback(
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        lr: float,
        n_recursions: int,
        attn_entropy: float,
        samples_per_sec: float,
        memory_gb: float
    ):
        client.update(
            status="running",
            epoch=epoch,
            step=step,
            total_steps=total_steps,
            loss=loss,
            lr=lr,
            n_recursions=n_recursions,
            attn_entropy=attn_entropy,
            samples_per_sec=samples_per_sec,
            memory_gb=memory_gb
        )
    
    return callback
