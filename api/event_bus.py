"""Thread-safe event bus for cross-component communication."""

import asyncio
import threading
from typing import Any, Dict, List, Tuple


class EventBus:
    """Publish/subscribe with thread-safe publish and async subscribers.

    Subscribers receive events via asyncio.Queue — safe to use from FastAPI routes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each subscriber is a tuple of (optional event_type filter, asyncio.Queue)
        self._subscribers: List[Tuple[str | None, asyncio.Queue]] = []

    def publish(self, event: Dict[str, Any]) -> None:
        """Thread-safe broadcast to all matching subscribers."""
        event_type = event.get("type", "")
        with self._lock:
            for event_filter, queue in self._subscribers:
                if event_filter is None or event_filter == event_type:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass

    def subscribe(self, event_type: str | None = None) -> Tuple[asyncio.Queue, int]:
        """Register a subscriber. Returns (queue, subscription_id)."""
        with self._lock:
            sid = len(self._subscribers)
            queue: asyncio.Queue = asyncio.Queue(maxsize=500)
            self._subscribers.append((event_type, queue))
            return queue, sid

    def unsubscribe(self, sid: int) -> None:
        """Remove a subscriber by ID."""
        with self._lock:
            if 0 <= sid < len(self._subscribers):
                del self._subscribers[sid]
