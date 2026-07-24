"""Thread-safe command state for persistent robot deployment sessions."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(frozen=True)
class SessionSnapshot:
    """An immutable view used to decide whether a computed action is publishable."""

    mode: str
    task: str
    step: int
    generation: int


class SessionControl:
    """Serialize terminal commands and invalidate in-flight inferred actions.

    Every state transition increments ``generation``. An inference loop captures
    that value before computing an action and must match it again while holding
    this controller's lock immediately before publishing. This makes stop,
    start, continue, and exit effective even when a PI forward pass is running.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode = "disabled"
        self._task = ""
        self._step = 0
        self._generation = 0

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(self._mode, self._task, self._step, self._generation)

    def mark_enabled(self) -> bool:
        with self._lock:
            if self._mode == "exiting":
                return False
            if self._mode == "disabled":
                self._mode = "paused"
                self._generation += 1
            return True

    def begin_start(self) -> int:
        with self._lock:
            self._mode = "transition"
            self._generation += 1
            return self._generation

    def finish_start(self, generation: int, task: str) -> bool:
        with self._lock:
            if self._mode != "transition" or self._generation != generation:
                return False
            self._task = task
            self._step = 0
            self._mode = "running"
            return True

    def begin_continue(self, max_steps: int) -> tuple[str, int] | None:
        with self._lock:
            if self._mode != "paused" or not self._task:
                return None
            if max_steps > 0 and self._step >= max_steps:
                self._step = 0
            self._mode = "transition"
            self._generation += 1
            return self._task, self._generation

    def finish_continue(self, generation: int) -> bool:
        with self._lock:
            if self._mode != "transition" or self._generation != generation:
                return False
            self._mode = "running"
            return True

    def pause(self) -> bool:
        with self._lock:
            if self._mode != "running":
                return False
            self._mode = "paused"
            self._generation += 1
            return True

    def abort_transition(self, generation: int) -> None:
        with self._lock:
            if self._mode == "transition" and self._generation == generation:
                self._mode = "paused"

    def request_exit(self) -> None:
        with self._lock:
            self._mode = "exiting"
            self._generation += 1

    def publish_if_current(self, generation: int, publish: Callable[[], None]) -> bool:
        """Publish only while the captured action remains the current action."""

        with self._lock:
            if self._mode != "running" or self._generation != generation:
                return False
            publish()
            return True

    def record_published_step(self, generation: int, max_steps: int) -> bool:
        """Record one command and pause when an optional run budget is exhausted."""

        with self._lock:
            if self._mode != "running" or self._generation != generation:
                return False
            self._step += 1
            if max_steps > 0 and self._step >= max_steps:
                self._mode = "paused"
                self._generation += 1
                return True
            return False
