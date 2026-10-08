"""Unified, low-noise terminal progress renderer used by all ranker stages."""

from __future__ import annotations

import math
import os
import shutil
import sys
import threading
import time
from typing import Optional, TextIO


class ProgressBar:
    """Single-line progress bar with throttled, in-place terminal updates.

    The renderer deliberately avoids tqdm so every stage of the application uses
    the same compact format and refresh policy. Updates are throttled to avoid
    terminal flooding even when the caller processes items one-by-one.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str = "it",
        *,
        stream: Optional[TextIO] = None,
        min_interval: float = 0.5,
        bar_width: int = 24,
        enabled: bool = True,
        heartbeat: bool = True,
    ) -> None:
        self.stream = stream or sys.stdout
        self.total = max(1, int(total))
        self.desc = str(desc)
        self.unit = str(unit).strip() or "it"
        self.min_interval = max(0.1, float(min_interval))
        self.bar_width = max(10, int(bar_width))
        self.enabled = bool(enabled)
        self.n = 0
        self.postfix = ""
        self.start_time = time.monotonic()
        self.last_render = 0.0
        self.closed = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._terminal_width = 0
        self._thread = None

        if self.enabled:
            self.refresh(force=True)
            if heartbeat:
                self._thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="ranker-progress-heartbeat",
                    daemon=True,
                )
                self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.min_interval):
            self.refresh()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "--:--"
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _write(self, text: str) -> None:
        try:
            data = text.encode("utf-8", errors="replace")
            fileno = self.stream.fileno()
            os.write(fileno, data)
        except (AttributeError, OSError, ValueError):
            self.stream.write(text)
            self.stream.flush()

    def _build_line(self) -> str:
        elapsed = max(0.0, time.monotonic() - self.start_time)
        rate = self.n / elapsed if elapsed > 0 else 0.0
        fraction = min(1.0, max(0.0, self.n / self.total))
        filled = int(round(self.bar_width * fraction))
        filled = min(self.bar_width, max(0, filled))
        bar = "#" * filled + "-" * (self.bar_width - filled)
        remaining = max(0, self.total - self.n)
        eta = remaining / rate if rate > 0 else float("nan")
        rate_text = f"{rate:,.1f} {self.unit}/s" if rate > 0 else f"-- {self.unit}/s"
        postfix = f" | {self.postfix}" if self.postfix else ""
        return (
            f"{self.desc} [{bar}] {fraction * 100:5.1f}% "
            f"{self.n:,}/{self.total:,} | {self._format_duration(elapsed)} "
            f"| ETA {self._format_duration(eta)} | {rate_text}{postfix}"
        )

    def _fit_line(self, line: str) -> str:
        try:
            width = shutil.get_terminal_size((120, 20)).columns
        except (OSError, ValueError):
            width = 120
        if width <= 20 or len(line) <= width:
            return line
        return line[: max(1, width - 1)]

    def refresh(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.closed:
                return
            now = time.monotonic()
            if not force and (now - self.last_render) < self.min_interval:
                return
            line = self._fit_line(self._build_line())
            self.last_render = now
            # Always use one carriage-returned frame while the process is alive.
            # This avoids the newline flood caused by repeated progress writes.
            self._write("\r" + line + "\x1b[K")

    def update(self, value: int = 1, *, refresh: bool = True) -> None:
        with self._lock:
            if self.closed:
                return
            self.n += max(0, int(value))
        if refresh:
            self.refresh()

    def set_total(self, total: int, *, refresh: bool = True) -> None:
        with self._lock:
            if self.closed:
                return
            self.total = max(1, int(total), self.n)
        if refresh:
            self.refresh(force=True)

    def set_postfix_str(self, text: str, *, refresh: bool = True) -> None:
        with self._lock:
            if self.closed:
                return
            self.postfix = str(text)
        if refresh:
            self.refresh(force=True)

    def set_postfix(self, values, *, refresh: bool = True) -> None:
        if isinstance(values, dict):
            text = " | ".join(f"{k}={v}" for k, v in values.items())
        else:
            text = str(values)
        self.set_postfix_str(text, refresh=refresh)

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.5, self.min_interval * 2))
        with self._lock:
            # Do not fake completion. The final frame reflects the actual count.
            self._write("\r" + self._fit_line(self._build_line()) + "\x1b[K\n")
