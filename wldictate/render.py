"""Render thread: typing off the session loop, with latest-wins coalescing.

The session loop is the pipeline's real-time core — it must service 32 ms
audio frames (VAD, decode dispatch, transform polling). But delivering text
to the screen costs a process spawn plus ~6 ms per keystroke, so a long
rewrite executed inline blocks the loop for over a second: audio backs up,
decodes are delayed, and tail latency spikes.

:class:`RenderProxy` wraps the session's emitter and owns a dedicated render
thread. Two kinds of operations cross the boundary:

- ``publish(desired)`` — non-blocking, **latest-wins**: the render thread
  syncs the screen to the newest published hypothesis. If decode N+1 lands
  while N is still being typed, N is skipped entirely — never typed and then
  backspaced away. Fewer keystrokes, less visible churn, and the session
  loop never waits.
- everything else (finalize syncs, voice commands, transform applies,
  commit-mode appends) is a **barrier**: it discards any pending publish
  (the barrier's own write supersedes it), runs on the render thread in
  submission order, and returns the result to the caller.

All emitter/device state is touched only by the render thread, so the
wrapped emitter needs no locking of its own. The proxy exposes the same
surface as the emitters it wraps; ``wrapped`` unwraps for type checks.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from .emitter import Emitter


class _Job:
    """One barrier operation: a callable plus its completion/result."""

    __slots__ = ("fn", "result", "exc", "done", "wait_result")

    def __init__(self, fn: Callable, wait_result: bool = True) -> None:
        self.fn = fn
        self.result = None
        self.exc: BaseException | None = None
        self.done = threading.Event()
        self.wait_result = wait_result

    def run(self) -> None:
        try:
            self.result = self.fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in wait()
            self.exc = e
        finally:
            self.done.set()

    def wait(self):
        self.done.wait()
        if self.exc is not None:
            raise self.exc
        return self.result


class RenderProxy(Emitter):
    """Emitter proxy executing all device I/O on a private render thread."""

    def __init__(self, emitter: Emitter, on_error: Callable[[str], None] | None = None) -> None:
        self._wrapped = emitter
        self._on_error = on_error
        self._cv = threading.Condition()
        # Newest published (desired_text, max_backspaces) — latest wins.
        self._pending: tuple[str, int | None] | None = None
        self._jobs: deque[_Job] = deque()
        self._closed = False
        self._publish_failing = False  # dedupe publish-failure reports
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="render"
        )
        self._thread.start()

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def wrapped(self) -> Emitter:
        """The wrapped emitter (for type checks; do NOT call it directly)."""
        return self._wrapped

    # ── Async path (tentative hypothesis renders) ────────────────────────

    def publish(self, desired: str, *, max_backspaces: int | None = None) -> None:
        """Latest-wins screen sync; never blocks the caller.

        ``max_backspaces`` overrides the wrapped emitter's default cap (the
        streamed-transform apply needs the full-replacement budget).

        Failures freeze the wrapped CorrectingEmitter (screen unknown) and
        are reported once via ``on_error``; the next barrier ``sync`` then
        observes the frozen state and returns False, so callers that need
        certainty (finalize, transforms) still get it.
        """
        with self._cv:
            if self._closed:
                return
            self._pending = (desired, max_backspaces)
            self._cv.notify()

    def flush(self) -> None:
        """Barrier no-op: returns once all prior work has been applied."""
        self._call(lambda: None)

    # ── Barrier plumbing ─────────────────────────────────────────────────

    def _call(self, fn: Callable, *, supersede: bool = False):
        """Run ``fn`` on the render thread; block for (and return) its result.

        ``supersede``: the operation writes authoritative screen state (a
        finalize/transform sync), so a not-yet-rendered tentative hypothesis
        is discarded instead of being typed first. Non-superseding barriers
        (flush, reads, region ops) let the pending publish render first —
        the loop applies pending publishes before jobs, which is safe
        because every barrier blocks the one publishing thread: a publish
        can never be *newer* than a queued barrier.
        """
        job = _Job(fn)
        with self._cv:
            if self._closed:
                raise RuntimeError("render thread is closed")
            if supersede:
                self._pending = None
            self._jobs.append(job)
            self._cv.notify()
        return job.wait()

    def _post(self, fn: Callable) -> None:
        """Queue ``fn`` in order without waiting (append-only commit path).

        Unlike publishes these are never coalesced — every append must land.
        Errors are reported via ``on_error`` by the job itself.
        """
        job = _Job(fn, wait_result=False)
        with self._cv:
            if self._closed:
                return
            self._jobs.append(job)
            self._cv.notify()

    # ── Render thread ────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not (
                    self._jobs or self._pending is not None or self._closed
                ):
                    self._cv.wait()
                # Pending publish first: any job still queued alongside it is
                # a non-superseding barrier that arrived after the publish.
                if self._pending is not None:
                    job, pending = None, self._pending
                    self._pending = None
                elif self._jobs:
                    job, pending = self._jobs.popleft(), None
                else:  # closed, queue drained
                    return
            if job is not None:
                job.run()
            else:
                self._do_publish(*pending)

    def _do_publish(self, desired: str, max_backspaces: int | None = None) -> None:
        try:
            ok = bool(self._wrapped.sync(desired, max_backspaces=max_backspaces))
        except Exception as e:  # defensive: a device bug must not kill the thread
            ok = False
            self._error(f"render failed: {e}")
        if ok:
            self._publish_failing = False
        elif not self._publish_failing:
            self._publish_failing = True
            self._error("emitter failed to sync tentative text")

    def _error(self, msg: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(msg)
            except Exception:
                pass

    # ── Forwarded emitter surface (all barriers) ─────────────────────────

    def emit(self, text: str) -> bool:
        """Ordered async append (commit mode): queued, errors via on_error."""

        def run() -> None:
            try:
                if not self._wrapped.emit(text):
                    self._error("emitter failed to type committed text")
            except Exception as e:
                self._error(f"emitter failed to type committed text: {e}")

        self._post(run)
        return True

    def sync(
        self,
        desired: str,
        *,
        max_backspaces: int | None = None,
        bulk: bool = False,
    ) -> bool:
        return self._call(
            lambda: self._wrapped.sync(
                desired, max_backspaces=max_backspaces, bulk=bulk
            ),
            supersede=True,
        )

    def rewrite(self, backspaces: int, text: str) -> str | None:
        return self._call(
            lambda: self._wrapped.rewrite(backspaces, text), supersede=True
        )

    def rewrite_bulk(self, backspaces: int, text: str) -> str | None:
        return self._call(
            lambda: self._wrapped.rewrite_bulk(backspaces, text), supersede=True
        )

    def press_key(self, keysym: str) -> bool:
        return self._call(lambda: self._wrapped.press_key(keysym))

    def set_clipboard(self, text: str) -> bool:
        return self._call(lambda: self._wrapped.set_clipboard(text))

    def begin_utterance(self, carry: bool = False) -> None:
        return self._call(lambda: self._wrapped.begin_utterance(carry))

    def merge_previous(self) -> bool:
        return self._call(lambda: self._wrapped.merge_previous())

    def reset_regions(self) -> None:
        return self._call(lambda: self._wrapped.reset_regions())

    @property
    def logical(self) -> str:
        return self._call(lambda: self._wrapped.logical)

    @property
    def previous_logical(self) -> str:
        return self._call(lambda: self._wrapped.previous_logical)

    @property
    def previous_len(self) -> int:
        return self._call(lambda: self._wrapped.previous_len)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Drain queued work, stop the thread, close the wrapped emitter."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._cv.notify()
        self._thread.join(timeout=10.0)
        try:
            self._wrapped.close()
        except Exception:
            pass
