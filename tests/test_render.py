"""RenderProxy: latest-wins coalescing + barrier ordering off the session loop."""

import threading
import time

import numpy as np

from wldictate.emitter import CorrectingEmitter, Emitter
from wldictate.render import RenderProxy
from wldictate.streaming import SAMPLE_RATE, StreamingSession
from wldictate.textproc import TextFormatter
from wldictate.transcriber import FakeTranscriber, Word


class ScreenDevice(Emitter):
    """Models the display: applies backspaces/text, records every op."""

    def __init__(self):
        self.ops = []
        self.screen = ""
        self.fail = False

    def emit(self, text):
        return self.rewrite(0, text) is not None

    def rewrite(self, backspaces, text):
        if self.fail:
            return None
        self.ops.append((backspaces, text))
        keep = len(self.screen) - backspaces
        self.screen = self.screen[:keep] + text
        return text


class GatedDevice(ScreenDevice):
    """Blocks inside the first rewrite until released — deterministic way to
    pile up publishes behind an in-progress render."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._gated_once = False

    def rewrite(self, backspaces, text):
        if not self._gated_once:
            self._gated_once = True
            self.entered.set()
            assert self.release.wait(timeout=5.0)
        return super().rewrite(backspaces, text)


def make_proxy(device=None, errors=None):
    device = device if device is not None else ScreenDevice()
    proxy = RenderProxy(
        CorrectingEmitter(device),
        on_error=(errors.append if errors is not None else None),
    )
    return proxy, device


def test_publish_renders_latest_state():
    proxy, device = make_proxy()
    proxy.begin_utterance()
    proxy.publish("hello")
    proxy.flush()
    assert device.screen == "hello"
    proxy.close()


def test_publishes_coalesce_to_newest_while_render_in_progress():
    device = GatedDevice()
    proxy, _ = make_proxy(device)
    proxy.begin_utterance()
    proxy.publish("h1")
    assert device.entered.wait(timeout=5.0)  # render thread is typing h1
    proxy.publish("h2 stale")
    proxy.publish("h3 newest")  # supersedes h2 before it was ever typed
    device.release.set()
    proxy.flush()
    # h1 (in progress) and h3 landed; h2 was skipped entirely.
    assert device.screen == "h3 newest"
    typed = "".join(text for _, text in device.ops)
    assert "stale" not in typed
    proxy.close()


def test_barrier_supersedes_pending_publish():
    device = GatedDevice()
    proxy, _ = make_proxy(device)
    proxy.begin_utterance()
    proxy.publish("first")
    assert device.entered.wait(timeout=5.0)
    proxy.publish("tentative never lands")
    releaser = threading.Timer(0.05, device.release.set)
    releaser.start()
    assert proxy.sync("final text") is True  # barrier: drops the pending publish
    assert device.screen == "final text"
    typed = "".join(text for _, text in device.ops)
    assert "never lands" not in typed
    proxy.close()


def test_publish_failure_reported_once_and_freezes_emitter():
    errors = []
    proxy, device = make_proxy(errors=errors)
    proxy.begin_utterance()
    device.fail = True
    proxy.publish("a")
    proxy.flush()
    proxy.publish("ab")
    proxy.flush()
    assert len(errors) == 1  # deduped
    # The barrier sync observes the frozen CorrectingEmitter.
    assert proxy.sync("anything") is False
    proxy.close()


def test_emit_appends_in_order_without_blocking():
    proxy, device = make_proxy()
    proxy.begin_utterance()
    assert proxy.emit("a") is True
    assert proxy.emit("b") is True
    proxy.flush()
    assert device.screen == "ab"
    proxy.close()


def test_emit_failure_reports_via_on_error():
    errors = []
    proxy, device = make_proxy(errors=errors)
    proxy.begin_utterance()
    device.fail = True
    proxy.emit("x")
    proxy.flush()
    assert errors
    proxy.close()


def test_forwarded_state_reads():
    proxy, device = make_proxy()
    proxy.begin_utterance()
    proxy.sync("one")
    assert proxy.logical == "one"
    proxy.begin_utterance()
    assert proxy.previous_logical == "one"
    assert proxy.previous_len == 3
    assert proxy.merge_previous() is True
    assert proxy.logical == "one"
    proxy.close()


def test_close_drains_queued_work():
    proxy, device = make_proxy()
    proxy.begin_utterance()
    for i in range(5):
        proxy.emit(str(i))
    proxy.close()
    assert device.screen == "01234"


def test_wrapped_exposes_inner_emitter():
    proxy, _ = make_proxy()
    assert isinstance(proxy.wrapped, CorrectingEmitter)
    proxy.close()


# ── Integration: StreamingSession renders through the proxy ─────────────────


def W(text, start, end, prob=0.9):
    return Word(text, start, end, prob)


def test_streaming_session_publishes_through_proxy():
    script = [
        [W(" the", 0, 0.2), W(" quit", 0.2, 0.5)],
        [W(" the", 0, 0.2), W(" quick", 0.2, 0.5)],
    ]
    device = ScreenDevice()
    proxy = RenderProxy(CorrectingEmitter(device))
    t = [0.0]
    session = StreamingSession(
        FakeTranscriber(script),
        TextFormatter(capitalize_sentences=False),
        proxy,
        infer_interval_s=0.5,
        min_new_audio_s=0.1,
        min_speech_s=0.1,
        correcting=True,
        clock=lambda: t[0],
    )
    session.start_utterance()
    chunk = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
    for _ in range(20):
        session.feed([chunk])
        t[0] += 0.1
        session.tick()
        deadline = time.monotonic() + 2.0
        while session._inflight is not None and time.monotonic() < deadline:
            if session._inflight[0].done():
                break
            time.sleep(0.001)
        session.tick()
        proxy.flush()  # deterministic: apply each hypothesis before the next
    final = session.finalize()
    assert device.screen == "the quick"
    assert final == "the quick"
    session.stop()
    proxy.close()
