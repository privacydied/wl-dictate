"""Capture manager: persistent streams must survive device index drift."""

from wldictate.worker import _CaptureManager


class FakeCapture:
    instances = []

    def __init__(self, device):
        self.device = device
        self.device_name = f"name-of-{device}"
        self.active = False
        self.flushed = 0
        self.stopped = False
        FakeCapture.instances.append(self)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        self.stopped = True

    def flush(self):
        self.flushed += 1


def make_manager(monkeypatch, persistent=True):
    FakeCapture.instances = []
    monkeypatch.setattr("wldictate.worker.AudioCapture", FakeCapture)
    return _CaptureManager(persistent=persistent)


def test_reuse_when_name_matches_despite_index_drift(monkeypatch):
    mgr = make_manager(monkeypatch)
    first = mgr.acquire(17, "C920")
    first.device_name = "C920"  # what the open stream reports
    # Pulse indices drifted: same mic, new index. Must NOT reopen.
    second = mgr.acquire(25, "C920")
    assert second is first
    assert first.flushed == 1
    assert not first.stopped


def test_reopen_when_name_actually_changes(monkeypatch):
    mgr = make_manager(monkeypatch)
    first = mgr.acquire(17, "C920")
    first.device_name = "C920"
    second = mgr.acquire(3, "MG-XU")
    assert second is not first
    assert first.stopped


def test_index_compare_used_when_no_name(monkeypatch):
    mgr = make_manager(monkeypatch)
    first = mgr.acquire(5, None)
    again = mgr.acquire(5, None)
    assert again is first
    other = mgr.acquire(6, None)
    assert other is not first and first.stopped


def test_non_persistent_releases(monkeypatch):
    mgr = make_manager(monkeypatch, persistent=False)
    cap = mgr.acquire(1, "mic")
    mgr.release()
    assert cap.stopped


def test_persistent_release_keeps_stream(monkeypatch):
    mgr = make_manager(monkeypatch, persistent=True)
    cap = mgr.acquire(1, "mic")
    mgr.release()
    assert not cap.stopped
    mgr.shutdown()
    assert cap.stopped


def test_dead_stream_reopened(monkeypatch):
    mgr = make_manager(monkeypatch)
    cap = mgr.acquire(1, "mic")
    cap.active = False  # stream died underneath
    again = mgr.acquire(1, "mic")
    assert again is not cap