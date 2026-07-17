from wldictate import ipc


def test_command_roundtrip():
    line = ipc.format_command("start", 3)
    cmd = ipc.parse_command(line)
    assert cmd == ipc.Command("start", 3)


def test_command_without_device():
    assert ipc.parse_command(ipc.format_command("stop")) == ipc.Command("stop")


def test_command_junk_tolerated():
    assert ipc.parse_command("") is None
    assert ipc.parse_command("hello world") is None
    assert ipc.parse_command('{"cmd": "reboot"}') is None
    assert ipc.parse_command('{"cmd": 5}') is None
    assert ipc.parse_command("{broken json") is None
    assert ipc.parse_command("[1, 2]") is None


def test_command_bad_device_dropped():
    cmd = ipc.parse_command('{"cmd": "start", "device": "abc"}')
    assert cmd == ipc.Command("start", None)


def test_command_device_name_roundtrip():
    line = ipc.format_command("start", 17, "HD Pro Webcam C920 Analog Stereo")
    cmd = ipc.parse_command(line)
    assert cmd == ipc.Command("start", 17, "HD Pro Webcam C920 Analog Stereo")


def test_command_bad_device_name_dropped():
    cmd = ipc.parse_command('{"cmd": "start", "device": 3, "device_name": 42}')
    assert cmd == ipc.Command("start", 3, None)


def test_event_roundtrip():
    line = ipc.format_event("commit", text="hello world")
    ev = ipc.parse_event(line)
    assert ev is not None
    assert ev.ev == "commit"
    assert ev.text == "hello world"


def test_event_junk_returns_none():
    assert ipc.parse_event("Loading model...") is None
    assert ipc.parse_event('{"ev": "unknown-event"}') is None
    assert ipc.parse_event("") is None


def test_event_non_string_fields_dropped():
    ev = ipc.parse_event('{"ev": "error", "msg": 42}')
    assert ev is not None and ev.msg is None


def test_command_mode_round_trip():
    line = ipc.format_command("start", 3, "Mic", mode="contextual")
    cmd = ipc.parse_command(line)
    assert cmd.mode == "contextual"
    assert cmd.device == 3


def test_command_mode_defaults_and_junk():
    assert ipc.parse_command('{"cmd": "start"}').mode == "standard"
    assert ipc.parse_command('{"cmd": "start", "mode": "chaos"}').mode == "standard"
    # mode key omitted from the wire format when not set (old-worker compat)
    assert '"mode"' not in ipc.format_command("start")
