"""A/B eval for contextual-transform models.

Runs a fixed set of dictation-transform cases against one or more contextual
profiles and prints outputs side by side with latency. Use it to measure a
model/prompt change instead of guessing.

Run:  .venv/bin/python utils/eval_transform.py [profile ...]
      (default: the active profile from config.json)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wldictate.config import Config  # noqa: E402
from wldictate.transform import ScreenContext, Transformer  # noqa: E402
import wldictate.transform as tm  # noqa: E402

# (name, context, history, transcript)
CASES = [
    (
        "cleanup",
        ScreenContext(window_class="kitty", window_title="~ — zsh"),
        (),
        "so basically i recieve teh package yesterday and its all good now",
    ),
    (
        "instruction->command",
        ScreenContext(
            window_class="kitty",
            clipboard="ERROR: could not bind to 0.0.0.0:8080: address already in use",
        ),
        (),
        "give me the command to find what's using that port",
    ),
    (
        "reply-to-selection",
        ScreenContext(
            window_class="vesktop",
            window_title="#dev — Discord",
            selection="hey can you review my PR today? it's the auth refactor, ~400 lines",
        ),
        (),
        "reply to this say i can do it after lunch and ask if the tests pass",
    ),
    (
        "translate",
        ScreenContext(window_class="vesktop", window_title="#espanol"),
        (),
        "say good morning everyone the build is fixed in spanish",
    ),
    (
        "deramble-email",
        ScreenContext(window_class="eu.betterbird.Betterbird", window_title="Compose"),
        (),
        "um so basically tell them uh that the meeting got moved to like thursday"
        " 3 p.m. and they should uh bring the the quarterly report yeah",
    ),
    (
        "revise",
        ScreenContext(window_class="vesktop"),
        (("reply saying i can do it after lunch", "I can do it after lunch."),),
        "make that sound way more excited",
    ),
    (
        "no-false-instruction",
        ScreenContext(window_class="vesktop"),
        (),
        "i told him to give me the report by friday",  # dictation, NOT a command
    ),
    (
        "verbatim-voice",
        ScreenContext(window_class="kitty"),
        (),
        "the fix is to set spec draft n max to eleven in the toml",
    ),
]


def run_profile(profile: str, cfg) -> list[tuple[str, float, str]]:
    cfg.contextual.profile = profile
    tr = Transformer(cfg.contextual)
    results = []
    # Warm the prompt cache so latencies reflect steady state.
    ctx0 = CASES[0][1]
    try:
        tr.prewarm(ctx0)
    except Exception:
        pass
    for name, ctx, history, transcript in CASES:
        t0 = time.perf_counter()
        try:
            out = tr.transform(transcript, context=ctx, history=history)
        except Exception as e:
            out = f"<ERROR: {e}>"
        results.append((name, time.perf_counter() - t0, out))
    return results


def main() -> int:
    # Screenshots off for eval determinism.
    cfg = Config.load()
    cfg.contextual.screenshot = "off"
    profiles = sys.argv[1:] or [cfg.contextual.profile]

    all_results = {}
    for profile in profiles:
        print(f"=== {profile} ({cfg.contextual.profiles[profile].model}) ===")
        all_results[profile] = run_profile(profile, cfg)
        for name, dt, out in all_results[profile]:
            print(f"  [{dt:5.2f}s] {name:22} -> {out!r}")
        total = sum(dt for _, dt, _ in all_results[profile])
        print(f"  total: {total:.2f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
