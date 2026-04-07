# Global Engineering Rules (Windsurf) — `global_rules.md`

> **Purpose**: One canonical, enforceable rulebook for how we build, fix, test, log, document, and communicate across this codebase. This file supersedes ad‑hoc conventions.

---

## 0) Abbreviations (RAT tags)

Use these bracketed tags in commits/PRs/explanations to indicate which rules you applied.

* **CA** – Clean Architecture
* **REH** – Robust Error Handling
* **CSD** – Code Smell Detection
* **IV** – Input Validation
* **RM** – Resource Management
* **CMV** – Constants over Magic Values
* **SFT** – Security‑First Thinking
* **PA** – Performance Awareness
* **RAT** – Rule Application Tracking
* **EDC** – Explanation Depth Control
* **AS** – Alternative Suggestions
* **KBT** – Knowledge Boundary Transparency

> Example: *“Refactor logging bootstrap to enforce dual sinks \[REH]\[SFT]\[CA].”*

---

## 1) Non‑Negotiables (TL;DR)

1. **Run Python via uv**: `uv run python <file.py>` only. Tests: `uv run -m pytest -q`. Lint/format: `uv run ruff check .` / `uv run ruff format .`.
   *Why*: locked interpreter/env, reproducibility. \[PA]\[RM]
2. **Autonomy**: If you say “fix X”, the patch arrives without a permission gate. I pause only for destructive/ambiguous changes (data loss, API breaks). \[EDC]\[KBT]
3. **Tracebacks**: When errors are pasted, responses include **root cause, minimal repro, and a ready patch**. Logging examples use **RichHandler** with rich tracebacks + locals. \[REH]
4. **Repo hygiene**: Ad‑hoc runnable scripts → `utils/`. Verification/proof-of-fix → `tests/`. **Never** drop new files in repo root. \[CA]
5. **Logging**: Must use **Dual Sink Strategy** (Pretty Rich console + Structured JSONL file). Startup **enforcer** aborts if both handlers aren’t active. \[REH]\[SFT]

---

## 2) Repository Layout (authoritative)

```
project/
  src/                     # app code (layered; see CA)
  utils/                   # one-off tools, migrations, scripts
  tests/                   # pytest (unit/integration/property)
  logs/                    # runtime logs (gitignored)
  docs/                    # living docs (CDiP)
  pyproject.toml
  .env.example             # required env vars (no secrets)
```

* Don’t add top-level dirs without rationale in PR.
* Keep `logs/` out of version control.

---

## 3) Clean Architecture (CA)

**Layers**

* `src/domain/` – pure business logic (entities, value objects, domain services). No I/O, no globals.
* `src/usecases/` – application services orchestrating domain.
* `src/adapters/` – boundaries to the outside world (db, HTTP, Discord, FS, etc.).
* `src/framework/` – entrypoints/CLI, DI/wiring, config, logging bootstrap.

**Rules**

* Inner layers must not import outward. Depend on interfaces/protocols; inject concrete adapters at composition root.
* No global state inside `domain/` or `usecases/`.

---

## 4) Execution & Tooling

* **Python**: `uv run python <file.py>`.
* **Tests**: `uv run -m pytest -q` (markers: `unit`, `integration`, `slow`).
* **Lint/format**: `uv run ruff check . && uv run ruff format .`.
* **(Optional)** Make targets: `make run ARGS=…`, `make test`, `make lint`, `make format`. \[PA]

---

## 5) Logging Spec (Rich + JSONL) — Dual Sink Strategy

**Pretty Console Sink (RichHandler)**

* Rich tracebacks enabled, **show locals** on DEBUG.
* Timestamps: **local time**, millisecond precision.
* Icons & colour palette: INFO=green, WARNING=yellow, ERROR/CRIT=red, DEBUG=blue‑grey.
* Level‑based symbols: ✔, ⚠, ✖, ℹ. Icons **left‑pad** each line; align field grid after icon.
* Auto‑truncate oversized fields with `…(+N)` indicator.
* May use `Tree` and `Panel` for structured dumps.

**Structured JSONL Sink**

* File: `logs/app.jsonl` (rotating 10MB × 5).
* Keys (preserve set): `ts, level, name, subsys, guild_id, user_id, msg_id, event, detail` (+ `message`).
* One JSON object per line.

**Enforcer Function (mandatory)**

* On startup, assert **exactly two handlers** named `pretty_handler` and `jsonl_handler`. **Abort** if misconfigured. \[REH]

**Privacy**

* Never log secrets/PII; scrub tokens; hash IDs when required. \[SFT]

---

## 6) Robust Error Handling (REH)

* **Top‑level guard**: visible crash banner on unrecoverable errors; **non‑zero exit**.
* **Timeouts**: external calls default ≤ 10s per request; no unbounded waits. \[PA]
* **Retries**: exponential backoff with jitter for transient I/O; cap retries; log attempts. \[RM]
* **Typed exceptions** per boundary (e.g., `DiscordGatewayError`, `StorageError`).
* **Tracebacks**: DEBUG includes locals; INFO presents root cause + remediation.

---

## 7) Code Smell Detection (CSD) — Gates

Fail CI unless justified in PR:

* Function > **30 lines** (excl. docstring) **or** nesting depth > **3**.
* File > **300 SLOC**.
* Class > **5 public methods**.

Refactor strategies: extract method/object, split adapters, strategy pattern, pipeline composition. \[AS]

---

## 8) Security & Performance

**Security (SFT)**

* **Input Validation (IV)** at all boundaries (Pydantic or dataclasses + validators).
* **Secrets policy**: `.env` for local only; production uses env/secret store. Never commit secrets.
* **Static analysis**: run `bandit` over `src/`.
* **Least privilege** for tokens/keys; avoid over‑scoped API grants.

**Performance (PA)**

* Avoid O(N²) in hot paths; add micro‑benchmarks when risk is suspected.
* Prefer async I/O for network/file concurrency; no blocking inside event loops.
* Bounded queues; circuit breakers for flaky dependencies.
* **Resource Management (RM)**: close files/sockets; cancel tasks on shutdown.

**Constants over Magic Values (CMV)**

* No magic strings/numbers; extract to `src/framework/constants.py` or local `Enum`s.

---

## 9) Testing & Coverage

* **Pytest** with markers: `unit`, `integration`, `slow`.
* **Coverage**: ≥ **85%** lines across `src/` (adapters that require live creds may be partially exempted with rationale).
* **Property‑based** tests (Hypothesis) for parsing/validation logic.
* Tests must prove:

  * Both logging sinks exist and emit expected fields.
  * Enforcer aborts when a sink is missing.
  * Retry policy engages on transient errors.

---

## 10) AI Communication Guidelines

* **RAT**: Tag rules applied in responses/commits (e.g., `[REH][CA]`).
* **EDC**: Scale explanation: simple changes → brief; cross‑cutting → structured plan with diffs/risks.
* **AS**: Provide at least one credible alternative when trade‑offs exist.
* **KBT**: If context is missing, proceed with clearly stated assumptions and safest defaults.

---

## 11) Continuous Documentation in Process (CDiP)

* Keep `docs/` **continuously updated** during work, not post‑hoc.
* Living docs:

  * `TASK_LIST.md` – daily progress, TODOs, blockers.
  * `ARCHITECTURE.md` – layer map, data flows, invariants.
  * `LOGGING.md` – sinks, formats, examples.
  * `SECURITY.md` – threat model, secrets, validation rules.
  * `CHANGELOG.md` – Conventional Commits.
* PRs must update relevant docs or state why N/A.

---

## 12) PR & Commit Policy

**Conventional Commits**; include RAT tags where relevant.

**Commit & Push After Every Work**

* After every meaningful change (code, config, docs, tests), **commit and push immediately**. Do not batch changes or defer commits.
* Each commit should be atomic and self-contained with a clear Conventional Commits message.
* This applies to all agents working in this repo — no exceptions.

**PR Checklist (copy‑paste)**

* [ ] Used `uv run` for execution & tests. \[RM]
* [ ] New scripts in `utils/`; new tests in `tests/`. \[CA]
* [ ] Dual logging sinks active; enforcer passes; JSONL fields verified. \[REH]
* [ ] No functions > 30 lines, nesting ≤ 3, file < 300 SLOC (or justified). \[CSD]
* [ ] Inputs validated; timeouts/retries on I/O. \[IV]\[PA]
* [ ] No secrets in logs/code; `.env.example` updated. \[SFT]
* [ ] Tests updated; coverage ≥ 85%. \[REH]
* [ ] Docs in `docs/` updated (list which). \[CDiP]
* [ ] RAT tags added to PR description for major rules applied. \[RAT]

---

## 13) Example: Logging Bootstrap (reference snippet)

```python
# utils/logging_setup.py
from __future__ import annotations
import json, logging, os, sys, time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict
from rich.console import Console
from rich.logging import RichHandler

ICON = {logging.DEBUG:"ℹ", logging.INFO:"✔", logging.WARNING:"⚠", logging.ERROR:"✖", logging.CRITICAL:"✖"}

class JSONLFormatter(logging.Formatter):
    KEYS = ("ts","level","name","subsys","guild_id","user_id","msg_id","event","detail","message")
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        for k in ("subsys","guild_id","user_id","msg_id","event","detail"):
            payload[k] = getattr(record, k, None)
        return json.dumps({k: payload.get(k) for k in self.KEYS if payload.get(k) is not None})

class IconFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.icon = ICON.get(record.levelno, "•")
        return True

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(level)

    console = Console(stderr=True, soft_wrap=False)
    pretty = RichHandler(
        console=console,
        show_path=False,
        enable_link_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        log_time_format="%Y-%m-%d %H:%M:%S.%f",
    )
    pretty.set_name("pretty_handler")
    pretty.addFilter(IconFilter())

    log_path = os.getenv("APP_JSONL_PATH", "logs/app.jsonl")
    _ensure_dir(log_path)
    jsonl = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    jsonl.set_name("jsonl_handler")
    jsonl.setFormatter(JSONLFormatter())

    logger.handlers = [pretty, jsonl]

    names = sorted(h.get_name() for h in logger.handlers)
    if names != ["jsonl_handler", "pretty_handler"]:
        sys.stderr.write(f"[logging-enforcer] expected pretty_handler + jsonl_handler, got {names}\n")
        sys.stderr.flush()
        os._exit(2)

    logger._configured = True  # type: ignore[attr-defined]
    return logger
```

**Entrypoint usage**

```python
from utils.logging_setup import setup_logging
logger = setup_logging()
logger.info("%s Startup complete", "✔", extra={"subsys":"bootstrap","event":"startup"})
```

---

## 14) Example Tests (reference)

```python
# tests/test_logging_setup.py
import json, logging, os
from utils.logging_setup import setup_logging

def test_dual_handlers(tmp_path, monkeypatch):
    p = tmp_path / "app.jsonl"
    monkeypatch.setenv("APP_JSONL_PATH", str(p))
    root = setup_logging(logging.DEBUG)
    names = sorted(h.get_name() for h in root.handlers)
    assert names == ["jsonl_handler", "pretty_handler"]

    logging.getLogger("test").info("hello", extra={"subsys":"x","event":"unit"})
    assert p.exists()
    line = p.read_text().splitlines()[-1]
    obj = json.loads(line)
    assert obj["level"] == "INFO" and obj["event"] == "unit" and obj["message"] == "hello"

def test_enforcer_reconfigures_cleanly(monkeypatch):
    root = logging.getLogger()
    root.handlers = []
    # next call to setup_logging should rebuild handlers; enforced by previous test
    assert True
```

---

## 15) Working Agreements

* Prefer full‑file rewrites for fixes (avoid piecemeal patches) to maintain coherence. \[CA]
* Keep answers and commits **action‑oriented**; explanations sized by EDC.
* When ambiguity exists, choose the safest, reversible path and state assumptions. \[KBT]

---

### Appendix: Quick Commands

* Run: `uv run python path/to/app.py`
* Tests: `uv run -m pytest -q`
* Lint: `uv run ruff check .`
* Format: `uv run ruff format .`
* Bandit: `uv run bandit -q -r src`

## 16) Operational Discipline — Context First, No Guesswork

* **DO NOT WRITE A SINGLE LINE OF CODE UNTIL YOU UNDERSTAND THE SYSTEM.** \[KBT]
* **Immediately list files** in the target directory and map relevant entrypoints before edits. \[CA]
* **Ask only necessary clarifying questions**—no fluff. \[EDC]
* **Detect and follow existing patterns** (style, structure, logic); **match** conventions. \[CA]
* **Identify env vars, config files, and dependencies** up front (what/where/how loaded). \[IV]\[SFT]

## 17) Challenge the Request — Don’t Blindly Follow

* Surface **edge cases** immediately. \[REH]
* Lock down **inputs / outputs / constraints** before changes. \[KBT]
* Question vagueness; call out assumptions explicitly. \[KBT]
* Refine the task until the **goal is bullet‑proof** (acceptance criteria). \[EDC]

## 18) Hold the Standard — Every Line Must Count

* Code must be **modular, testable, clean**; follow CA boundaries. \[CA]
* **Docstrings and comments** explain intent and non‑obvious logic. \[EDC]
* Suggest **modern best practices** when current approach is outdated. \[AS]
* If there’s a better way, **say so** and justify trade‑offs. \[AS]

## 19) Zoom Out — Design, Don’t Patch

* Prefer **design** over ad‑hoc fixes; think maintainability, usability, scalability. \[PA]
* Consider all components (frontend, backend, DB, UX). \[CA]
* Plan for the **user experience**, not just bare functionality. \[EDC]

## 20) Web Terminology — Speak the Right Language

* Frame solutions in terms of **APIs, routes, components, and data flow**.
* Understand **frontend↔backend interactions** before changing either. \[KBT]

## 21) One File, One Response

* Do **not split file responses**; provide complete files for changes. \[CA]
* Do **not rename methods** unless necessary for correctness. \[CSD]
* Seek approval **only** when clarity is needed; otherwise **execute**. \[EDC]

## 22) Enforce Strict Standards

* **Clean code, clean structure**; run linters/formatters. If missing, **flag it**. \[CSD]
* **File size limits**: CSD gate at **300 SLOC** (review/refactor threshold); **hard cap** at **1600 lines per file** (reject/partition). \[CSD]
* **Highlight any file** trending large or complex; propose decomposition. \[AS]

## 23) Move Fast, But With Context

Before execution, **bullet your plan**:

* **What** you’re doing
* **Why** you’re doing it
* **What** you expect to change (behaviour, interfaces, performance)
  Tag the plan with RAT codes (e.g., `[REH][CA][PA]`).

## 24) ABSOLUTE DO‑NOTs

* Do **not** change translation keys unless explicitly specified. \[SFT]
* Do **not** add unnecessary logic; keep scope tight. \[CSD]
* Do **not** wrap everything in try/except—**think first**; handle errors where they belong. \[REH]
* Do **not** spam files with non‑essential components. \[CA]
* Do **not** introduce side effects without **calling them out** in the plan/PR. \[RM]

## 25) Remember

* The job isn’t done until the **system is stable**. \[REH]
* Think through **consequences** and cross‑module impacts. \[CA]
* If you break something **anywhere**, fix it **everywhere**. \[REH]
* **Cleanup, document, review**. Update living docs under `docs/`. \[CDiP]

## 26) Think Like a Human

* Consider **natural behaviour** and real user flows. \[EDC]
* How would a **user interact** with this? What happens when it fails? \[REH]
* Aim for **seamless** experiences (clear messages, safe defaults, recovery paths). \[SFT]

---

**Mantra**: *Execute like a professional coder. Think like an architect. Deliver like a leader.*
