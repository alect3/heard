"""On-screen vocabulary bias, sourced from the focused herdr pane.

The idea: whatever terminal pane you're dictating into is full of exactly the
words Whisper mangles — identifiers, filenames, flags, jargon. We read that
pane's visible text, pull out the technical-looking tokens, and feed them to
the STT server as `hotwords` (faster-whisper term-boost) plus a budgeted tail
appended to the static `initial_prompt`. The transcript snaps to what's on
screen instead of guessing at plain-English homophones.

Everything here degrades to a plain no-op: outside herdr (HERDR_ENV != "1"), or
if the `herdr` CLI is missing/slow/errors, `build_bias()` just returns the
static prompt/hotwords from config.

`ptt` calls `prefetch()` on keydown so the pane read overlaps the recording;
`transcribe` calls `build_bias()`, which reuses that fresh cache when present.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from heard.config import Config

HERDR = "herdr"
HERDR_TIMEOUT = 2.0


@dataclass(frozen=True)
class Bias:
    prompt: str
    hotwords: str
    # Where it came from, for the `heard context` debug view.
    pane_id: str | None = None
    on_screen: tuple[str, ...] = ()


# --- herdr plumbing -------------------------------------------------------


def herdr_available() -> bool:
    return os.environ.get("HERDR_ENV") == "1"


def run_herdr(args: list[str]) -> str | None:
    """Run a herdr subcommand, returning stdout or None on any failure."""
    try:
        out = subprocess.run(
            [HERDR, *args],
            capture_output=True,
            text=True,
            timeout=HERDR_TIMEOUT,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def focused_pane_id() -> str | None:
    """The pane the transcript will be typed into — the focused one."""
    raw = run_herdr(["pane", "list"])
    if raw:
        try:
            panes = json.loads(raw)["result"]["panes"]
        except ValueError, KeyError, TypeError:
            panes = []
        for pane in panes:
            if pane.get("focused"):
                return pane.get("pane_id")
    # Fallback: the pane this process is attached to.
    raw = run_herdr(["pane", "current", "--current"])
    if raw:
        try:
            return json.loads(raw)["result"]["pane"]["pane_id"]
        except ValueError, KeyError, TypeError:
            return None
    return None


def read_pane(pane_id: str, source: str, lines: int) -> str | None:
    return run_herdr(
        [
            "pane",
            "read",
            pane_id,
            "--source",
            source,
            "--lines",
            str(lines),
            "--format",
            "text",
        ]
    )


# --- vocabulary extraction ------------------------------------------------

# Split on anything that isn't part of an identifier. Keeps internal hyphens
# and underscores (kebab/snake), so "ec-image" and "snake_case" stay whole;
# dots and slashes split, so "heard.cli" -> heard, cli and a path -> its parts.
TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_-]+")

# Plain lowercase words this common carry no signal and cause the decoder to
# echo. Technical-shaped tokens bypass this list entirely.
STOPWORDS = frozenset("""
    the a an and or but if then else for while of to in on at by with from into
    over under is are was were be been being do does did done has have had this
    that these those it its as not no yes can could should would will shall may
    might must here there when where what which who whom whose why how all any
    each few more most other some such only own same so than too very just also
    about above after again against because before between both down during
    further off out same until up down now new see let use used using make made
    get got run ran runs way like want need know think look looks looking check
    still even much many one two file files line lines code note notes yeah okay
    """.split())

MIN_LEN = 2
PLAIN_MIN_LEN = 4


def shape_is_technical(tok: str) -> bool:
    has_upper = any(c.isupper() for c in tok)
    has_lower = any(c.islower() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if "_" in tok:
        return True
    if "-" in tok and any(c.isalpha() for c in tok):
        return True
    # camelCase / PascalCase: an uppercase letter *after* the first character.
    # This deliberately excludes plain Title-case words ("Welcome", "Fixed"),
    # which have only a leading capital — they'd otherwise crowd out genuine
    # identifiers in the ranking.
    if has_lower and any(c.isupper() for c in tok[1:]):
        return True
    if has_digit and (has_upper or has_lower):  # z20, s16, k3s
        return True
    if has_upper and not has_lower and len(tok) >= 2:  # ALLCAPS acronym: SSH, GPU
        return True
    return False


def keep_token(tok: str) -> bool:
    tok = tok.strip("-_")
    if len(tok) < MIN_LEN or tok.isdigit():
        return False
    if shape_is_technical(tok):
        return True
    # Plain word: keep only reasonably-long, non-stopword tokens (proper nouns
    # like "Wayland"/"podman"/"runbook" survive; "the"/"this"/"code" don't).
    return len(tok) >= PLAIN_MIN_LEN and tok.lower() not in STOPWORDS


def extract_terms(text: str) -> list[str]:
    """On-screen terms, ranked most-useful first.

    Technical-shaped tokens (identifiers, flags, acronyms) always rank ahead of
    plain words; within each class, more-frequent-on-screen ranks higher. Dedup
    is case-insensitive, keeping the casing as it first appeared (spelling of an
    identifier matters to the transcript).
    """
    first_seen: dict[str, str] = {}
    freq: Counter[str] = Counter()
    technical: set[str] = set()
    for raw in TOKEN_SPLIT.split(text):
        tok = raw.strip("-_")
        if not keep_token(tok):
            continue
        key = tok.lower()
        first_seen.setdefault(key, tok)
        freq[key] += 1
        if shape_is_technical(tok):
            technical.add(key)

    def score(key: str) -> tuple[int, float]:
        return (1 if key in technical else 0, math.log1p(freq[key]))

    ranked = sorted(freq, key=score, reverse=True)
    return [first_seen[k] for k in ranked]


# --- bias assembly + prefetch cache ---------------------------------------


def merge_hotwords(static: str, on_screen: list[str], limit: int) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for term in [*static.split(), *on_screen]:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return " ".join(out)


def augment_prompt(static: str, on_screen: list[str], limit: int) -> str:
    if limit <= 0 or not on_screen:
        return static
    tail = ", ".join(on_screen[:limit])
    clause = f"On screen: {tail}."
    return f"{static.rstrip()} {clause}".strip() if static.strip() else clause


def compute_bias(cfg: Config) -> Bias:
    """Build the bias fresh from the focused pane (no cache)."""
    static = Bias(prompt=cfg.prompt, hotwords=cfg.hotwords)
    ctx = cfg.context
    if not ctx.enabled or not herdr_available():
        return static
    pane_id = focused_pane_id()
    if not pane_id:
        return static
    text = read_pane(pane_id, ctx.source, ctx.lines)
    if not text:
        return static
    terms = extract_terms(text)
    if not terms:
        return static
    return Bias(
        prompt=augment_prompt(cfg.prompt, terms, ctx.prompt_terms),
        hotwords=merge_hotwords(cfg.hotwords, terms, ctx.max_hotwords),
        pane_id=pane_id,
        on_screen=tuple(terms[: ctx.max_hotwords]),
    )


def cache_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return Path(runtime) / "heard" / "context.json"


def prefetch(cfg: Config) -> Bias:
    """Compute the bias now and cache it (called on PTT keydown)."""
    bias = compute_bias(cfg)
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "prompt": bias.prompt,
        "hotwords": bias.hotwords,
        "pane_id": bias.pane_id,
        "on_screen": list(bias.on_screen),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
    return bias


def read_cache(max_age: float) -> Bias | None:
    path = cache_path()
    try:
        payload = json.loads(path.read_text())
    except OSError, ValueError:
        return None
    if time.time() - float(payload.get("ts", 0)) > max_age:
        return None
    return Bias(
        prompt=payload.get("prompt", ""),
        hotwords=payload.get("hotwords", ""),
        pane_id=payload.get("pane_id"),
        on_screen=tuple(payload.get("on_screen", ())),
    )


def build_bias(cfg: Config) -> Bias:
    """The bias to send with a transcription request.

    Prefers a fresh keydown-prefetched cache (zero added latency on the hot
    path); otherwise computes it inline.
    """
    if cfg.context.enabled:
        cached = read_cache(cfg.context.cache_ttl)
        if cached is not None:
            return cached
    return compute_bias(cfg)
