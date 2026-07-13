"""Runtime configuration for heard.

Values are resolved in this order (later wins):

  1. built-in defaults (below)
  2. ~/.config/heard/config.toml  (override with $HEARD_CONFIG)
  3. HEARD_* environment variables

Unlike the Salt-templated original, nothing is baked in at install time — the
same installed artifact reads its config at runtime, so one build serves every
machine and the deployer just drops a config file.

`load()` is the entry point; everything else is here so the shape of the
resolution is legible.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


def default_endpoint() -> str:
    port = os.environ.get("HEARD_PORT", "8083")
    return f"http://127.0.0.1:{port}/v1/audio/transcriptions"


@dataclass(frozen=True)
class ContextConfig:
    """On-screen vocabulary bias sourced from the focused herdr pane."""

    # Master switch. Even when True, context is a graceful no-op outside herdr
    # (HERDR_ENV != "1") or when the `herdr` CLI isn't reachable.
    enabled: bool = True
    # Which slice of the pane to read: "visible" (the viewport the user is
    # looking at) is the most relevant; "recent"/"recent-unwrapped" pull
    # scrollback for more terms at the cost of noise.
    source: str = "visible"
    # Cap how much text we read/scan (pane rows).
    lines: int = 200
    # Max on-screen terms promoted to Whisper `hotwords` (term-boost).
    max_hotwords: int = 48
    # Extra on-screen terms appended to the `initial_prompt`, budgeted so the
    # combined prompt stays well under Whisper's ~224-token window.
    prompt_terms: int = 24
    # Trust a keydown-prefetched bias cache younger than this (seconds).
    cache_ttl: float = 8.0


@dataclass(frozen=True)
class Config:
    endpoint: str = field(default_factory=default_endpoint)
    model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    language: str = "en"
    # Press Enter after typing the transcript.
    submit: bool = True
    # GLOBAL vocabulary bias (Whisper initial_prompt) — terms you say across
    # every project. Project/on-screen jargon comes from [context] at runtime.
    prompt: str = ""
    # Space-separated global boost terms, always merged ahead of on-screen ones.
    hotwords: str = ""
    context: ContextConfig = field(default_factory=ContextConfig)


def config_path() -> Path:
    if env := os.environ.get("HEARD_CONFIG"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "heard" / "config.toml"


def as_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in TRUTHY:
        return True
    if s in FALSY:
        return False
    return fallback


def from_toml(path: Path) -> Config:
    cfg = Config()
    if not path.is_file():
        return cfg
    data = tomllib.loads(path.read_text())
    ctx_data = data.pop("context", None)
    overrides = {
        f.name: data[f.name]
        for f in fields(Config)
        if f.name != "context" and f.name in data
    }
    if "submit" in overrides:
        overrides["submit"] = as_bool(overrides["submit"], cfg.submit)
    cfg = replace(cfg, **overrides)
    if isinstance(ctx_data, dict):
        ctx_over = {
            f.name: ctx_data[f.name]
            for f in fields(ContextConfig)
            if f.name in ctx_data
        }
        if "enabled" in ctx_over:
            ctx_over["enabled"] = as_bool(ctx_over["enabled"], cfg.context.enabled)
        cfg = replace(cfg, context=replace(cfg.context, **ctx_over))
    return cfg


def apply_env(cfg: Config) -> Config:
    env = os.environ
    top = {}
    if "HEARD_ENDPOINT" in env:
        top["endpoint"] = env["HEARD_ENDPOINT"]
    if "HEARD_MODEL" in env:
        top["model"] = env["HEARD_MODEL"]
    if "HEARD_LANGUAGE" in env:
        top["language"] = env["HEARD_LANGUAGE"]
    if "HEARD_SUBMIT" in env:
        top["submit"] = as_bool(env["HEARD_SUBMIT"], cfg.submit)
    if "HEARD_PROMPT" in env:
        top["prompt"] = env["HEARD_PROMPT"]
    if "HEARD_HOTWORDS" in env:
        top["hotwords"] = env["HEARD_HOTWORDS"]
    if top:
        cfg = replace(cfg, **top)

    ctx = {}
    if "HEARD_CONTEXT" in env:
        ctx["enabled"] = as_bool(env["HEARD_CONTEXT"], cfg.context.enabled)
    if "HEARD_CONTEXT_SOURCE" in env:
        ctx["source"] = env["HEARD_CONTEXT_SOURCE"]
    if ctx:
        cfg = replace(cfg, context=replace(cfg.context, **ctx))
    return cfg


def load() -> Config:
    """Resolve the effective configuration."""
    return apply_env(from_toml(config_path()))
