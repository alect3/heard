"""End-to-end scenarios that simulate real usage of `heard`.

Each test drives the actual CLI (or the PTT class) against the faked external
world from conftest.py, and asserts on what the user would observe: text typed
into their window, and what the STT server was asked to transcribe.
"""

from __future__ import annotations


import pytest

# The context module reaches herdr; import the daemon lazily inside the one PTT
# test so evdev import failures (if any) don't break the CLI tests.


def test_transcribe_types_the_transcript(harness):
    """`heard transcribe <wav>` -> the transcript is typed and submitted."""
    harness.stt.set_transcript("deploy the cluster")
    wav = harness.make_wav()

    result = harness.run("transcribe", str(wav))

    assert result.returncode == 0, result.stderr
    assert harness.stt.request_count == 1
    assert harness.typed_text() == "deploy the cluster"
    assert harness.pressed_return()  # submit = true by default


def test_transcript_is_whitespace_trimmed(harness):
    harness.stt.set_transcript("  spaced out \n")
    result = harness.run("transcribe", str(harness.make_wav()))
    assert result.returncode == 0, result.stderr
    assert harness.typed_text() == "spaced out"


def test_submit_false_does_not_press_return(harness):
    harness.write_config(submit=False)
    harness.stt.set_transcript("no newline please")

    harness.run("transcribe", str(harness.make_wav()))

    assert harness.typed_text() == "no newline please"
    assert not harness.pressed_return()


def test_empty_transcription_types_nothing(harness):
    harness.stt.set_transcript("   ")  # server heard only silence
    result = harness.run("transcribe", str(harness.make_wav()))
    assert result.returncode == 0, result.stderr
    assert harness.typed_text() is None  # nothing typed into the window


def test_server_error_is_reported_not_typed(harness):
    harness.stt.set_status(500)
    result = harness.run("transcribe", str(harness.make_wav()))
    assert result.returncode == 1
    assert harness.typed_text() is None


def test_request_carries_model_and_language(harness):
    harness.run("transcribe", str(harness.make_wav()))
    body = harness.stt.last_body()
    assert 'name="model"' in body and "test-whisper" in body
    assert 'name="response_format"' in body and "text" in body
    assert 'name="language"' in body and "en" in body


def test_request_carries_decode_knobs(harness):
    """vad_filter and temperature are sent explicitly, defaulting off/0.0 so
    the server's own defaults can't drift under us."""
    harness.run("transcribe", str(harness.make_wav()))
    body = harness.stt.last_body()
    assert 'name="vad_filter"' in body and "false" in body
    assert 'name="temperature"' in body and "0.0" in body


def test_vad_filter_can_be_enabled(harness):
    harness.run(
        "transcribe",
        str(harness.make_wav()),
        extra_env={"HEARD_VAD_FILTER": "true"},
    )
    body = harness.stt.last_body()
    assert 'name="vad_filter"' in body and "true" in body


# --- the headline feature: on-screen vocabulary bias ----------------------


def test_focused_pane_terms_bias_the_request(harness):
    """The identifiers visible in the focused herdr pane are sent as hotwords
    and appended to the prompt, so the model is primed for on-screen jargon."""
    harness.set_pane_text(
        "editing opossum-ec wrapper in ec-image/PROD-EC-RUNBOOK.md, run kubectl rollout"
    )
    harness.stt.set_transcript("check the opossum ec wrapper")

    harness.run("transcribe", str(harness.make_wav()))
    body = harness.stt.last_body()

    assert 'name="hotwords"' in body
    assert "opossum-ec" in body  # kebab-case identifier from the pane
    assert "PROD-EC-RUNBOOK" in body  # ALLCAPS identifier from the pane
    assert "git" in body and "podman" in body  # static hotwords still present
    assert 'name="prompt"' in body
    assert "On screen:" in body  # prompt was augmented
    assert "Base prompt." in body  # static prompt preserved


def test_context_disabled_sends_only_static_bias(harness):
    harness.write_config(context={"enabled": False})
    harness.set_pane_text("opossum-ec kubectl nftables")

    harness.run("transcribe", str(harness.make_wav()))
    body = harness.stt.last_body()

    assert "opossum-ec" not in body  # pane not consulted
    assert "On screen:" not in body
    assert "git" in body and "podman" in body  # static hotwords only


def test_outside_herdr_falls_back_to_static(harness):
    harness.set_pane_text("opossum-ec kubectl")
    result = harness.run(
        "transcribe", str(harness.make_wav()), extra_env={"HERDR_ENV": "0"}
    )
    assert result.returncode == 0, result.stderr
    body = harness.stt.last_body()
    assert "opossum-ec" not in body
    assert "git" in body  # static bias survives


def test_context_subcommand_reports_on_screen_terms(harness):
    harness.set_pane_text("opossum-ec kubectl foobar-widget nftables")
    result = harness.run("context", "--show")
    assert result.returncode == 0, result.stderr
    assert "focused pane: t:p2" in result.stdout  # picked the focused pane
    assert "opossum-ec" in result.stdout
    assert "foobar-widget" in result.stdout


# --- the toggle (keybind fallback) flow -----------------------------------


def test_toggle_start_then_stop_types_transcript(harness):
    """Simulate the Mod+Shift+Space fallback: toggle to record, toggle to stop
    and transcribe."""
    harness.stt.set_transcript("toggled dictation works")

    start = harness.run("toggle")
    assert start.returncode == 0, start.stderr
    pidfile = harness.runtime / "heard" / "rec.pid"
    assert harness.wait_for(pidfile), "recording pidfile was never created"
    assert harness.wait_for(harness.runtime / "heard" / "rec.wav")

    stop = harness.run("toggle")
    assert stop.returncode == 0, stop.stderr

    assert harness.stt.request_count == 1
    assert harness.typed_text() == "toggled dictation works"
    assert not pidfile.exists()  # cleaned up on stop


def test_stop_when_idle_is_a_noop(harness):
    result = harness.run("stop")
    assert result.returncode == 0, result.stderr
    assert harness.stt.request_count == 0
    assert harness.typed_text() is None


# --- push-to-talk: hold / release without evdev ---------------------------


def test_ptt_keyup_records_and_transcribes(harness, monkeypatch):
    """Simulate a CapsLock press/release: the pre-roll buffer + captured audio
    are written to a WAV and pushed through transcription, biased by the pane."""
    pytest.importorskip("evdev")
    from heard.config import load
    from heard.ptt import PushToTalk

    # Point the daemon (in-process) at the same faked world as the CLI.
    for k, v in harness.env.items():
        if k in (
            "HEARD_CONFIG",
            "XDG_RUNTIME_DIR",
            "HERDR_ENV",
            "HEARD_TEST_PANE_TEXT",
            "HEARD_TEST_WTYPE_LOG",
            "PATH",
        ):
            monkeypatch.setenv(k, v)
    harness.set_pane_text("opossum-ec kubectl")
    monkeypatch.setenv("HEARD_TEST_PANE_TEXT", "opossum-ec kubectl")
    harness.stt.set_transcript("held the key and spoke")

    pt = PushToTalk(load())
    # Seed a second of "audio" into the pre-roll ring, then press & release.
    chunk = b"\x00\x00" * 16000
    with pt.lock:
        pt.ring.append(chunk)
        pt.ring_bytes += len(chunk)
    pt.start()  # keydown: seeds captured from ring, prefetches context
    pt.stop()  # keyup: writes WAV, normalizes, POSTs, types

    assert harness.stt.request_count == 1
    assert harness.typed_text() == "held the key and spoke"
    body = harness.stt.last_body()
    assert "opossum-ec" in body  # pane context flowed through the PTT path


def test_ptt_ignores_taps_below_minimum(harness, monkeypatch):
    pytest.importorskip("evdev")
    from heard.config import load
    from heard.ptt import PushToTalk

    for k in ("HEARD_CONFIG", "XDG_RUNTIME_DIR", "HERDR_ENV", "PATH"):
        monkeypatch.setenv(k, harness.env[k])

    pt = PushToTalk(load())
    # No audio buffered -> below MIN_MS -> nothing sent.
    pt.start()
    pt.stop()
    assert harness.stt.request_count == 0
