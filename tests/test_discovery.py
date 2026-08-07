"""Config-service discovery in the gateway (main._discover_service_urls).

The gateway resolves whisper/llm/tts (and command-center) from jarvis-config-service
like the rest of the stack, filling in anything not pinned by an explicit env override.

Regression for the prod chain of 2026-08-07:
  1. jarvis-config-client wasn't installed → discovery inert → localhost defaults →
     dial jobs dropped ("ConnectError: All connection attempts failed").
  2. even with it installed, _discover_service_urls called get_service_url WITHOUT
     init() first, so it raised "not initialized" and silently fell through.
Env-wins keeps the internal command-center override (its public URL is the one path
we pin) from being clobbered by discovery.
"""
import sys
import types

import pytest

import main
from config import GatewayConfig


def _fake_config_client(url_map: dict, record: dict) -> types.ModuleType:
    m = types.ModuleType("jarvis_config_client")

    def init(refresh_interval_seconds: int = 300, **_kw):  # noqa: ANN001
        record["init_called"] = True
        return True

    def get_service_url(name: str):
        record.setdefault("looked_up", []).append(name)
        return url_map.get(name)

    m.init = init  # type: ignore[attr-defined]
    m.get_service_url = get_service_url  # type: ignore[attr-defined]
    return m


@pytest.fixture
def install_fake(monkeypatch):
    def _install(url_map: dict) -> dict:
        record: dict = {}
        monkeypatch.setitem(sys.modules, "jarvis_config_client", _fake_config_client(url_map, record))
        return record
    return _install


def _clear(monkeypatch, *keys):
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_discovers_unpinned_services_and_calls_init(install_fake, monkeypatch):
    _clear(monkeypatch, "JARVIS_COMMAND_CENTER_BASE_URL", "CC_BASE_URL", "WHISPER_URL", "LLM_URL", "TTS_URL")
    rec = install_fake({
        "jarvis-command-center": "https://command-center.example.io",
        "jarvis-whisper-api": "https://whisper.example.io",
        "jarvis-llm-proxy-api": "https://llm.example.io",
        "jarvis-tts": "https://tts.example.io/",  # trailing slash must be stripped
    })
    cfg = GatewayConfig()
    main._discover_service_urls(cfg)

    assert rec["init_called"] is True  # the init() bug fix
    assert cfg.cc_base_url == "https://command-center.example.io"
    assert cfg.whisper_url == "https://whisper.example.io"
    assert cfg.llm_url == "https://llm.example.io"
    assert cfg.tts_url == "https://tts.example.io"


def test_explicit_env_pins_command_center_over_discovery(install_fake, monkeypatch):
    # CC pinned to an internal URL — discovery must NOT overwrite it (its public URL
    # is what we deliberately avoid for the internal hop).
    monkeypatch.setenv("JARVIS_COMMAND_CENTER_BASE_URL", "http://host.docker.internal:7703")
    _clear(monkeypatch, "CC_BASE_URL", "WHISPER_URL", "LLM_URL", "TTS_URL")
    rec = install_fake({
        "jarvis-command-center": "https://command-center.example.io",
        "jarvis-whisper-api": "https://whisper.example.io",
    })
    cfg = GatewayConfig()
    main._discover_service_urls(cfg)

    assert cfg.cc_base_url == "http://host.docker.internal:7703"  # env pin held
    assert "jarvis-command-center" not in rec.get("looked_up", [])  # not even queried
    assert cfg.whisper_url == "https://whisper.example.io"  # unpinned → discovered


def test_missing_client_leaves_env_defaults(monkeypatch):
    _clear(monkeypatch, "WHISPER_URL", "TTS_URL", "LLM_URL")
    # Force the `from jarvis_config_client import ...` to raise ImportError.
    monkeypatch.setitem(sys.modules, "jarvis_config_client", None)
    cfg = GatewayConfig()
    before = (cfg.cc_base_url, cfg.whisper_url, cfg.tts_url, cfg.llm_url)
    main._discover_service_urls(cfg)  # must not raise
    assert (cfg.cc_base_url, cfg.whisper_url, cfg.tts_url, cfg.llm_url) == before
