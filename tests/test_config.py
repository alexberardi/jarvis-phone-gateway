"""GatewayConfig env resolution.

The gateway must read the ECOSYSTEM-STANDARD env var names the compose generators
emit — JARVIS_AUTH_BASE_URL and JARVIS_COMMAND_CENTER_BASE_URL — with backward-compat
fallbacks. Prod incident 2026-08-07: dropped dial jobs ("session fetch failed") +
app-auth 503s traced to the gateway reading CC_BASE_URL / JARVIS_AUTH_URL, names
nothing in the generated compose set, so both fell back to localhost inside the
container and every outbound call raised ConnectError.
"""
from config import GatewayConfig

CC_KEYS = ("JARVIS_COMMAND_CENTER_BASE_URL", "CC_BASE_URL")
AUTH_KEYS = ("JARVIS_AUTH_BASE_URL", "JARVIS_AUTH_URL")


def _clear(mp, keys):
    for k in keys:
        mp.delenv(k, raising=False)


class TestCcBaseUrl:
    def test_prefers_ecosystem_name(self, monkeypatch):
        _clear(monkeypatch, CC_KEYS)
        monkeypatch.setenv("JARVIS_COMMAND_CENTER_BASE_URL", "http://host.docker.internal:7703")
        monkeypatch.setenv("CC_BASE_URL", "http://legacy:7703")  # must lose to the standard name
        assert GatewayConfig().cc_base_url == "http://host.docker.internal:7703"

    def test_falls_back_to_legacy_name(self, monkeypatch):
        _clear(monkeypatch, CC_KEYS)
        monkeypatch.setenv("CC_BASE_URL", "http://legacy:7703")
        assert GatewayConfig().cc_base_url == "http://legacy:7703"

    def test_default_when_unset(self, monkeypatch):
        _clear(monkeypatch, CC_KEYS)
        assert GatewayConfig().cc_base_url == "http://localhost:7703"


class TestAuthUrl:
    def test_prefers_ecosystem_name(self, monkeypatch):
        _clear(monkeypatch, AUTH_KEYS)
        monkeypatch.setenv("JARVIS_AUTH_BASE_URL", "http://host.docker.internal:7701")
        monkeypatch.setenv("JARVIS_AUTH_URL", "http://legacy:7701")  # must lose to the standard name
        assert GatewayConfig().auth_url == "http://host.docker.internal:7701"

    def test_falls_back_to_legacy_name(self, monkeypatch):
        _clear(monkeypatch, AUTH_KEYS)
        monkeypatch.setenv("JARVIS_AUTH_URL", "http://legacy:7701")
        assert GatewayConfig().auth_url == "http://legacy:7701"

    def test_default_when_unset(self, monkeypatch):
        _clear(monkeypatch, AUTH_KEYS)
        assert GatewayConfig().auth_url == "http://localhost:7701"
