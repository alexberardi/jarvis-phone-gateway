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


PUBLIC_KEYS = ("PHONE_GATEWAY_PUBLIC_WSS_URL", "PUBLIC_WSS_URL", "PUBLIC_URL")


class TestPublicWssUrl:
    """The media-stream break of 2026-08-07: the compose emits
    PHONE_GATEWAY_PUBLIC_WSS_URL, but the gateway read PUBLIC_WSS_URL/PUBLIC_URL —
    so both were empty, the TwiML <Stream> URL was blank, Twilio never opened the
    media socket, and calls hung up with no audio ("no media stream within 60s").
    """

    def test_reads_generator_wss_name_and_derives_https_public_url(self, monkeypatch):
        _clear(monkeypatch, PUBLIC_KEYS)
        monkeypatch.setenv("PHONE_GATEWAY_PUBLIC_WSS_URL", "wss://calls.jarvisautomation.io")
        c = GatewayConfig()
        assert c.public_wss_url == "wss://calls.jarvisautomation.io"   # TwiML <Stream> url
        assert c.public_url == "https://calls.jarvisautomation.io"     # derived signature base

    def test_public_wss_url_legacy_fallback(self, monkeypatch):
        _clear(monkeypatch, PUBLIC_KEYS)
        monkeypatch.setenv("PUBLIC_WSS_URL", "wss://legacy.example.io")
        assert GatewayConfig().public_wss_url == "wss://legacy.example.io"

    def test_explicit_public_url_wins_and_derives_wss(self, monkeypatch):
        _clear(monkeypatch, PUBLIC_KEYS)
        monkeypatch.setenv("PUBLIC_URL", "https://gw.example.io")
        c = GatewayConfig()
        assert c.public_url == "https://gw.example.io"
        assert c.public_wss_url == "wss://gw.example.io"  # derived from public_url

    def test_empty_when_nothing_set(self, monkeypatch):
        _clear(monkeypatch, PUBLIC_KEYS)
        c = GatewayConfig()
        assert c.public_url == "" and c.public_wss_url == ""
