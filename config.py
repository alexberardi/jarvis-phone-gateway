"""Environment configuration (bootstrap + secrets only — PRD security req 5).

Twilio credentials are gateway-only env secrets: never the settings DB,
never CC. The auth token doubly so — it is also the signature-verification
key for inbound media-stream upgrades.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _https_base_from_wss(wss: str) -> str:
    """Derive the https/http public base from a wss/ws URL — used to reconstruct the
    request URL for Twilio signature validation when PUBLIC_URL isn't set explicitly
    (the compose only emits the wss base, PHONE_GATEWAY_PUBLIC_WSS_URL)."""
    wss = (wss or "").rstrip("/")
    if wss.startswith("wss://"):
        return "https://" + wss[len("wss://"):]
    if wss.startswith("ws://"):
        return "http://" + wss[len("ws://"):]
    return wss


@dataclass
class GatewayConfig:
    twilio_account_sid: str = field(default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", ""))
    twilio_auth_token: str = field(default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", ""))
    twilio_from_number: str = field(default_factory=lambda: os.getenv("TWILIO_FROM_NUMBER", ""))
    # Public https base of THIS worker (behind the named Cloudflare tunnel), used to
    # reconstruct the request URL for Twilio signature validation. Prefer PUBLIC_URL;
    # else derive it from the wss base the generator emits (PHONE_GATEWAY_PUBLIC_WSS_URL),
    # so a deployment that only sets the wss URL still validates signatures.
    public_url: str = field(default_factory=lambda: (
        os.getenv("PUBLIC_URL")
        or _https_base_from_wss(os.getenv("PHONE_GATEWAY_PUBLIC_WSS_URL") or os.getenv("PUBLIC_WSS_URL") or "")
    ).rstrip("/"))
    # Command-center base. Ecosystem-standard name is JARVIS_COMMAND_CENTER_BASE_URL
    # (mirrors JARVIS_AUTH_BASE_URL; what the admin/installer compose generators emit
    # for services that depend on jarvis-command-center). CC_BASE_URL kept as a
    # backward-compat fallback for older .envs.
    cc_base_url: str = field(
        default_factory=lambda: os.getenv("JARVIS_COMMAND_CENTER_BASE_URL")
        or os.getenv("CC_BASE_URL")
        or "http://localhost:7703"
    )
    whisper_url: str = field(default_factory=lambda: os.getenv("WHISPER_URL", "http://localhost:7706"))
    llm_url: str = field(default_factory=lambda: os.getenv("LLM_URL", "http://localhost:7704"))
    tts_url: str = field(default_factory=lambda: os.getenv("TTS_URL", "http://localhost:7707"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    app_id: str = field(default_factory=lambda: os.getenv("JARVIS_APP_ID", ""))
    app_key: str = field(default_factory=lambda: os.getenv("JARVIS_APP_KEY", ""))
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "7713")))
    vad_rms: float = field(default_factory=lambda: float(os.getenv("VAD_RMS", "250")))
    # jarvis-auth base. Ecosystem-standard name is JARVIS_AUTH_BASE_URL (what the
    # compose generators emit for auth-dependent services); JARVIS_AUTH_URL kept as a
    # backward-compat fallback.
    auth_url: str = field(
        default_factory=lambda: os.getenv("JARVIS_AUTH_BASE_URL")
        or os.getenv("JARVIS_AUTH_URL")
        or "http://localhost:7701"
    )
    # Explicit wss base for the TwiML <Stream> URL. The compose generator emits
    # PHONE_GATEWAY_PUBLIC_WSS_URL (the ecosystem name); PUBLIC_WSS_URL is a
    # backward-compat fallback. Empty → derived from public_url (https→wss) below.
    _public_wss_url: str = field(
        default_factory=lambda: (
            os.getenv("PHONE_GATEWAY_PUBLIC_WSS_URL") or os.getenv("PUBLIC_WSS_URL") or ""
        ).rstrip("/")
    )
    run_dial_worker: bool = field(
        default_factory=lambda: os.getenv("RUN_DIAL_WORKER", "true").lower()
        in ("1", "true", "yes")
    )

    @property
    def public_wss_url(self) -> str:
        if self._public_wss_url:
            return self._public_wss_url
        if self.public_url.startswith("https://"):
            return "wss://" + self.public_url[len("https://"):]
        if self.public_url.startswith("http://"):
            return "ws://" + self.public_url[len("http://"):]
        return self.public_url

    @property
    def signature_validation_enabled(self) -> bool:
        """On whenever the Twilio auth token is configured. Dev without
        Twilio creds runs open (fake-client testing); production always has
        the token set, so production always validates."""
        return bool(self.twilio_auth_token)
