# Pinned by digest, matching the sibling services' convention (see
# jarvis-tts/Dockerfile for the incident that motivated it: upstream tag
# drift ABI-broke native deps). Bump deliberately:
#   docker buildx imagetools inspect python:3.11-slim
FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

WORKDIR /app

# git is required to pip-install jarvis-config-client from its git+https URL (used
# for config-service discovery). numpy/scipy ship manylinux wheels, so no compiler
# is needed — just git.
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY audio /app/audio
COPY llm /app/llm
COPY queues /app/queues
COPY services /app/services
COPY telephony /app/telephony
COPY config.py main.py /app/

EXPOSE 7713

# uvicorn[standard] (wsproto) is REQUIRED: bare uvicorn has no WebSocket
# protocol lib -> 404 on the Twilio media-stream upgrade -> 0-second calls
# (P0 failure ladder item 1). requirements.txt pins the [standard] extra.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7713"]
