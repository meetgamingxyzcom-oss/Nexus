FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 nexus \
    && mkdir -p /data/Nexus /workspaces \
    && chown -R nexus:nexus /data /workspaces

WORKDIR /opt/nexus
COPY agent.py gui.py runner.py ./
COPY providers ./providers
COPY ui ./ui

ENV HOME=/home/nexus \
    LOCALAPPDATA=/data \
    NEXUS_DATA_DIR=/data/Nexus \
    NEXUS_WORKSPACE_ROOT=/workspaces \
    NEXUS_BIND_HOST=0.0.0.0 \
    PYTHONDONTWRITEBYTECODE=1

USER nexus
EXPOSE 8765
CMD ["python", "gui.py", "--port", "8765", "--no-browser"]
