# syntax=docker/dockerfile:1

# Fedora's official Python 3.14 application base (SCLorg s2i image): runs as
# non-root uid 1001, WORKDIR /opt/app-root/src, exposes 8080 by convention,
# ships pip preinstalled. Swap for quay.io/fedora/python-313 to track 3.13
# instead -- pyproject.toml only requires >=3.13, it isn't pinned to 3.14.
FROM quay.io/fedora/python-314:latest

USER 0

# ffmpeg-free is Fedora's patent-unencumbered ffmpeg build; --setopt
# skips the weak/recommended deps (systemd, pipewire, ...) that the full
# dependency set otherwise pulls in for desktop multimedia use we don't need.
RUN dnf -y install --setopt=install_weak_deps=False ffmpeg-free \
    && dnf clean all

WORKDIR /opt/app-root/src

COPY requirements.txt ./
COPY app ./app
COPY static ./static

# uid 1001 with gid 0, group-writable: matches this base image's existing
# OpenShift-style "arbitrary UID" permission convention.
RUN chown -R 1001:0 /opt/app-root/src && chmod -R g=u /opt/app-root/src

USER 1001

# requirements.txt is exported from uv.lock (`uv export --no-hashes
# --no-dev --no-emit-project -o requirements.txt`) so the container gets the
# same pinned versions as local uv-managed development. Re-run that export
# and rebuild whenever pyproject.toml/uv.lock change.
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
