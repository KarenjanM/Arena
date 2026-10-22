#!/bin/bash
set -eo pipefail
apt-get update
apt-get install -y --no-install-recommends ca-certificates ccache curl git python3-vcstool zstd
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
mkdir -p /etc/arena-ci
dpkg-query -W -f '${db:Status-Status} ${Package}\n' | awk '$1 == "installed" { print $2 }' | sort -u > /etc/arena-ci/base-packages
cp -a "$1" /tmp/ws
cd /tmp/ws
.github/ci/fetch-deps.sh
mapfile -t tested < <(.github/ci/tested-packages.sh)
.github/ci/system-deps.sh --purge-dropped "${tested[@]}"
cd /
rm -rf /tmp/ws /var/lib/apt/lists/*
