#!/bin/bash
set -eo pipefail
repo=ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/arena-ci
current=$repo:jazzy-$(date -u +%GW%V)
image=ros:jazzy
for ref in "$repo:jazzy-$(date -u -d '7 days ago' +%GW%V)" "$current"; do
  if docker manifest inspect "$ref" > /dev/null 2>&1; then image=$ref; fi
done
echo "running in $image"
{
  echo "image=$image"
  echo "current=$current"
} >> "$GITHUB_OUTPUT"
