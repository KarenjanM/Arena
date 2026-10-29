#!/bin/bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
colcon list --base-paths . -t | while read -r name path _; do
  case "$path" in deps/rosnav_rl) ;; deps/* | */deps/* | arena_tools* | arena_planners/planners/*) continue ;; esac
  if [ -d "$path/test" ] || [ -d "$path/tests" ]; then echo "$name"; fi
done | sort -u
