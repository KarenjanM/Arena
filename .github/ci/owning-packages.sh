#!/bin/bash
set -eo pipefail
for path in "$@"; do
  dir=$path
  [ -d "$dir" ] || dir=$(dirname "$dir")
  while [ ! -f "$dir/package.xml" ]; do
    if [ "$dir" = . ] || [ "$dir" = / ]; then
      echo "no package owns $path" >&2
      exit 1
    fi
    dir=$(dirname "$dir")
  done
  sed -n 's:.*<name>\(.*\)</name>.*:\1:p' "$dir/package.xml" | head -1
done | sort -u | paste -sd' '
