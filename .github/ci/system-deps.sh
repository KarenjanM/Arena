#!/bin/bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
purge=false
if [ "$1" = --purge-dropped ]; then
  purge=true
  shift
fi
rosdep update --rosdistro jazzy
mapfile -t paths < <(colcon list --packages-up-to "$@" -p)
keys=$(env -u AMENT_PREFIX_PATH -u ROS_PACKAGE_PATH rosdep keys --ignore-src --rosdistro jazzy --from-paths "${paths[@]}" | sort -u)
record=/etc/arena-ci
if $purge && [ -f "$record/rosdep-keys" ] && [ -f "$record/base-packages" ]; then
  mapfile -t dropped < <(comm -23 "$record/rosdep-keys" - <<< "$keys" \
    | xargs -r rosdep resolve --rosdistro jazzy 2> /dev/null \
    | awk '/^#apt/ { apt = 1; next } /^#/ { apt = 0; next } apt { for (i = 1; i <= NF; i++) print $i }' \
    | sort -u | comm -23 - "$record/base-packages" \
    | comm -12 - <(dpkg-query -W -f '${db:Status-Status} ${Package}\n' | awk '$1 == "installed" { print $2 }' | sort -u))
  if [ ${#dropped[@]} -gt 0 ]; then
    echo "purging what only dropped rosdep keys installed: ${dropped[*]}"
    apt-get purge -y --auto-remove "${dropped[@]}"
  fi
fi
plan=$(rosdep install --simulate --ignore-src -r --rosdistro jazzy --from-paths "${paths[@]}" 2>&1 || true)
if grep -q 'Installation commands' <<< "$plan"; then
  apt-get update
  rosdep install --ignore-src -r -y --rosdistro jazzy --from-paths "${paths[@]}"
fi
mkdir -p "$record"
echo "$keys" > "$record/rosdep-keys"
