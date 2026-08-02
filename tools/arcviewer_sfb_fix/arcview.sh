#!/usr/bin/env bash
# Open a map in ArcViewer WITHOUT the native file dialog.
#
# ArcViewer 0.8.1 aborts with "free(): invalid pointer" inside GTK's file-chooser
# teardown (gtk_widget_unparent / gtk_container_remove) on this system. The crash
# is in the picker, not in map loading -- passing the path directly loads the same
# maps in ~34ms with no crash.
#
# Usage: arcview <map.zip> [extra=args ...]     e.g. arcview foo.zip difficulty=Expert
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: arcview <map.zip> [key=value ...]" >&2
  exit 2
fi
MAP="$(readlink -f "$1")"; shift
if [ ! -f "$MAP" ]; then echo "arcview: no such file: $MAP" >&2; exit 1; fi
exec /home/kyle/.local/bin/arcviewer "path=$MAP" "$@"
