#!/usr/bin/env bash
# Make sure the GTK-free file-dialog plugin is the one in place, and re-install it
# if an ArcViewer update has put the crashing GTK build back.
#
# Exists because re-extracting ArcViewer.Linux.zip silently restores the upstream
# plugin, which aborts the app on every dialog dismiss (invalid free in GTK widget
# teardown, see README.md). A caveat in a file nobody rereads is not a fix, so the
# launcher enforces it instead.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGINS="$HERE/../ArcViewer-bin/ArcViewer_Data/Plugins"
FIXED="$HERE/libStandaloneFileBrowser.so"
LIVE="$PLUGINS/libStandaloneFileBrowser.so"

[ -f "$FIXED" ] || { echo "arcviewer: fixed plugin missing at $FIXED" >&2; exit 0; }
[ -d "$PLUGINS" ] || exit 0

# Already ours? (compare content, not timestamps)
if [ -f "$LIVE" ] && cmp -s "$FIXED" "$LIVE"; then
    exit 0
fi

# Not ours: keep the vendor copy once, then install the fix.
if [ -f "$LIVE" ] && ! [ -f "$PLUGINS/libStandaloneFileBrowser.so.gtk-original" ]; then
    cp "$LIVE" "$PLUGINS/libStandaloneFileBrowser.so.gtk-original"
fi
cp "$FIXED" "$LIVE"
echo "arcviewer: re-applied the GTK-free file-dialog plugin (an update had reverted it)" >&2
