#!/usr/bin/env bash
# ArcViewer launcher.
#
# Self-heals the native file-dialog plugin before starting. An ArcViewer update
# re-extracts the upstream GTK build of libStandaloneFileBrowser.so, which aborts
# the app on every dialog dismiss (invalid free during GTK widget teardown, because
# it links system GTK into a Unity 6 process that has its own allocator).
# Write-up: /mnt/giga_speed/repos/ArcViewer/native-sfb-zenity/README.md
ARC=/mnt/giga_speed/repos/ArcViewer
"$ARC/native-sfb-zenity/ensure-installed.sh" >/dev/null || true
exec "$ARC/ArcViewer-bin/ArcViewer.x86_64" "$@"
