#!/usr/bin/env bash
# Build the GTK-free StandaloneFileBrowser replacement. No dependencies but libc.
set -euo pipefail
cd "$(dirname "$0")"
gcc -O2 -fPIC -shared -Wall -Wextra -o libStandaloneFileBrowser.so sfb_zenity.c
gcc -O2 -Wall -o selftest selftest.c -ldl
echo "built:"
ls -l libStandaloneFileBrowser.so selftest
