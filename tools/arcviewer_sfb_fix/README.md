# GTK-free `libStandaloneFileBrowser.so` for ArcViewer

Replaces ArcViewer's native file-dialog plugin with one that shells out to
`zenity` instead of running GTK inside the Unity process.

## The bug

ArcViewer **0.7.7 and 0.8.1** (Unity 6000.0.67f1) abort on this machine the moment
the "open map" dialog is dismissed — the app hangs, then dies:

```
free(): invalid pointer
Caught fatal signal - signo:6 code:-6
#3  malloc_printerr
#4  _int_free
#7  g_object_unref
#33 gtk_widget_unparent
#39 gtk_container_remove
```

Every frame is GLib/GTK, and it fires immediately after the dialog's filter
strings (`Map Files;zip,dat / zip / dat`) are logged — **before any map file is
read**.

### Why it happens

`Assets/StandaloneFileBrowser/Plugins/Linux/x86_64/libStandaloneFileBrowser.so`
links the **system** GTK3 stack straight into the Unity process:

```
libgtk-3.so.0, libgdk-3.so.0, libgobject-2.0.so.0, libglib-2.0.so.0,
libcairo-gobject.so.2, libgdk_pixbuf-2.0.so.0
```

Unity 6 ships its own copies of much of that stack and its own allocator. Widget
teardown then frees a pointer a different allocator owns, and glibc aborts. It is
an in-process library conflict, not a map-parsing problem — confirmed by loading
the identical map through the command line (`ArcViewer.x86_64 "path=<abs>"`),
which parses it in **34 ms with no crash**.

## The fix

Keep the exact C ABI that `StandaloneFileBrowserLinux.cs` P/Invokes, but run the
dialog in a **child process** (`zenity`). No GTK is loaded into Unity's address
space, so the conflicting-allocator teardown cannot occur. If the dialog process
crashes, ArcViewer survives.

Same seven exports, verified identical to the original:

```
DialogInit  DialogOpenFilePanel  DialogOpenFilePanelAsync
DialogOpenFolderPanel  DialogOpenFolderPanelAsync
DialogSaveFilePanel  DialogSaveFilePanelAsync
```

Design notes:

- **`fork` + `execvp` with an argv array, never a shell** — paths containing
  spaces, quotes or newlines cannot be misparsed or injected.
- **One reusable heap buffer** for the returned string; C# copies it immediately
  via `Marshal.PtrToStringAnsi`, so ownership never crosses the boundary. That is
  the exact class of bug being fixed, and not one worth reintroducing in the fix.
- **Never returns `NULL`** — the caller does `paths.Split((char)28)` and would
  throw a `NullReferenceException`. Cancel returns `""`, which
  `ExplorerManager.cs` already handles ("No path selected!").
- **Async entry points call the callback inline.** A modal dialog blocks the UI
  anyway, and invoking a managed delegate from an unattached native thread is a
  worse risk than a frozen frame.

## Build, test, install

```bash
./build.sh                       # gcc, no deps beyond libc
./selftest                       # exercises the ABI without launching Unity
PATH=/nonexistent ./selftest     # also check the no-zenity / cancel path
```

`selftest` checks that all seven exports resolve, that **no GTK is pulled into the
process**, that nothing ever returns `NULL`, that cancel yields `""`, and that 200
repeated calls don't corrupt the shared buffer.

Install (already applied):

```bash
P=../ArcViewer-bin/ArcViewer_Data/Plugins
cp -n $P/libStandaloneFileBrowser.so $P/libStandaloneFileBrowser.so.gtk-original
cp libStandaloneFileBrowser.so $P/
```

Revert:

```bash
P=../ArcViewer-bin/ArcViewer_Data/Plugins
cp $P/libStandaloneFileBrowser.so.gtk-original $P/libStandaloneFileBrowser.so
```

## Status / caveats

- ✅ **VERIFIED END TO END.** Kyle opened every map through the normal "Select Map"
  dialog on ArcViewer 0.8.1 — the crash is gone. Previously this aborted the app
  100% of the time on dismiss.
- Also verified mechanically: builds clean, identical export set to the original,
  links only `libc`, self-test passes, no `DllNotFoundException` at runtime.
- Requires `zenity` on `PATH` (3.44.2 present here). Without it, every dialog
  behaves as "cancelled" rather than crashing — a deliberate degradation.
- Survives `git pull` (this directory is untracked) but **not** a re-extract of
  `ArcViewer.Linux.zip`, which would restore the GTK plugin. Re-run the install
  step after any ArcViewer update.
- Worth reporting upstream (AllPoland/ArcViewer): in-process GTK from the
  StandaloneFileBrowser plugin is fragile under Unity 6 on Linux. Logs with the
  full 70-frame trace are preserved in the automapper repo at
  `outputs/arcviewer_crash_logs_2026-08-02/`.
