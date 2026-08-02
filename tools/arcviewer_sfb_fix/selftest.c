/* Exercise the replacement plugin's ABI exactly as Mono's P/Invoke would.
 *
 * The point is to prove the contract WITHOUT launching Unity: every export
 * resolves, none of them returns NULL (the C# side calls .Split on the result and
 * would throw), and a cancel behaves like a cancel. Run with ZENITY_CANCEL=1 to
 * simulate the user dismissing the dialog without picking anything.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*init_fn)(void);
typedef const char *(*open_fn)(const char *, const char *, const char *, int);
typedef const char *(*folder_fn)(const char *, const char *, int);
typedef const char *(*save_fn)(const char *, const char *, const char *, const char *);
typedef void (*cb_fn)(const char *);
typedef void (*open_async_fn)(const char *, const char *, const char *, int, cb_fn);

static int failures = 0;
static const char *cb_seen = NULL;

static void check(const char *what, int ok)
{
    printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok)
        failures++;
}

static void my_cb(const char *p)
{
    cb_seen = p;
}

int main(int argc, char **argv)
{
    const char *so = argc > 1 ? argv[1] : "./libStandaloneFileBrowser.so";
    void *h = dlopen(so, RTLD_NOW);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }
    printf("loaded %s\n\n", so);

    const char *names[] = {
        "DialogInit", "DialogOpenFilePanel", "DialogOpenFilePanelAsync",
        "DialogOpenFolderPanel", "DialogOpenFolderPanelAsync",
        "DialogSaveFilePanel", "DialogSaveFilePanelAsync",
    };
    printf("exports required by StandaloneFileBrowserLinux.cs:\n");
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
        void *s = dlsym(h, names[i]);
        check(names[i], s != NULL);
    }

    printf("\nno GTK pulled into this process:\n");
    check("libgtk-3 not loaded", dlopen("libgtk-3.so.0", RTLD_NOLOAD) == NULL);
    check("libgobject-2.0 not loaded",
          dlopen("libgobject-2.0.so.0", RTLD_NOLOAD) == NULL);

    init_fn init = (init_fn)dlsym(h, "DialogInit");
    open_fn open_p = (open_fn)dlsym(h, "DialogOpenFilePanel");
    folder_fn folder_p = (folder_fn)dlsym(h, "DialogOpenFolderPanel");
    save_fn save_p = (save_fn)dlsym(h, "DialogSaveFilePanel");
    open_async_fn open_async = (open_async_fn)dlsym(h, "DialogOpenFilePanelAsync");

    printf("\nbehaviour (PATH is stubbed so zenity is 'absent' = cancel path):\n");
    init();
    check("DialogInit survives", 1);

    const char *r = open_p("Select Map", "/tmp", "Map Files;zip,dat", 0);
    check("OpenFilePanel never returns NULL", r != NULL);
    check("OpenFilePanel returns \"\" when no selection", r && *r == '\0');

    r = folder_p("Pick Folder", "/tmp", 0);
    check("OpenFolderPanel never returns NULL", r != NULL);

    r = save_p("Save", "/tmp", "out.zip", "Map Files;zip");
    check("SaveFilePanel never returns NULL", r != NULL);

    cb_seen = NULL;
    open_async("Select Map", "/tmp", "Map Files;zip,dat", 0, my_cb);
    check("Async invokes the callback", cb_seen != NULL);
    check("Async callback arg is never NULL", cb_seen != NULL);

    /* repeated calls must not corrupt the shared result buffer */
    for (int i = 0; i < 200; i++) {
        const char *x = open_p("t", "/tmp", "A;zip|B;dat,json", i % 2);
        if (!x) {
            check("200 repeated calls stay non-NULL", 0);
            break;
        }
    }
    check("200 repeated calls stay non-NULL", 1);

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
