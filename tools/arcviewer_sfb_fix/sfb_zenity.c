/*
 * libStandaloneFileBrowser.so — drop-in replacement that does NOT link GTK.
 *
 * WHY THIS EXISTS
 * ---------------
 * ArcViewer (0.7.7 and 0.8.1, Unity 6000.0.67f1) aborts on this system the moment
 * the "open map" dialog is dismissed:
 *
 *     free(): invalid pointer
 *     Caught fatal signal - signo:6
 *     #7  g_object_unref
 *     #33 gtk_widget_unparent
 *     #39 gtk_container_remove
 *
 * The upstream StandaloneFileBrowser plugin links the SYSTEM libgtk-3 / libgobject
 * / libglib directly into the Unity process and runs a GTK dialog in-process.
 * Unity 6 ships its own copies of much of that stack and its own allocator, so
 * widget teardown frees a pointer the other allocator owns. Every frame in the
 * trace is GLib/GTK, and the abort fires immediately after the dialog's filter
 * strings are printed — i.e. before any map file is read. The maps were never
 * implicated: passing a path directly on the command line loads the same map in
 * 34 ms with no crash.
 *
 * THE FIX
 * -------
 * Keep the exact C ABI the C# side P/Invokes (see StandaloneFileBrowserLinux.cs)
 * but implement it by running `zenity` in a CHILD PROCESS and reading the chosen
 * path from its stdout. No GTK is loaded into Unity's address space, so the
 * conflicting-allocator teardown cannot happen. The dialog is somebody else's
 * process; if it crashes, ArcViewer does not.
 *
 * Deliberate choices:
 *   - fork + execvp with an argv array, never a shell, so paths containing
 *     spaces, quotes or newlines cannot be misparsed or injected.
 *   - a single reusable heap buffer for the returned string. The C# side copies
 *     it immediately (Marshal.PtrToStringAnsi), so ownership never crosses the
 *     boundary — which is the class of bug being fixed, and not one worth
 *     reintroducing in the fix.
 *   - NEVER return NULL. The caller does paths.Split((char)28) on the result and
 *     would throw a NullReferenceException; on cancel it expects "".
 *   - the async entry points call the callback inline and then return. A modal
 *     dialog blocks the UI anyway, and hopping threads to invoke a managed
 *     delegate from unattached native code is a worse risk than a frozen frame.
 *
 * Build:  ./build.sh        (gcc, no dependencies beyond libc)
 * Verify: ./selftest        (dlopens the .so and exercises the ABI)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

#define SEP_CHAR ((char)28)   /* what StandaloneFileBrowserLinux.cs splits on */
#define MAX_ARGV 64

typedef void (*AsyncCallback)(const char *path);

/* Single reusable result buffer. Returned to C#, which copies it at once. */
static char *g_result = NULL;

static const char *set_result(const char *s)
{
    free(g_result);
    g_result = strdup(s ? s : "");
    if (!g_result) {
        static char empty[] = "";   /* out of memory: still never return NULL */
        return empty;
    }
    return g_result;
}

/* Append "--file-filter=Name | *.a *.b" args parsed from SFB's filter string.
 * Format produced by GetFilterFromFileExtensionList():
 *     "Name;ext1,ext2|Name2;ext3"
 * Anything malformed is skipped rather than guessed at. */
static void add_filters(char **argv, int *argc, const char *filters, char **owned)
{
    if (!filters || !*filters)
        return;

    char *copy = strdup(filters);
    if (!copy)
        return;
    *owned = copy;

    char *group_state = NULL;
    for (char *group = strtok_r(copy, "|", &group_state);
         group && *argc < MAX_ARGV - 4;
         group = strtok_r(NULL, "|", &group_state)) {

        char *semi = strchr(group, ';');
        if (!semi)
            continue;
        *semi = '\0';
        const char *name = group;
        char *exts = semi + 1;

        /* "Name | *.zip *.dat" */
        size_t cap = strlen(name) + strlen(exts) * 4 + 32;
        char *arg = malloc(cap);
        if (!arg)
            continue;
        int n = snprintf(arg, cap, "--file-filter=%s |", name);

        char *ext_state = NULL;
        for (char *e = strtok_r(exts, ",", &ext_state); e;
             e = strtok_r(NULL, ",", &ext_state)) {
            if (!*e)
                continue;
            n += snprintf(arg + n, cap - (size_t)n, " *.%s", e);
        }
        argv[(*argc)++] = arg;
    }
}

/* Run zenity with argv; return its stdout (trailing newline stripped).
 * Returns "" on cancel, on non-zero exit, or if zenity is unavailable. */
static const char *run_zenity(char **argv)
{
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return set_result("");

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return set_result("");
    }

    if (pid == 0) {
        /* child */
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        /* keep stderr: zenity warnings land in ArcViewer's Player.log, which is
         * where anyone debugging this will already be looking */
        execvp("zenity", argv);
        _exit(127);           /* zenity missing -> parent sees "" */
    }

    close(pipefd[1]);

    size_t len = 0, cap = 4096;
    char *buf = malloc(cap);
    if (!buf) {
        close(pipefd[0]);
        waitpid(pid, NULL, 0);
        return set_result("");
    }
    for (;;) {
        if (len + 1024 > cap) {
            size_t ncap = cap * 2;
            char *nbuf = realloc(buf, ncap);
            if (!nbuf)
                break;
            buf = nbuf;
            cap = ncap;
        }
        ssize_t r = read(pipefd[0], buf + len, cap - len - 1);
        if (r > 0)
            len += (size_t)r;
        else if (r == 0 || errno != EINTR)
            break;
    }
    buf[len] = '\0';
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR)
        ;

    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
        buf[--len] = '\0';

    const char *out = "";
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
        out = buf;            /* exit 1 = user cancelled, 127 = zenity absent */

    const char *res = set_result(out);
    free(buf);
    return res;
}

static void free_owned(char **argv, int from, int to, char *owned)
{
    for (int i = from; i < to; i++)
        free(argv[i]);
    free(owned);
}

/* ---- exported ABI (must match StandaloneFileBrowserLinux.cs exactly) ---- */

void DialogInit(void)
{
    /* Nothing to initialise: the dialog lives in a child process. The upstream
     * plugin called gtk_init() here, which is where the trouble started. */
}

const char *DialogOpenFilePanel(const char *title, const char *directory,
                                const char *extension, int multiselect)
{
    char *argv[MAX_ARGV];
    int argc = 0;
    char sep[2] = { SEP_CHAR, '\0' };
    char sep_arg[32];
    snprintf(sep_arg, sizeof sep_arg, "--separator=%s", sep);

    argv[argc++] = "zenity";
    argv[argc++] = "--file-selection";
    char title_arg[512];
    snprintf(title_arg, sizeof title_arg, "--title=%s",
             (title && *title) ? title : "Open");
    argv[argc++] = title_arg;

    char dir_arg[4096];
    if (directory && *directory) {
        /* trailing slash tells zenity to open IN the directory */
        size_t n = strlen(directory);
        snprintf(dir_arg, sizeof dir_arg, "--filename=%s%s", directory,
                 (n && directory[n - 1] == '/') ? "" : "/");
        argv[argc++] = dir_arg;
    }
    if (multiselect) {
        argv[argc++] = "--multiple";
        argv[argc++] = sep_arg;
    }

    int filters_from = argc;
    char *owned = NULL;
    add_filters(argv, &argc, extension, &owned);
    argv[argc++] = "--file-filter=All Files | *";
    int filters_to = argc - 1;   /* the literal above is not heap-allocated */
    argv[argc] = NULL;

    const char *res = run_zenity(argv);
    free_owned(argv, filters_from, filters_to, owned);
    return res;
}

const char *DialogOpenFolderPanel(const char *title, const char *directory,
                                  int multiselect)
{
    char *argv[MAX_ARGV];
    int argc = 0;
    char sep[2] = { SEP_CHAR, '\0' };
    char sep_arg[32];
    snprintf(sep_arg, sizeof sep_arg, "--separator=%s", sep);

    argv[argc++] = "zenity";
    argv[argc++] = "--file-selection";
    argv[argc++] = "--directory";
    char title_arg[512];
    snprintf(title_arg, sizeof title_arg, "--title=%s",
             (title && *title) ? title : "Select Folder");
    argv[argc++] = title_arg;

    char dir_arg[4096];
    if (directory && *directory) {
        size_t n = strlen(directory);
        snprintf(dir_arg, sizeof dir_arg, "--filename=%s%s", directory,
                 (n && directory[n - 1] == '/') ? "" : "/");
        argv[argc++] = dir_arg;
    }
    if (multiselect) {
        argv[argc++] = "--multiple";
        argv[argc++] = sep_arg;
    }
    argv[argc] = NULL;
    return run_zenity(argv);
}

const char *DialogSaveFilePanel(const char *title, const char *directory,
                                const char *defaultName, const char *extension)
{
    char *argv[MAX_ARGV];
    int argc = 0;

    argv[argc++] = "zenity";
    argv[argc++] = "--file-selection";
    argv[argc++] = "--save";
    argv[argc++] = "--confirm-overwrite";
    char title_arg[512];
    snprintf(title_arg, sizeof title_arg, "--title=%s",
             (title && *title) ? title : "Save");
    argv[argc++] = title_arg;

    char name_arg[4096];
    if ((directory && *directory) || (defaultName && *defaultName)) {
        const char *d = (directory && *directory) ? directory : "";
        const char *f = (defaultName && *defaultName) ? defaultName : "";
        size_t n = strlen(d);
        snprintf(name_arg, sizeof name_arg, "--filename=%s%s%s", d,
                 (n && d[n - 1] != '/') ? "/" : "", f);
        argv[argc++] = name_arg;
    }

    int filters_from = argc;
    char *owned = NULL;
    add_filters(argv, &argc, extension, &owned);
    int filters_to = argc;
    argv[argc] = NULL;

    const char *res = run_zenity(argv);
    free_owned(argv, filters_from, filters_to, owned);
    return res;
}

void DialogOpenFilePanelAsync(const char *title, const char *directory,
                              const char *extension, int multiselect,
                              AsyncCallback cb)
{
    const char *r = DialogOpenFilePanel(title, directory, extension, multiselect);
    if (cb)
        cb(r);
}

void DialogOpenFolderPanelAsync(const char *title, const char *directory,
                                int multiselect, AsyncCallback cb)
{
    const char *r = DialogOpenFolderPanel(title, directory, multiselect);
    if (cb)
        cb(r);
}

void DialogSaveFilePanelAsync(const char *title, const char *directory,
                              const char *defaultName, const char *extension,
                              AsyncCallback cb)
{
    const char *r = DialogSaveFilePanel(title, directory, defaultName, extension);
    if (cb)
        cb(r);
}
