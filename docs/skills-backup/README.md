# Backup of the project's Claude Code skills

**The authoritative copies live at `~/.claude/skills/{close,todo,quickstart}/SKILL.md`** — edit
those, not these. `~/.claude` is not a git repo, so these three skills existed in exactly one place
on disk with no history. They encode the project's working conventions (forward-only TODO, the
validate-before-recording discipline, the research loop), which is too much hard-won process to keep
unbacked.

Refresh with:

```bash
for s in close todo quickstart; do
  cp ~/.claude/skills/$s/SKILL.md docs/skills-backup/$s.SKILL.md
done
```

Backed up 2026-08-02, after `/close` was rewritten to stop inflating TODO.md.
