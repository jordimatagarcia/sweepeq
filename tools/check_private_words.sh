#!/usr/bin/env bash
# Refuses to commit files that contain words from a local, untracked block list.
#
# The list lives in .git/info/blocked-words (one case-insensitive word per line), so it never
# ends up in the repository itself. If the file doesn't exist, the check passes.
# `git rev-parse --git-path` finds it from a linked worktree too.
set -euo pipefail

list="$(git rev-parse --git-path info/blocked-words)"
[[ -s "$list" ]] || exit 0
[[ $# -gt 0 ]] || exit 0

# Ignore blank lines (they would match everything) and Windows line endings.
words="$(mktemp)"
trap 'rm -f "$words"' EXIT
tr -d '\r' <"$list" | grep -v '^\s*$' >"$words" || true
[[ -s "$words" ]] || exit 0

if matches="$(grep -H -I -n -i -F -w -f "$words" -- "$@")"; then
    echo "Commit blocked: staged files contain blocked words:" >&2
    echo "$matches" >&2
    exit 1
fi
exit 0
