#!/usr/bin/env bash
# Refuses to commit unless both the author and the committer email are allowed personal
# addresses (tools/allowed_identities.txt). Run by pre-commit before every commit. `git var`
# resolves the identities exactly as git will record them, including --author and
# GIT_*_EMAIL overrides.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mapfile -t allowed < <(grep -v '^\s*#' "$here/allowed_identities.txt" | tr -d '\r' | grep .)

email_of() { git var "$1" 2>/dev/null | sed -n 's/.*<\(.*\)>.*/\1/p'; }

status=0
for role in AUTHOR COMMITTER; do
    email="$(email_of "GIT_${role}_IDENT")"
    ok=0
    for a in "${allowed[@]}"; do
        [[ "${email,,}" == "${a,,}" ]] && ok=1
    done
    if [[ $ok == 0 ]]; then
        echo "Commit blocked: ${role,,} email '${email:-<unset>}' is not an allowed personal address." >&2
        status=1
    fi
done
if [[ $status != 0 ]]; then
    echo "Set it for this repository with:  git config user.email \"${allowed[0]}\"" >&2
fi
exit $status
