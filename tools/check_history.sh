#!/usr/bin/env bash
# Checks that every commit in a range has an allowed author and committer email
# (tools/allowed_identities.txt). Used as a pre-push hook (the commits being pushed) and in
# CI (the whole history), so a commit made with another identity, e.g. in a clone without
# the pre-commit hooks or by an edit on GitHub, cannot land unnoticed.
#
#   tools/check_history.sh [<revision range>]     (default: the whole history)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mapfile -t allowed < <(grep -v '^\s*#' "$here/allowed_identities.txt" | tr -d '\r' | grep . | tr 'A-Z' 'a-z')

range="${1:-}"
if [[ -z "$range" && -n "${PRE_COMMIT_FROM_REF:-}" && -n "${PRE_COMMIT_TO_REF:-}" ]]; then
    # pre-push: the commits being pushed (the whole branch if it is new)
    if [[ "$PRE_COMMIT_FROM_REF" =~ ^0+$ ]]; then
        range="$PRE_COMMIT_TO_REF"
    else
        range="$PRE_COMMIT_FROM_REF..$PRE_COMMIT_TO_REF"
    fi
fi
range="${range:-HEAD}"

bad=0
while IFS='|' read -r sha author committer; do
    for email in "$author" "$committer"; do
        if ! printf '%s\n' "${allowed[@]}" | grep -qxF "${email,,}"; then
            echo "commit $sha: email '$email' is not an allowed identity" >&2
            bad=1
        fi
    done
done < <(git log --format='%h|%ae|%ce' "$range")
if [[ $bad != 0 ]]; then
    echo "Rewrite these commits with the allowed identity before pushing." >&2
fi
exit $bad
