#!/usr/bin/env bash
# Publish one npm package from the release workflow, then check the registry
# serves it. A version already on npm is skipped, so a re-run of a partly
# published release completes it; every other failure fails the release. The
# old `pnpm publish || echo 'Already published'` hid an expired token: npm
# answers that with a 404, and v0.11.9 never reached npm while the job passed.
#
# Usage: scripts/npm-publish.sh <package-dir>     (NODE_AUTH_TOKEN set)
set -euo pipefail

dir="$1"
name="$(node -p "require('./$dir/package.json').name")"
version="$(node -p "require('./$dir/package.json').version")"
published() { [[ "$(npm view "$name@$version" version 2>/dev/null)" == "$version" ]]; }

if published; then
    echo "$name@$version is already on npm; skipping."
    exit 0
fi

# A pre-release must not become what `npm install $name` resolves to.
tag=latest
[[ "$version" == *-* ]] && tag=next

# pnpm, not npm: only pnpm rewrites `workspace:*` dependencies on publish.
# Its exit status is the authority. A re-run can reach here while npm still
# hides a version it accepted minutes ago; npm then refuses the duplicate, which
# means the earlier publish worked.
if ! output="$(cd "$dir" && pnpm publish --no-git-checks --access public --tag "$tag" 2>&1)"; then
    echo "$output"
    if grep -qE 'EPUBLISHCONFLICT|cannot publish over|previously published' <<<"$output"; then
        echo "$name@$version was already published (npm has not served it yet); continuing."
        exit 0
    fi
    exit 1
fi
echo "$output"

# npm can take minutes to serve a large package ("being processed"), so the
# read-back only warns; v0.12.0 stopped here after a 60 s wait for @taladb/node.
for _ in $(seq 1 20); do
    if published; then
        echo "✓ $name@$version is on npm (dist-tag $tag)"
        exit 0
    fi
    sleep 15
done
echo "::warning::$name@$version was published, but npm did not serve it within 5 minutes; check it"
