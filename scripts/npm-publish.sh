#!/usr/bin/env bash
# Publish one npm package from the release workflow, then confirm the registry
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
(cd "$dir" && pnpm publish --no-git-checks --access public --tag "$tag")

for _ in $(seq 1 10); do
    if published; then
        echo "✓ $name@$version is on npm (dist-tag $tag)"
        exit 0
    fi
    sleep 6
done
echo "::error::$name@$version was published but npm does not serve it after 60 s"
exit 1
