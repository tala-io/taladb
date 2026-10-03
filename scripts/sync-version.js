#!/usr/bin/env node
import { execFileSync } from "child_process";
import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");

const rootPkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const version = rootPkg.version;

const packages = [
  "packages/clients/taladb",
  "packages/bindings/web",
  "packages/bindings/node",
  "packages/bindings/react-native",
  "packages/clients/react",
];

// `--check <version>`: change nothing; fail unless every file this script
// writes already carries <version> and CHANGELOG.md has a dated entry for it.
// The release workflow runs this against the tag before anything is built,
// because publishing takes its versions from these files, not from the tag.
if (process.argv[2] === "--check") {
  const expected = process.argv[3] ?? version;
  const read = (path) => readFileSync(resolve(root, path), "utf8");
  const found = [["package.json", version]];
  for (const pkg of packages) {
    found.push([`${pkg}/package.json`, JSON.parse(read(`${pkg}/package.json`)).version]);
  }
  found.push(["Cargo.toml", /^version\s*=\s*"([^"]*)"/m.exec(read("Cargo.toml"))?.[1]]);
  found.push(["Cargo.lock (taladb)", /name = "taladb"\nversion = "([^"]*)"/.exec(read("Cargo.lock"))?.[1]]);
  found.push(["docs/.vitepress/config.mts", /text:\s*"v(\d+\.\d+\.\d+[^"]*)"/.exec(read("docs/.vitepress/config.mts"))?.[1]]);
  const problems = found
    .filter(([, actual]) => actual !== expected)
    .map(([path, actual]) => `${path} has ${actual ?? "no version"}`);
  const heading = read("CHANGELOG.md").split("\n").find((line) => line.startsWith(`## ${expected} `));
  if (!heading) problems.push(`CHANGELOG.md has no "## ${expected} — <date>" entry`);
  else if (!/\d{4}-\d{2}-\d{2}/.test(heading)) problems.push(`CHANGELOG.md entry is not dated: "${heading}"`);
  if (problems.length) {
    console.error(`✗ not ready to release ${expected}:\n  ${problems.join("\n  ")}`);
    console.error("  Run `node scripts/sync-version.js` after setting package.json, and date the changelog.");
    process.exit(1);
  }
  console.log(`✓ every manifest, Cargo.lock, the docs badge and CHANGELOG.md are at ${expected}`);
  process.exit(0);
}

for (const pkg of packages) {
  const pkgPath = resolve(root, pkg, "package.json");
  const pkgJson = JSON.parse(readFileSync(pkgPath, "utf8"));
  pkgJson.version = version;

  // Sync taladb / @taladb/* dependency references, but leave workspace:* untouched
  for (const depField of [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
  ]) {
    if (!pkgJson[depField]) continue;
    for (const dep of Object.keys(pkgJson[depField])) {
      if (
        (dep === "taladb" || dep.startsWith("@taladb/")) &&
        pkgJson[depField][dep] !== "workspace:*"
      ) {
        pkgJson[depField][dep] = `^${version}`;
      }
    }
  }

  writeFileSync(pkgPath, JSON.stringify(pkgJson, null, 2) + "\n");
  console.log(`✓ ${pkg} → ${version}`);
}

// Sync Cargo.toml workspace version
const cargoPath = resolve(root, "Cargo.toml");
let cargo = readFileSync(cargoPath, "utf8");
cargo = cargo.replace(/^(version\s*=\s*)"[^"]*"/m, `$1"${version}"`);
writeFileSync(cargoPath, cargo);
console.log(`✓ Cargo.toml → ${version}`);

// Sync Cargo.lock — `cargo publish --locked` in the release workflow fails on
// a lockfile that still records the old workspace version. `-w` touches only
// the workspace's own crates, never third-party dependencies.
try {
  execFileSync("cargo", ["update", "-w"], { cwd: root, stdio: "inherit" });
  console.log(`✓ Cargo.lock → ${version}`);
} catch (err) {
  console.error(`✗ Cargo.lock not updated (${err.message}) — run \`cargo update -w\` before releasing`);
  process.exitCode = 1;
}

// Sync VitePress nav version badge
const vpConfigPath = resolve(root, "docs/.vitepress/config.mts");
let vpConfig = readFileSync(vpConfigPath, "utf8");
vpConfig = vpConfig.replace(/(text:\s*"v)\d+\.\d+\.\d+(")/, `$1${version}$2`);
writeFileSync(vpConfigPath, vpConfig);
console.log(`✓ docs/.vitepress/config.mts → v${version}`);
