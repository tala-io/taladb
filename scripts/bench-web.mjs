#!/usr/bin/env node
/**
 * TalaDB browser benchmark driver.
 *
 * Serves the repo over HTTP, launches headless Chrome on
 * scripts/bench-web/index.html, and collects the results the page POSTs back.
 * No browser-automation dependency required — plain Chrome.
 *
 *   pnpm --filter @taladb/web build   # build the WASM package first
 *   node scripts/bench-web.mjs [--json]
 *   node scripts/bench-web.mjs --vectors --cache-bytes 1048576 --json
 *   node scripts/bench-web.mjs --vectors --serve --port 3000 --json
 *
 * The page drives the @taladb/web worker (WASM + OPFS) over its message
 * protocol — the same path the `taladb` wrapper uses, so timings include the
 * full JS ↔ worker ↔ WASM round-trip.
 */
import { createServer } from 'node:http'
import { existsSync } from 'node:fs'
import { readFile, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { tmpdir, cpus, arch, platform } from 'node:os'
import { join, dirname, extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dir = dirname(fileURLToPath(import.meta.url))
const harnessRoot = resolve(__dir, '..')
const args = process.argv.slice(2)
const option = (name, fallback) => {
  const i = args.indexOf(name)
  if (i < 0) return fallback
  if (!args[i + 1] || args[i + 1].startsWith('--')) throw new Error(`missing ${name} value`)
  return args[i + 1]
}
const root = resolve(option('--repo', harnessRoot))

const CHROME_CANDIDATES = [
  process.env.CHROME_BIN,
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium-browser',
  '/usr/bin/chromium',
].filter(Boolean)

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
}

const fmtMs = (ms) => (ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : ms < 100 ? `${ms.toFixed(2)} ms` : `${ms.toFixed(0)} ms`)
const fmtOps = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : n.toFixed(0))

async function main() {
  const chrome = CHROME_CANDIDATES.find((c) => existsSync(c))
  const serve = args.includes('--serve')
  const vectors = args.includes('--vectors')
  if (!chrome && !serve) throw new Error('Chrome not found — set CHROME_BIN or use --serve')
  const query = new URLSearchParams()
  if (args.includes('--quick')) query.set('quick', '1')
  if (vectors) {
    const { settings } = await import('./bench-web/vector-workload.js')
    for (const name of ['count', 'dims', 'queries', 'cache-bytes', 'memory-hint-bytes', 'quantization', 'concurrency']) {
      const value = option(`--${name}`, null)
      if (value !== null) query.set(name, value)
    }
    if (args.includes('--pressure')) query.set('pressure', '1')
    settings(query.toString())
  }
  const requestedPort = Number(option('--port', '0'))
  if (!Number.isInteger(requestedPort) || requestedPort < 0 || requestedPort > 65535) throw new Error('invalid --port')

  let resolveResult, rejectResult
  const resultPromise = new Promise((res, rej) => { resolveResult = res; rejectResult = rej })

  const server = createServer(async (req, res) => {
    // Enable origin-wide memory measurements where the browser supports them.
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
    if (req.method === 'POST' && req.url?.startsWith('/__bench/')) {
      let body = ''
      for await (const chunk of req) body += chunk
      res.writeHead(204).end()
      const kind = req.url.slice('/__bench/'.length)
      const payload = JSON.parse(body || '{}')
      if (kind === 'progress') console.error(payload.msg)
      else if (kind === 'result') resolveResult(payload)
      else if (kind === 'error') rejectResult(new Error(payload.error))
      return
    }
    // Static files, repo-rooted. Path traversal is blocked by resolve+prefix
    // check; this server only ever binds 127.0.0.1 and lives for one run.
    let pathname
    try { pathname = decodeURIComponent((req.url ?? '/').split('?')[0]) } catch { return void res.writeHead(400).end() }
    // Run the candidate's harness against either checkout's WASM and worker.
    const base = pathname.startsWith('/scripts/bench-web/') ? harnessRoot : root
    const path = resolve(base, `.${pathname}`)
    if (!path.startsWith(base + sep)) return void res.writeHead(403).end()
    try {
      const data = await readFile(path)
      res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' })
      res.end(data)
    } catch {
      res.writeHead(404).end()
    }
  })
  await new Promise((res, reject) => { server.once('error', reject); server.listen(requestedPort, '127.0.0.1', res) })
  const port = server.address().port
  const url = `http://127.0.0.1:${port}/scripts/bench-web/${vectors ? 'vector' : 'index'}.html?${query}`

  const profile = await mkdtemp(join(tmpdir(), 'taladb-bench-chrome-'))
  console.error(serve ? `Open this URL in the browser to measure: ${url}` : `launching headless Chrome on ${url}`)
  const proc = serve ? null : spawn(chrome, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    `--user-data-dir=${profile}`,
    url,
  ], { stdio: 'ignore' })
  proc?.once('error', rejectResult)
  proc?.once('exit', code => { if (code) rejectResult(new Error(`Chrome exited with ${code}`)) })

  const timeout = setTimeout(() => rejectResult(new Error('benchmark timed out after 15 min')), 15 * 60 * 1000)

  try {
    const report = await resultPromise
    const { ua, opfs, rows } = report
    clearTimeout(timeout)

    const cpu = cpus()[0]?.model ?? 'unknown CPU'
    const chromeVer = /Chrome\/([\d.]+)/.exec(ua)?.[1] ?? '?'
    console.log(`\nTalaDB browser bench · ${ua} · ${serve ? 'manual browser/device' : `Chrome ${chromeVer} (headless) · ${cpu} · ${platform()} ${arch()}`}\n`)
    if (vectors) {
      console.log(`${report.config.count} vectors × ${report.config.dimensions} dimensions · ${report.config.cacheBytes ?? 'adaptive'} cache bytes · ${report.capabilities.storage}`)
      console.log('| Filter / mode | efSearch | First query ms | Warm p50/p95 ms | Recall@k |')
      console.log('|---|---:|---:|---:|---:|')
      for (const row of report.cases) console.log(`| ${row.filter} / ${row.mode} | ${row.efSearch} | ${row.firstQueryMs.toFixed(3)} | ${row.p50Ms.toFixed(3)}/${row.p95Ms.toFixed(3)} | ${(row.recallAtK * 100).toFixed(1)}% |`)
    } else {
      console.log(`OPFS ${opfs ? 'active' : 'UNAVAILABLE (in-memory fallback!)'}`)
      for (const r of rows) {
        if (r.section) {
          console.log(`\n### ${r.section}\n`)
          console.log('| Operation | Detail | Result |')
          console.log('|---|---|---|')
        } else {
          let value
          if (r.unit === 'opsPerSec') value = `${fmtOps(1000 / r.median)} ops/s`
          else if (r.unit === 'docsPerSec') value = `${fmtOps(r.batch * (1000 / r.median))} docs/s`
          else if (r.unit === 'ingest') value = `${fmtOps(r.n / (r.median / 1000))} docs/s`
          else value = fmtMs(r.median)
          console.log(`| ${r.name} | ${r.detail} | **${value}** |`)
        }
      }
    }
    if (process.argv.includes('--json')) {
      const output = option('--output', vectors ? 'vector-browser-benchmark.json' : 'bench-web-results.json')
      await writeFile(output, JSON.stringify(report, null, 2))
      console.error(`\nwrote ${output}`)
    }
  } finally {
    clearTimeout(timeout)
    proc?.kill('SIGKILL')
    server.close()
    await rm(profile, { recursive: true, force: true }).catch(() => {})
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
