import { WorkerClient } from './worker-client.js'

export function settings(search) {
  const params = new URLSearchParams(search)
  const integer = (name, fallback, min, max) => {
    const value = Number(params.get(name) ?? fallback)
    if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`invalid ${name}`)
    return value
  }
  const quantization = params.get('quantization') ?? 'binary'
  if (!['none', 'scalar', 'binary'].includes(quantization)) throw new Error('invalid quantization')
  return {
    count: integer('count', 2000, 100, 100000),
    dimensions: integer('dims', 128, 1, 4096),
    queries: integer('queries', 30, 1, 1000),
    concurrency: integer('concurrency', 4, 1, 16),
    cacheBytes: integer('cache-bytes', 8 * 1024 * 1024, 0, 256 * 1024 * 1024),
    quantization,
    m: 8, efConstruction: 64, batchSize: 32, topK: 10, seed: 42,
  }
}

export function dataset(config) {
  let seed = config.seed
  const random = () => {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5
    return (seed >>> 0) / 4294967296
  }
  const normalize = values => {
    const norm = Math.sqrt(values.reduce((sum, x) => sum + x * x, 0)) || 1
    return values.map(x => x / norm)
  }
  const centers = Array.from({ length: 32 }, () => normalize(Array.from({ length: config.dimensions }, () => random() * 2 - 1)))
  const point = i => normalize(centers[i % centers.length].map(x => x + (random() * 2 - 1) * 0.2))
  const documents = Array.from({ length: config.count }, (_, i) => ({ ordinal: i, tenant: i % 10, bucket: i % 100, v: point(i) }))
  seed = config.seed ^ 0x12345
  return {
    documents,
    probes: Array.from({ length: config.queries }, (_, i) => point(i * 7)),
  }
}

export function percentiles(samples) {
  const sorted = [...samples].sort((a, b) => a - b)
  return { p50Ms: sorted[Math.floor(sorted.length / 2)], p95Ms: sorted[Math.floor(sorted.length * 0.95)] }
}

export async function originMemory() {
  if (!globalThis.crossOriginIsolated || !performance.measureUserAgentSpecificMemory) return null
  let timer
  try {
    return await Promise.race([
      performance.measureUserAgentSpecificMemory().then(result => result.bytes),
      new Promise(resolve => { timer = setTimeout(() => resolve(null), 5000) }),
    ])
  } catch { return null } finally { clearTimeout(timer) }
}

// Production worker + persistent storage. Timings include postMessage, JSON,
// WASM traversal, exact rescoring and returned documents. Synthetic clustered
// vectors exercise retrieval structure; they are not an embedding-quality test.
export async function runVectorBenchmark(config, {
  Client = WorkerClient, now = () => performance.now(), progress = () => {},
  memory = originMemory, workerUrl = '/packages/bindings/web/worker/taladb.worker.js',
} = {}) {
  const { documents, probes } = dataset(config)
  const dbName = `vector-bench-${Date.now()}-${Math.random().toString(36).slice(2)}.db`
  const configJson = JSON.stringify({ vector_cache_bytes: config.cacheBytes })
  let client
  const decode = value => typeof value === 'string' ? JSON.parse(value) : value
  const command = request => client.call('vectorCommand', { collection: 'vectors', requestJson: JSON.stringify(request) }).then(decode)
  const cacheStats = async () => {
    try { return await command({ op: 'cacheStats' }) } catch { return null } // older baseline
  }
  const open = async () => {
    client = new Client(workerUrl)
    await client.call('init', { dbName, configJson })
  }
  const close = async () => {
    if (!client) return
    try { await client.call('close') } finally {
      client.worker.terminate(); client = null
      // Wait for worker termination to release the file's ownership lock.
      if (globalThis.navigator?.locks?.request) await navigator.locks.request(`taladb:taladb_${dbName}.redb`, () => {})
    }
  }
  const search = (query, filter, mode, efSearch) => command({ op: 'search', field: 'v', query, topK: config.topK, filter, options: { mode, efSearch } })
  const filters = [{ name: 'all', value: null }, { name: 'tenant-10pct', value: { tenant: 0 } }, { name: 'bucket-1pct', value: { bucket: 0 } }]
  const cases = []
  try {
    await open()
    const capabilities = decode(await client.call('capabilities'))
    if (capabilities.storage !== 'opfs') throw new Error(`benchmark requires OPFS; got ${capabilities.storage}`)
    const originMemoryBeforeBytes = await memory()
    const insertStart = now()
    // Bound messages and transient JS/WASM copies on smaller devices.
    for (let i = 0; i < documents.length; i += 128) {
      await client.call('insertMany', { collection: 'vectors', docsJson: JSON.stringify(documents.slice(i, i + 128)) })
    }
    for (const field of ['tenant', 'bucket']) await client.call('createIndex', { collection: 'vectors', field })
    const insertMs = now() - insertStart
    const buildStart = now()
    await command({ op: 'create', field: 'v', dimensions: config.dimensions, options: { m: config.m, efConstruction: config.efConstruction, quantization: config.quantization }, deferBuild: true })
    const build = await command({ op: 'beginBuild', field: 'v', options: { m: config.m, efConstruction: config.efConstruction, quantization: config.quantization } })
    const steps = []
    for (;;) {
      const start = now()
      const state = await command({ op: 'stepBuild', field: 'v', id: build.id, batchSize: config.batchSize })
      steps.push(now() - start)
      progress(`build ${state.processed}/${state.total}`)
      if (state.state === 'ready') break
      if (state.state !== 'building') throw new Error(`build ${state.state}: ${state.error}`)
    }
    const buildMs = now() - buildStart
    // Ground truth is measured per filter. IDs differ across checkouts, so
    // fingerprints use stable ordinals and exactly rescored f32 score bits.
    const bits = new DataView(new ArrayBuffer(4))
    for (const filter of filters) {
      progress(`ground truth: ${filter.name}`)
      const truth = []
      for (const query of probes) truth.push((await search(query, filter.value, 'exact', 100)).hits)
      for (const efSearch of [64, 100, 200]) {
        await close()
        const reopenStart = now()
        await open()
        const reopenMs = now() - reopenStart
        const firstStart = now()
        const first = await search(probes[0], filter.value, 'ann', efSearch)
        const firstQueryMs = now() - firstStart
        if (first.execution.path !== 'hnsw' && first.execution.reason !== 'memoryBudget') throw new Error('ANN was not used')
        // Warm the fixed query sweep before measuring it.
        for (const query of probes) await search(query, filter.value, 'ann', efSearch)
        const times = []
        let recall = 0, distances = 0, fingerprint = 2166136261, memoryFallbacks = 0
        for (let i = 0; i < probes.length; i++) {
          const start = now()
          const result = await search(probes[i], filter.value, 'ann', efSearch)
          times.push(now() - start)
          if (result.execution.reason === 'memoryBudget') memoryFallbacks++
          else if (result.execution.path !== 'hnsw') throw new Error('ANN was not used')
          distances += result.execution.distanceComputations
          const ids = new Set(truth[i].map(hit => hit.document.ordinal))
          recall += ids.size ? result.hits.filter(hit => ids.has(hit.document.ordinal)).length / ids.size : 1
          for (const hit of result.hits) {
            bits.setFloat32(0, hit.score)
            fingerprint = Math.imul(fingerprint ^ hit.document.ordinal ^ bits.getUint32(0), 16777619) >>> 0
          }
        }
        // Model concurrent requests from a browser UI. The production worker
        // serializes core operations; these measure queueing, not Rust threads.
        const burstTimes = []
        let burstFallbacks = 0
        for (let i = 0; i < probes.length; i++) {
          await Promise.all(Array.from({ length: config.concurrency }, async (_, j) => {
            const start = now()
            const result = await search(probes[(i + j) % probes.length], filter.value, 'ann', efSearch)
            burstTimes.push(now() - start)
            if (result.execution.reason === 'memoryBudget') burstFallbacks++
            else if (result.execution.path !== 'hnsw') throw new Error('ANN was not used')
          }))
        }
        const stats = await cacheStats()
        if (stats && (stats.activeBytes !== 0 || stats.retainedBytes > stats.budgetBytes || stats.peakBytes > stats.memoryBudgetBytes)) {
          throw new Error('shared search memory accounting exceeded budget or leaked a reservation')
        }
        cases.push({ filter: filter.name, efSearch, reopenMs, firstQueryMs, ...percentiles(times), recallAtK: recall / probes.length, distances, fingerprint,
          firstMemoryFallback: first.execution.reason === 'memoryBudget', memoryFallbacks, cacheStats: stats,
          burst: { concurrency: config.concurrency, requests: burstTimes.length, ...percentiles(burstTimes), memoryFallbacks: burstFallbacks } })
        progress(`${filter.name} ef=${efSearch}: ${cases.at(-1).p50Ms.toFixed(2)} ms`)
      }
    }
    const originMemoryAfterBytes = await memory()
    return { schema: 1, workload: 'browser-vector', config, capabilities, insertMs, buildMs,
      buildStep: { ...percentiles(steps), count: steps.length }, cases,
      originMemoryBeforeBytes, originMemoryAfterBytes }
  } finally {
    await close()
    // A benchmark owns only its unique database. Remove its persisted data,
    // including when a build/query fails, so phone storage does not accumulate.
    if (globalThis.navigator?.storage?.getDirectory) {
      const root = await navigator.storage.getDirectory()
      await root.removeEntry(`taladb_${dbName}.redb`).catch(() => {})
    }
  }
}
