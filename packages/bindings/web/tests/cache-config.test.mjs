import { test } from 'node:test'
import assert from 'node:assert/strict'
import { adaptiveConfigJson } from '../worker/cache-config.js'

test('browser hints use bytes without losing caller settings or overriding explicit budgets', () => {
  assert.deepEqual(JSON.parse(adaptiveConfigJson(null, 8)), { vector_cache_memory_bytes: 8 * 1024 ** 3 })
  const raw = '{ "durability": {"flush_every_write":false}, "webhook":{"enabled":false} }'
  assert.deepEqual(JSON.parse(adaptiveConfigJson(raw, 0.25)), {
    durability: { flush_every_write: false }, webhook: { enabled: false }, vector_cache_memory_bytes: 256 * 1024 ** 2,
  })
  for (const config of ['{"vector_cache_bytes":0}', '{"vector_cache_bytes":1048576}', '{"vector_cache_memory_bytes":1073741824}']) {
    assert.equal(adaptiveConfigJson(config, 16), config)
  }
  for (const hint of [undefined, null, 0, -1, NaN, Infinity, '8', Number.MAX_SAFE_INTEGER]) {
    assert.equal(adaptiveConfigJson(null, hint), null)
    assert.equal(adaptiveConfigJson(raw, hint), raw)
  }
})
