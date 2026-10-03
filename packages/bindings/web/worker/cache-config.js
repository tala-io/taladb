// Device Memory is an approximate capability hint, not free/available RAM.
// Keep the caller's config string unchanged for multi-tab ownership checks.
export function adaptiveConfigJson(configJson, deviceMemoryGiB) {
  const config = JSON.parse(configJson ?? '{}')
  if (config.vector_cache_bytes != null || config.vector_cache_memory_bytes != null) return configJson
  if (typeof deviceMemoryGiB !== 'number' || !Number.isFinite(deviceMemoryGiB) || deviceMemoryGiB <= 0) return configJson
  const bytes = Math.floor(deviceMemoryGiB * 1024 ** 3)
  if (!Number.isSafeInteger(bytes) || bytes <= 0) return configJson
  return JSON.stringify({ ...config, vector_cache_memory_bytes: bytes })
}
