export class WorkerClient {
  constructor(url) {
    this.worker = new Worker(url, { type: 'module', name: 'taladb' })
    this.nextId = 1
    this.pending = new Map()
    this.worker.onmessage = (e) => {
      const { id, result, error } = e.data
      const p = this.pending.get(id)
      if (!p) return
      this.pending.delete(id)
      if (error !== undefined) p.reject(new Error(error))
      else p.resolve(result)
    }
    this.worker.onerror = (e) => {
      const err = new Error(`worker error: ${e.message ?? 'unknown'}`)
      for (const p of this.pending.values()) p.reject(err)
      this.pending.clear()
    }
  }
  call(op, args = {}) {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ id, op, ...args })
    })
  }
}
