import { settings, runVectorBenchmark } from './vector-workload.js'

const log = document.querySelector('#log')
const post = (kind, body) => fetch(`/__bench/${kind}`, {
  method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
}).catch(() => {})
const progress = msg => { log.textContent += `\n${msg}`; post('progress', { msg }) }

async function main() {
  const baseline = new URLSearchParams(location.search).get('baseline') === '1'
  const report = await runVectorBenchmark(settings(location.search), { progress, baseline })
  report.ua = navigator.userAgent
  report.device = { hardwareConcurrency: navigator.hardwareConcurrency, deviceMemoryGiB: navigator.deviceMemory ?? null }
  report.measuredAt = new Date().toISOString()
  log.textContent += `\n\n${JSON.stringify(report, null, 2)}`
  const download = document.querySelector('#download')
  download.href = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }))
  download.hidden = false
  await post('result', report)
}
main().catch(error => { log.textContent += `\nERROR: ${error.stack ?? error}`; post('error', { error: String(error.stack ?? error) }) })
