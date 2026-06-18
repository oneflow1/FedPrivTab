<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, watch } from 'vue'

const API_BASE = `${window.location.protocol}//${window.location.hostname}:5000`
const STORAGE_KEY = 'fedprivtab.simple.state.v1'
const pages = ['首页', '客户端数据准备页', '集中式训练页', '分布式训练页', '结果分析页']
const clientIds = ['client-1', 'client-2', 'client-3', 'client-4']
const schemes = {
  centralized: '集中式 MLP',
  fedavg: 'FedAvg + MLP',
  dp_fedavg: 'DP-FedAvg + MLP'
}
const defaultConfigs = {
  centralized: { epochs: 50, batch_size: 128, lr: 0.05, lr_schedule: 'step_decay', lr_decay: 0.5, lr_step_size: 15, lr_min: 0.005, hidden_layers: 2, hidden_units: '64,32', activation: 'ReLU', seed: 42 },
  fedavg: { clients: 4, rounds: 50, local_epochs: 1, batch_size: 32, lr: 0.05, lr_schedule: 'step_decay', lr_decay: 0.5, lr_step_size: 15, lr_min: 0.005, hidden_layers: 2, hidden_units: '64,32', activation: 'ReLU', client_fraction: 1, dirichlet_alpha: 0.3, non_iid: true, seed: 42 },
  dp_fedavg: { clients: 4, rounds: 50, local_epochs: 1, batch_size: 32, lr: 0.03, lr_schedule: 'step_decay', lr_decay: 0.5, lr_step_size: 15, lr_min: 0.005, hidden_layers: 2, hidden_units: '64,32', activation: 'ReLU', client_fraction: 1, dirichlet_alpha: 0.3, clip_norm: 1, noise_multiplier: 0.1, epsilon: 4, delta: 1e-5, non_iid: true, seed: 42 }
}
const distributedModes = new Set(['fedavg', 'dp_fedavg'])

const state = reactive({
  user: null,
  sessionId: '',
  page: '首页',
  login: { username: 'admin', password: '', error: '' },
  toast: '',
  preprocess: {
    file: null,
    name: '',
    loading: false,
    processing: false,
    target: 'income',
    columns: [],
    rows: [],
    rowCount: 0,
    missingSummary: [],
    missingStrategies: {},
    numericColumns: [],
    scalerStrategies: {},
    dataset: null,
    job: null,
    message: '等待上传 CSV'
  },
  train: {
    loading: false,
    mode: 'centralized',
    configs: JSON.parse(JSON.stringify(defaultConfigs)),
    job: null,
    message: ''
  },
  clientPrep: clientIds.map(id => ({
    id,
    enabled: true,
    file: null,
    name: '',
    loading: false,
    processing: false,
    target: 'income',
    columns: [],
    rows: [],
    rowCount: 0,
    missingSummary: [],
    missingStrategies: {},
    numericColumns: [],
    scalerStrategies: {},
    prepared: null,
    status: 'upload',
    message: '等待上传 CSV'
  })),
  runs: [],
  detailRunId: '',
  report: ''
})

const latestDataset = computed(() => state.preprocess.dataset)
const missingPreprocessRows = computed(() => state.preprocess.missingSummary.filter(item => item.missing > 0))
const summarizedClients = computed(() => state.clientPrep.filter(client => client.enabled && client.prepared))
const preparedClients = computed(() => state.clientPrep.filter(client => client.enabled && client.prepared?.validation_result?.valid))
const preparedClientDatasets = computed(() => preparedClients.value.filter(client => client.prepared?.dataset_id))
const stalePreparedClients = computed(() => summarizedClients.value.filter(client => !client.prepared?.dataset_id && client.prepared?.sample_count))
const preparedClientRows = computed(() => preparedClientDatasets.value.reduce((total, client) => total + (Number(client.prepared?.sample_count) || 0), 0))
const canTrainDistributed = computed(() => preparedClientDatasets.value.length > 0)
const distributedTrainMode = computed({
  get: () => isDistributedMode(state.train.mode) ? state.train.mode : 'fedavg',
  set: mode => { state.train.mode = isDistributedMode(mode) ? mode : 'fedavg' }
})
const distributedTrainConfig = computed(() => state.train.configs[distributedTrainMode.value])
const distributedDisabledReason = computed(() => {
  if (state.train.loading) return '训练任务正在运行'
  if (canTrainDistributed.value) return ''
  if (stalePreparedClients.value.length) return '客户端摘要来自浏览器缓存，后端内存数据可能已过期；请在客户端数据准备页重新点击每个客户端的“预处理”。'
  return '请先进入客户端数据准备页，完成客户端预处理后再启动分布式训练。'
})
const currentDistributedData = computed(() => {
  if (preparedClientDatasets.value.length) return `${preparedClientDatasets.value.length} 客户端 / ${preparedClientRows.value} 行`
  if (stalePreparedClients.value.length) return `${stalePreparedClients.value.length} 客户端摘要需重新预处理`
  return '-'
})
const clientLocalDimensionList = computed(() => summarizedClients.value
  .filter(client => client.prepared?.feature_dim)
  .map(client => `${client.id}:${client.prepared.feature_dim}`)
  .join(' / '))
const federatedAlignedFeatureDim = computed(() => {
  const clientsForAlignment = preparedClientDatasets.value.length ? preparedClientDatasets.value : summarizedClients.value
  if (clientsForAlignment.length) {
    const columns = new Set()
    clientsForAlignment.forEach(client => {
      ;(client.prepared?.feature_columns || []).forEach(column => columns.add(column))
    })
    if (columns.size) return columns.size
  }
  const results = state.runs.flatMap(run => Object.values(run.results || {}))
  const result = results.find(item => item?.aligned_feature_dim || (item?.mode !== 'centralized' && item?.feature_dim))
  return result?.aligned_feature_dim || result?.feature_dim || null
})
const detailRun = computed(() => state.runs.find(run => run.id === state.detailRunId) || null)
const selectedRun = computed(() => detailRun.value)
const resultRows = computed(() => selectedRun.value ? Object.entries(selectedRun.value.results).map(([mode, result]) => ({ mode, label: schemes[mode], result, metrics: result.metrics || {} })) : [])
const schemeOrder = ['centralized', 'fedavg', 'dp_fedavg']
const metricOptions = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'f1', label: 'F1' },
  { key: 'auc', label: 'AUC' }
]
const comparisonRows = computed(() => schemeOrder.map(mode => resultRows.value.find(row => row.mode === mode)).filter(Boolean))
const currentScheme = computed(() => resultRows.value[0] || null)
const conclusionSummary = computed(() => {
  if (!selectedRun.value || !currentScheme.value) return '暂无可查看的训练指标。'
  const current = currentScheme.value
  const runTime = new Date(selectedRun.value.createdAt).toLocaleString()
  const metrics = current.metrics
  const risk = zeroPositiveClassMetrics(metrics)
    ? '；Precision、Recall、F1 均为 0，说明正类识别失败'
    : ''
  return `当前查看的是 ${current.label}（${runTime}），Accuracy ${formatNumber(metrics.accuracy)}，Precision ${formatNumber(metrics.precision)}，Recall ${formatNumber(metrics.recall)}，F1 ${formatNumber(metrics.f1)}，AUC ${formatNumber(metrics.auc)}${risk}。`
})
const selectedConfusionRows = computed(() => resultRows.value.map(row => {
  const matrix = normalizeMatrix(row.metrics.confusion_matrix)
  return matrix ? { ...row, matrix, diagnosis: confusionDiagnosis(row.metrics, matrix) } : null
}).filter(Boolean))
const selectedHasFederatedResult = computed(() => resultRows.value.some(row => row.mode === 'fedavg' || row.mode === 'dp_fedavg'))
const selectedClientDistribution = computed(() => {
  if (!selectedHasFederatedResult.value) return []
  const fromRun = resultRows.value.filter(row => row.mode === 'fedavg' || row.mode === 'dp_fedavg').flatMap(row => (row.result.client_distribution || []).map(item => ({
    client: item.client,
    negative: numberOrNull(item.negative),
    positive: numberOrNull(item.positive)
  }))).filter(item => item.client && item.negative !== null && item.positive !== null)
  if (fromRun.length) return fromRun
  return preparedClients.value.map(client => {
    const distribution = client.prepared?.label_distribution || {}
    const negative = labelCount(distribution, ['<=50K', '<=50K.', '0', 0, 'negative'])
    const positive = labelCount(distribution, ['>50K', '>50K.', '1', 1, 'positive'])
    return { client: client.id, negative, positive }
  }).filter(item => item.negative !== null && item.positive !== null)
})
const trainingCurveRows = computed(() => resultRows.value.map(row => ({ ...row, curves: curveSeries(row.result) })).filter(row => row.curves.length))
const reportHref = computed(() => state.report ? `data:text/markdown;charset=utf-8,${encodeURIComponent(state.report)}` : '')

function showToast(message) {
  state.toast = message
  setTimeout(() => { state.toast = '' }, 2400)
}

function isDistributedMode(mode) {
  return distributedModes.has(mode)
}

function ensureDistributedMode() {
  if (!isDistributedMode(state.train.mode)) state.train.mode = 'fedavg'
}

function navigateTo(page) {
  if (page === '分布式训练页') ensureDistributedMode()
  state.page = page
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const text = await response.text()
  const body = text ? JSON.parse(text) : {}
  if (!response.ok) throw new Error(body.error || response.statusText)
  return body
}

async function waitForJob(jobId, onUpdate) {
  for (;;) {
    const { job } = await api(`/jobs/${jobId}`)
    onUpdate?.(job)
    if (job.status === 'completed') return job.result
    if (job.status === 'failed') throw new Error(job.message || '任务失败')
    await new Promise(resolve => setTimeout(resolve, 800))
  }
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits).replace(/\.?0+$/, '')
}

function numberOrNull(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function labelCount(distribution, keys) {
  for (const key of keys) {
    if (distribution[key] !== undefined) return numberOrNull(distribution[key])
  }
  return null
}

function normalizeMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length < 2 || !Array.isArray(matrix[0]) || !Array.isArray(matrix[1])) return null
  const values = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]].map(numberOrNull)
  if (values.some(value => value === null)) return null
  return [
    { key: 'TN', label: 'TN <=50K', value: values[0] },
    { key: 'FP', label: 'FP >50K', value: values[1] },
    { key: 'FN', label: 'FN <=50K', value: values[2] },
    { key: 'TP', label: 'TP >50K', value: values[3] }
  ]
}

function zeroPositiveClassMetrics(metrics) {
  return [metrics?.precision, metrics?.recall, metrics?.f1].every(value => Number(value) === 0)
}

function confusionValue(matrix, key) {
  return matrix.find(cell => cell.key === key)?.value ?? 0
}

function confusionDiagnosis(metrics, matrix) {
  const tn = confusionValue(matrix, 'TN')
  const fp = confusionValue(matrix, 'FP')
  const fn = confusionValue(matrix, 'FN')
  const tp = confusionValue(matrix, 'TP')
  if ((tp === 0 && fp === 0) || zeroPositiveClassMetrics(metrics)) {
    return '模型没有预测出任何 >50K 样本，TP=0，FP=0，所以 Precision、Recall、F1 全为 0。'
  }
  const recall = numberOrNull(metrics?.recall)
  const recallText = recall === null ? 'Recall 未返回' : `Recall ${formatNumber(recall)}，${recall >= 0.7 ? '正类召回较高' : recall >= 0.4 ? '正类召回中等' : '正类召回偏低'}`
  return `正类检测：TP=${tp}，FN=${fn}，FP=${fp}；${recallText}。`
}

function barPercent(value, max) {
  if (!Number.isFinite(Number(value)) || !max) return 0
  return Math.max(0, Math.min(100, (Number(value) / max) * 100))
}

function metricMax(metric) {
  return Math.max(...comparisonRows.value.map(row => Number(row.metrics[metric])).filter(Number.isFinite), 1)
}

function distributionTotal(item) {
  return (Number(item.negative) || 0) + (Number(item.positive) || 0)
}

function distributionPercent(value, item) {
  const total = distributionTotal(item)
  return total ? `${Math.max(0, (Number(value) / total) * 100)}%` : '0%'
}

function curveSeries(result) {
  const history = result?.history || {}
  const loss = history.loss || result?.loss_history || []
  const accuracy = history.accuracy || result?.accuracy_history || []
  const curves = []
  if (Array.isArray(loss) && loss.length) curves.push({ key: 'loss', label: 'Loss', values: loss.map(Number).filter(Number.isFinite) })
  if (Array.isArray(accuracy) && accuracy.length) curves.push({ key: 'accuracy', label: 'Accuracy', values: accuracy.map(Number).filter(Number.isFinite) })
  return curves.filter(curve => curve.values.length)
}

function curvePoints(values) {
  if (!values.length) return ''
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  return values.map((value, index) => `${curveX(index, values)},${curveY(value, values)}`).join(' ')
}

function curveX(index, values) {
  return 10 + (index / Math.max(1, values.length - 1)) * 84
}

function curveY(value, values) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  return 86 - ((value - min) / range) * 68
}

function curveTicks(values) {
  if (!values.length) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (min === max) return [{ value: min, y: curveY(min, values) }]
  return [max, (max + min) / 2, min].map(value => ({ value, y: curveY(value, values) }))
}

function epochTicks(values) {
  const length = values.length
  if (!length) return []
  const indexes = length === 1 ? [0] : Array.from(new Set([0, Math.floor((length - 1) / 2), length - 1]))
  return indexes.map(index => ({ label: index + 1, x: curveX(index, values) }))
}

function curveTickStyle(tick) {
  return { top: `${tick.y}%` }
}

function epochTickStyle(tick) {
  return { left: `${tick.x}%` }
}

function nowId(prefix) {
  return `${prefix}-${Date.now().toString(36)}`
}

function applyInspection(target, metadata) {
  target.columns = metadata.columns || []
  target.rows = metadata.rows || []
  target.rowCount = metadata.row_count || 0
  target.target = metadata.target_column || target.target || 'income'
  target.missingSummary = metadata.missing_summary || []
  target.missingStrategies = { ...(metadata.missing_strategies || {}) }
  target.numericColumns = metadata.numeric_columns || []
  target.scalerStrategies = { ...(metadata.scaler_strategies || {}) }
}

function stalePreparedSummary(prepared) {
  if (!prepared) return null
  return {
    ...prepared,
    dataset_id: '',
    validation_result: {
      ...(prepared.validation_result || {}),
      valid: false,
      message: '浏览器仅恢复了预处理摘要；后端内存中的 dataset_id 可能已随服务重启失效，请重新点击预处理。'
    }
  }
}

function staleCentralDataset(dataset) {
  if (!dataset) return null
  return {
    ...dataset,
    dataset_id: '',
    validation: {
      ...(dataset.validation || {}),
      valid: false,
      message: '浏览器仅恢复了集中式数据摘要；后端内存数据可能已过期，请重新准备数据。'
    }
  }
}

async function inspectFile(target, file) {
  target.file = file
  target.name = file.name
  target.loading = true
  target.message = '后端正在读取 CSV 元数据...'
  try {
    const form = new FormData()
    form.append('file', file)
    form.append('target_column', target.target)
    form.append('preview_rows', '20')
    const metadata = await api('/preprocess/inspect', { method: 'POST', body: form })
    applyInspection(target, metadata)
    target.message = `已读取预览 ${metadata.preview_rows || 0} 行 / 共 ${metadata.row_count || 0} 行`
  } catch (error) {
    target.message = error.message
    target.status = 'failed'
  } finally {
    target.loading = false
  }
}

function readPreprocessFile(event) {
  const file = event.target.files?.[0]
  if (file) {
    state.preprocess.dataset = null
    inspectFile(state.preprocess, file)
  }
}

function readClientFile(event, client) {
  const file = event.target.files?.[0]
  if (file) {
    client.prepared = null
    client.status = 'upload'
    inspectFile(client, file)
  }
}

async function refreshInspection(target) {
  if (target.file) await inspectFile(target, target.file)
}

async function runPreprocess() {
  if (!state.preprocess.file) return showToast('请先上传 CSV')
  state.preprocess.processing = true
  try {
    const form = new FormData()
    form.append('file', state.preprocess.file)
    form.append('target_column', state.preprocess.target)
    form.append('missing_strategies', JSON.stringify(state.preprocess.missingStrategies))
    form.append('scaler_strategies', JSON.stringify(state.preprocess.scalerStrategies))
    form.append('summary_only', 'true')
    const queued = await api('/preprocess?async=true', { method: 'POST', body: form })
    state.preprocess.job = queued.job
    const body = await waitForJob(queued.job.job_id, job => {
      state.preprocess.job = job
      state.preprocess.message = `${job.message || job.status}（${job.progress || 0}%）`
    })
    state.preprocess.dataset = {
      dataset_id: body.dataset_id,
      rows: body.rows,
      target: body.target_column || state.preprocess.target,
      columns: body.columns || [],
      validation: body.validation,
      summary: body.summary,
      name: state.preprocess.name,
      createdAt: new Date().toISOString()
    }
    state.preprocess.message = body.validation?.message || '数据已准备'
    showToast('集中式数据已准备')
  } catch (error) {
    state.preprocess.message = error.message
  } finally {
    state.preprocess.processing = false
  }
}

async function runClientPreprocess(client) {
  if (!client.file) return showToast(`${client.id} 请先上传 CSV`)
  client.processing = true
  client.status = 'preprocess'
  try {
    const form = new FormData()
    form.append('file', client.file)
    form.append('target_column', client.target)
    form.append('missing_strategies', JSON.stringify(client.missingStrategies))
    form.append('scaler_strategies', JSON.stringify(client.scalerStrategies))
    form.append('summary_only', 'true')
    const queued = await api('/preprocess?async=true', { method: 'POST', body: form })
    const body = await waitForJob(queued.job.job_id, job => {
      client.message = `${job.message || job.status}（${job.progress || 0}%）`
    })
    client.prepared = {
      dataset_id: body.dataset_id,
      sample_count: body.summary?.sample_count || body.rows || 0,
      feature_dim: body.summary?.feature_dim || 0,
      feature_columns: (body.columns || []).filter(column => column !== (body.target_column || client.target) && column !== 'client_id'),
      label_distribution: body.summary?.label_distribution || {},
      validation_result: body.validation || { valid: false, message: '校验失败' },
      data_version: body.dataset_id || `${client.id}-${Date.now()}`
    }
    client.status = client.enabled && client.prepared.validation_result.valid ? 'ready' : 'disabled'
    client.message = client.prepared.validation_result.message
    showToast(`${client.id} 数据已准备`)
  } catch (error) {
    client.status = 'failed'
    client.message = error.message
  } finally {
    client.processing = false
  }
}

function trainPayload(mode) {
  const config = state.train.configs[mode]
  const payload = {
    mode,
    target_column: latestDataset.value?.target || state.preprocess.target,
    ...config
  }
  if (mode === 'fedavg' || mode === 'dp_fedavg') {
    if (preparedClientDatasets.value.length) {
      payload.client_datasets = preparedClientDatasets.value.map(client => ({
        client_id: client.id,
        dataset_id: client.prepared.dataset_id
      }))
      payload.client_dataset_ids = preparedClientDatasets.value.map(client => client.prepared.dataset_id)
      payload.clients = preparedClientDatasets.value.length
      payload.target_column = preparedClientDatasets.value[0]?.target || payload.target_column
    }
  } else {
    payload.dataset_id = latestDataset.value?.dataset_id
  }
  return payload
}

async function startTraining(mode = state.train.mode) {
  const isDistributed = mode === 'fedavg' || mode === 'dp_fedavg'
  if (!isDistributed && !latestDataset.value?.dataset_id) return showToast('请先完成集中式数据准备')
  if (isDistributed && !canTrainDistributed.value) return showToast(distributedDisabledReason.value)
  state.train.loading = true
  state.train.mode = mode
  try {
    const queued = await api('/train?async=true', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(trainPayload(mode))
    })
    state.train.job = queued.job
    const body = await waitForJob(queued.job.job_id, job => {
      state.train.job = job
      state.train.message = `${job.message || job.status}（${job.progress || 0}%）`
    })
    const run = {
      id: nowId('run'),
      createdAt: new Date().toISOString(),
      results: {
        [mode]: {
          ...body,
          data_version: isDistributed && preparedClientDatasets.value.length
            ? preparedClientDatasets.value.map(client => client.prepared.dataset_id).join(',')
            : latestDataset.value?.dataset_id,
          data_version_label: isDistributed && preparedClientDatasets.value.length
            ? `${preparedClientDatasets.value.length} 个客户端数据集`
            : latestDataset.value?.name,
          config: state.train.configs[mode]
        }
      }
    }
    state.runs.unshift(run)
    state.detailRunId = ''
    state.report = ''
    state.train.message = '训练完成'
    showToast('训练完成')
  } catch (error) {
    state.train.message = error.message
    showToast(error.message)
  } finally {
    state.train.loading = false
  }
}

async function login() {
  state.login.error = ''
  try {
    const body = await api('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.login)
    })
    if (body.username !== 'admin') {
      await api('/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': body.session_id },
        body: JSON.stringify({ session_id: body.session_id })
      })
      state.login.error = '请使用 admin 账号登录'
      return
    }
    state.user = { username: body.username, role: body.role }
    state.sessionId = body.session_id
    state.login.password = ''
  } catch (error) {
    state.login.error = error.message
  }
}

async function logout() {
  const sessionId = state.sessionId
  state.user = null
  state.sessionId = ''
  if (sessionId) {
    try {
      await api('/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': sessionId },
        body: JSON.stringify({ session_id: sessionId })
      })
    } catch {}
  }
}

async function generateReport() {
  if (!selectedRun.value) return
  const firstResult = Object.values(selectedRun.value.results)[0]
  const body = await api('/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ result: firstResult })
  })
  state.report = body.markdown
}

function openResultDetail(runId) {
  state.detailRunId = runId
  state.report = ''
}

function closeResultDetail() {
  state.detailRunId = ''
  state.report = ''
}

function handleKeydown(event) {
  if (event.key === 'Escape' && state.detailRunId) closeResultDetail()
}

function persistState() {
  const snapshot = {
    user: state.user,
    sessionId: state.sessionId,
    page: state.page,
    preprocess: {
      name: state.preprocess.name,
      target: state.preprocess.target,
      columns: state.preprocess.columns,
      rows: state.preprocess.rows,
      rowCount: state.preprocess.rowCount,
      missingSummary: state.preprocess.missingSummary,
      missingStrategies: state.preprocess.missingStrategies,
      numericColumns: state.preprocess.numericColumns,
      scalerStrategies: state.preprocess.scalerStrategies,
      dataset: state.preprocess.dataset,
      message: state.preprocess.message
    },
    clientPrep: state.clientPrep.map(client => ({
      id: client.id,
      enabled: client.enabled,
      name: client.name,
      target: client.target,
      columns: client.columns,
      rows: client.rows,
      rowCount: client.rowCount,
      missingSummary: client.missingSummary,
      missingStrategies: client.missingStrategies,
      numericColumns: client.numericColumns,
      scalerStrategies: client.scalerStrategies,
      prepared: client.prepared,
      status: client.status,
      message: client.message
    })),
    train: { mode: state.train.mode, configs: state.train.configs },
    runs: state.runs
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot))
  } catch {
    localStorage.removeItem(STORAGE_KEY)
  }
}

function restoreState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    if (!saved || typeof saved !== 'object') return
    if (saved.user && saved.sessionId) {
      state.user = saved.user
      state.sessionId = saved.sessionId
    }
    if (pages.includes(saved.page)) state.page = saved.page
    if (saved.preprocess) {
      Object.assign(state.preprocess, saved.preprocess, {
        file: null,
        loading: false,
        processing: false,
        job: null,
        dataset: staleCentralDataset(saved.preprocess.dataset),
        message: saved.preprocess.dataset ? '浏览器仅恢复了集中式数据摘要；如需训练请重新准备数据。' : (saved.preprocess.message || '等待上传 CSV')
      })
    }
    if (saved.train) {
      state.train.mode = saved.train.mode || 'centralized'
      state.train.configs = { ...JSON.parse(JSON.stringify(defaultConfigs)), ...(saved.train.configs || {}) }
    }
    if (Array.isArray(saved.runs)) state.runs = saved.runs
    if (Array.isArray(saved.clientPrep)) {
      state.clientPrep.forEach(client => {
        const item = saved.clientPrep.find(row => row.id === client.id)
        if (item) {
          const prepared = stalePreparedSummary(item.prepared)
          Object.assign(client, item, {
            file: null,
            loading: false,
            processing: false,
            prepared,
            status: prepared ? '需要重新预处理' : (item.status || 'upload'),
            message: prepared?.validation_result?.message || item.message || '等待上传 CSV'
          })
        }
      })
    }
    if (state.page === '分布式训练页') ensureDistributedMode()
  } catch {}
}

watch(state, persistState, { deep: true })
onMounted(() => {
  restoreState()
  window.addEventListener('keydown', handleKeydown)
})
onBeforeUnmount(() => window.removeEventListener('keydown', handleKeydown))
</script>

<template>
  <main v-if="!state.user" class="login-page">
    <section class="login-card">
      <h1>管理员登录</h1>
      <label>用户名<input v-model="state.login.username" /></label>
      <label>密码<input v-model="state.login.password" type="password" @keyup.enter="login" /></label>
      <button @click="login">登录</button>
      <p v-if="state.login.error" class="error">{{ state.login.error }}</p>
    </section>
  </main>

  <main v-else class="app-shell">
    <aside>
      <div class="brand-mark"><span>F</span><div><h2>FedPrivTab</h2><p>MLP Lab</p></div></div>
      <div class="user-panel"><p>{{ state.user.username }} · {{ state.user.role }}</p></div>
      <nav class="side-nav">
        <button v-for="item in pages" :key="item" :class="{ active: state.page === item }" @click="navigateTo(item)">{{ item }}</button>
      </nav>
    </aside>

    <section class="app-main">
      <header class="app-header">
        <div><strong>{{ state.user.username }}</strong><span>{{ state.user.role }}</span></div>
        <button class="ghost-button" @click="logout">退出登录</button>
      </header>

      <div class="content">
        <div v-if="state.toast" class="toast">{{ state.toast }}</div>

        <section v-if="state.page === '首页'">
          <h1>首页</h1>
          <div class="metrics">
            <article><b>数据集</b><strong>Adult</strong></article>
            <article><b>模型</b><strong>MLP</strong></article>
            <article><b>运行次数</b><strong>{{ state.runs.length }}</strong></article>
          </div>
          <div class="cards">
            <article v-for="label in schemes" :key="label"><h3>{{ label }}</h3><p>二分类表格训练流程</p></article>
          </div>
        </section>

        <section v-if="state.page === '客户端数据准备页'" class="compact-page">
          <h1>客户端数据准备页</h1>
          <p class="hint">只做上传、预览和预处理摘要，不在浏览器保存完整数据。</p>
          <div class="client-prep-grid">
            <article v-for="client in state.clientPrep" :key="client.id" class="client-prep-card">
              <header class="module-header"><h2>{{ client.id }}</h2><span class="status-pill">{{ client.status }}</span></header>
              <label class="toggle-line"><input type="checkbox" v-model="client.enabled" />参与训练</label>
              <label>CSV 文件<input type="file" accept=".csv" @change="event => readClientFile(event, client)" /></label>
              <label v-if="client.columns.length">目标列<select v-model="client.target" @change="refreshInspection(client)"><option v-for="column in client.columns" :key="column">{{ column }}</option></select></label>
              <details v-if="client.columns.length" class="details-panel">
                <summary>预处理配置</summary>
                <div class="grid compact-grid">
                  <label v-for="item in client.missingSummary.filter(row => row.missing > 0)" :key="item.column">{{ item.column }}<select v-model="client.missingStrategies[item.column]"><option value="drop">删除行</option><option value="mean">均值</option><option value="median">中位数</option><option value="mode">众数</option></select></label>
                  <label v-for="column in client.numericColumns" :key="column">{{ column }}<select v-model="client.scalerStrategies[column]"><option value="none">不缩放</option><option value="standard">标准化</option><option value="minmax">归一化</option></select></label>
                </div>
              </details>
              <button @click="runClientPreprocess(client)" :disabled="client.processing || !client.file">{{ client.processing ? '处理中...' : '预处理' }}</button>
              <p class="hint">{{ client.message }}</p>
              <div class="kv-grid">
                <div class="kv-item"><span>样本</span><strong>{{ client.prepared?.sample_count ?? (client.rowCount || '-') }}</strong></div>
                <div class="kv-item"><span>本地预处理维度</span><strong>{{ client.prepared?.feature_dim ?? '-' }}</strong></div>
                <div class="kv-item"><span>训练状态</span><strong>{{ client.prepared?.dataset_id ? '可训练' : (client.prepared ? '需重新预处理' : '-') }}</strong></div>
              </div>
              <p v-if="client.prepared?.feature_dim" class="hint">本地维度只反映该客户端出现过的独热类别；联邦训练会按所有客户端特征并集对齐。</p>
            </article>
          </div>
        </section>

        <section v-if="state.page === '集中式训练页'" class="compact-page">
          <h1>集中式训练页</h1>
          <section class="result-block compact-block">
            <header class="module-header"><h2>数据准备</h2><span>{{ state.preprocess.message }}</span></header>
            <label>CSV 文件<input type="file" accept=".csv" @change="readPreprocessFile" /></label>
            <label v-if="state.preprocess.columns.length">目标列<select v-model="state.preprocess.target" @change="refreshInspection(state.preprocess)"><option v-for="column in state.preprocess.columns" :key="column">{{ column }}</option></select></label>
            <details v-if="state.preprocess.columns.length" class="details-panel" open>
              <summary>预处理配置</summary>
              <p class="hint">预览 {{ state.preprocess.rows.length }} 行 / 共 {{ state.preprocess.rowCount }} 行</p>
              <table v-if="missingPreprocessRows.length"><thead><tr><th>字段</th><th>缺失数</th><th>处理方式</th></tr></thead><tbody><tr v-for="item in missingPreprocessRows" :key="item.column"><td>{{ item.column }}</td><td>{{ item.missing }}</td><td><select v-model="state.preprocess.missingStrategies[item.column]"><option value="drop">删除行</option><option value="mean">均值</option><option value="median">中位数</option><option value="mode">众数</option></select></td></tr></tbody></table>
              <div class="grid compact-grid"><label v-for="column in state.preprocess.numericColumns" :key="column">{{ column }}<select v-model="state.preprocess.scalerStrategies[column]"><option value="none">不缩放</option><option value="standard">标准化</option><option value="minmax">归一化</option></select></label></div>
            </details>
            <button @click="runPreprocess" :disabled="state.preprocess.processing || !state.preprocess.file">{{ state.preprocess.processing ? '后端处理中...' : '准备数据' }}</button>
            <span v-if="latestDataset" class="status-pill status-completed">已准备 {{ latestDataset.rows }} 行</span>
          </section>

          <section class="train-card standalone">
            <header class="module-header"><h2>集中式 MLP</h2><span>{{ state.train.message }}</span></header>
            <div class="grid compact-grid">
              <label>Epoch<input type="number" min="1" v-model.number="state.train.configs.centralized.epochs" /></label>
              <label>Batch<input type="number" min="4" v-model.number="state.train.configs.centralized.batch_size" /></label>
              <label>学习率<input type="number" step="0.001" v-model.number="state.train.configs.centralized.lr" /></label>
              <label>隐藏层<input v-model="state.train.configs.centralized.hidden_units" /></label>
              <label>激活函数<select v-model="state.train.configs.centralized.activation"><option>ReLU</option><option>Tanh</option><option>LeakyReLU</option></select></label>
            </div>
            <button @click="startTraining('centralized')" :disabled="state.train.loading || !latestDataset?.dataset_id" :title="latestDataset && !latestDataset.dataset_id ? '后端内存数据可能已过期，请重新准备数据。' : ''">{{ state.train.loading ? '训练中...' : '开始集中式训练' }}</button>
          </section>
        </section>

        <section v-if="state.page === '分布式训练页'" class="compact-page">
          <h1>分布式训练页</h1>
          <p class="hint">FedAvg 和 DP-FedAvg 只使用客户端数据准备页已预处理的数据；请先完成客户端预处理再启动训练。</p>
          <div class="metrics compact-metrics">
            <article><b>可训练客户端</b><strong>{{ preparedClientDatasets.length }}</strong></article>
            <article><b>当前数据</b><strong>{{ currentDistributedData }}</strong></article>
            <article><b>联邦对齐维度</b><strong>{{ federatedAlignedFeatureDim || '-' }}</strong></article>
          </div>
          <p v-if="stalePreparedClients.length && !preparedClientDatasets.length" class="inline-warning">检测到 {{ stalePreparedClients.length }} 个客户端只有缓存摘要，没有当前后端可用的 dataset_id。后端数据保存在进程内存中，服务重启后请回到客户端数据准备页重新点击“预处理”。</p>
          <p class="hint">客户端卡片展示的是本地预处理维度；类别独热列会在后端联邦训练前按所有客户端的并集对齐，缺失特征填 0。</p>
          <p v-if="clientLocalDimensionList" class="hint">本地维度：{{ clientLocalDimensionList }}；联邦对齐维度：{{ federatedAlignedFeatureDim || '-' }}。</p>
          <div class="train-card">
            <label>模式<select v-model="distributedTrainMode"><option value="fedavg">FedAvg + MLP</option><option value="dp_fedavg">DP-FedAvg + MLP</option></select></label>
            <p class="hint">Adult Non-IID 数据类别不均衡，推荐默认使用 50 轮和 step-decay 自适应学习率；轮数过少时早期结果可能停在多数类基线。</p>
            <div class="grid compact-grid">
              <label>轮数<input type="number" min="1" v-model.number="distributedTrainConfig.rounds" /></label>
              <label>客户端数<input type="number" min="2" v-model.number="distributedTrainConfig.clients" /></label>
              <label>本地 Epoch<input type="number" min="1" v-model.number="distributedTrainConfig.local_epochs" /></label>
              <label>学习率<input type="number" step="0.001" v-model.number="distributedTrainConfig.lr" /></label>
              <label>Non-IID α<input type="number" step="0.05" v-model.number="distributedTrainConfig.dirichlet_alpha" /></label>
              <label v-if="distributedTrainMode === 'dp_fedavg'">裁剪 C<input type="number" step="0.1" v-model.number="state.train.configs.dp_fedavg.clip_norm" /></label>
              <label v-if="distributedTrainMode === 'dp_fedavg'">噪声 σ<input type="number" step="0.1" v-model.number="state.train.configs.dp_fedavg.noise_multiplier" /></label>
              <label v-if="distributedTrainMode === 'dp_fedavg'">ε<input type="number" step="0.1" v-model.number="state.train.configs.dp_fedavg.epsilon" /></label>
            </div>
            <button @click="startTraining(distributedTrainMode)" :disabled="state.train.loading || !canTrainDistributed" :title="distributedDisabledReason">{{ state.train.loading ? '训练中...' : '开始分布式训练' }}</button>
            <p v-if="distributedDisabledReason" class="hint">{{ distributedDisabledReason }}</p>
          </div>
        </section>

        <section v-if="state.page === '结果分析页'">
          <h1>结果分析页</h1>
          <p v-if="!state.runs.length" class="empty-state">暂无训练结果。</p>
          <template v-else>
            <section class="result-block compact-block">
              <header class="module-header"><h2>原始结果表</h2><span>{{ state.runs.length }} 次运行</span></header>
              <table><thead><tr><th>时间</th><th>方案</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>操作</th></tr></thead><tbody><tr v-for="run in state.runs" :key="run.id"><td>{{ new Date(run.createdAt).toLocaleString() }}</td><td>{{ schemes[Object.keys(run.results)[0]] }}</td><td>{{ formatNumber(Object.values(run.results)[0].metrics?.accuracy) }}</td><td>{{ formatNumber(Object.values(run.results)[0].metrics?.f1) }}</td><td>{{ formatNumber(Object.values(run.results)[0].metrics?.auc) }}</td><td><button class="ghost-button" @click="openResultDetail(run.id)">查看</button></td></tr></tbody></table>
            </section>
          </template>
        </section>

        <div v-if="selectedRun" class="modal-backdrop" @click.self="closeResultDetail">
          <section class="modal-panel result-detail-modal">
            <header class="modal-header">
              <h2>实验结果详情</h2>
              <button class="close-button" @click="closeResultDetail">关闭</button>
            </header>

            <section class="result-block compact-block">
              <header class="module-header"><h2>实验结论</h2><span>{{ selectedRun.id }}</span></header>
              <p class="detail-conclusion">{{ conclusionSummary }}</p>
            </section>

            <section class="result-block compact-block">
              <header class="module-header"><h2>方案指标柱状图</h2><span>Accuracy / F1 / AUC</span></header>
              <div v-if="comparisonRows.length" class="multi-metric-chart">
                <article v-for="row in comparisonRows" :key="row.mode" class="scheme-bars">
                  <header><strong>{{ row.label }}</strong></header>
                  <div v-for="metric in metricOptions" :key="metric.key" class="metric-bar-row" :class="{ danger: metric.key === 'f1' && Number(row.metrics[metric.key]) === 0 }">
                    <span>{{ metric.label }}</span>
                    <div class="bar-track">
                      <div class="bar-fill" :class="[metric.key, { zero: Number(row.metrics[metric.key]) === 0 }]" :style="{ width: `${barPercent(row.metrics[metric.key], metricMax(metric.key))}%` }"></div>
                      <em v-if="Number(row.metrics[metric.key]) === 0">0</em>
                    </div>
                    <strong>{{ formatNumber(row.metrics[metric.key]) }}</strong>
                  </div>
                </article>
              </div>
              <p v-else class="empty-state">当前训练结果未返回 Accuracy、F1、AUC 指标。</p>
            </section>

            <section class="result-block compact-block">
              <header class="module-header"><h2>联邦客户端标签分布</h2><span>&lt;=50K / &gt;50K</span></header>
              <div v-if="selectedClientDistribution.length" class="stacked-chart">
                <div v-for="item in selectedClientDistribution" :key="item.client" class="stacked-row">
                  <div class="stacked-meta"><strong>{{ item.client }}</strong><span>{{ distributionTotal(item) }} 样本</span></div>
                  <div class="stacked-track">
                    <div class="segment segment-negative" :style="{ width: distributionPercent(item.negative, item) }">{{ item.negative }}</div>
                    <div class="segment segment-positive" :style="{ width: distributionPercent(item.positive, item) }">{{ item.positive }}</div>
                  </div>
                </div>
                <div class="legend"><span><i class="segment-negative"></i>&lt;=50K</span><span><i class="segment-positive"></i>&gt;50K</span></div>
              </div>
              <p v-else-if="!selectedHasFederatedResult" class="empty-state">集中式训练不包含客户端分布。</p>
              <p v-else class="empty-state">当前联邦训练结果未返回客户端分布，且没有已准备的客户端数据。</p>
            </section>

            <section class="result-block compact-block">
              <header class="module-header"><h2>运行详情</h2><span>{{ selectedRun.id }}</span></header>
              <table><thead><tr><th>方案</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead><tbody><tr v-for="row in resultRows" :key="row.mode"><td>{{ row.label }}</td><td>{{ formatNumber(row.metrics.accuracy) }}</td><td>{{ formatNumber(row.metrics.precision) }}</td><td>{{ formatNumber(row.metrics.recall) }}</td><td>{{ formatNumber(row.metrics.f1) }}</td><td>{{ formatNumber(row.metrics.auc) }}</td></tr></tbody></table>
            </section>

            <section class="result-block compact-block">
              <header class="module-header"><h2>混淆矩阵</h2><span>diagnostic cards</span></header>
              <div v-if="selectedConfusionRows.length" class="matrix-grid">
                <article v-for="row in selectedConfusionRows" :key="row.mode" class="matrix-card">
                  <h3>{{ row.label }}</h3>
                  <p class="matrix-diagnosis">{{ row.diagnosis }}</p>
                  <div class="confusion-grid">
                    <div v-for="cell in row.matrix" :key="cell.key" class="confusion-cell">
                      <span>{{ cell.label }}</span>
                      <strong>{{ cell.value }}</strong>
                    </div>
                  </div>
                </article>
              </div>
              <p v-else class="empty-state">当前训练结果未返回混淆矩阵。</p>
            </section>

            <section class="result-block compact-block">
              <header class="module-header"><h2>训练曲线</h2><span>real history arrays only</span></header>
              <div v-if="trainingCurveRows.length" class="curve-grid">
                <article v-for="row in trainingCurveRows" :key="row.mode" class="curve-card">
                  <h3>{{ row.label }}</h3>
                  <div v-for="curve in row.curves" :key="curve.key" class="curve-panel">
                    <div class="curve-panel-head"><span>{{ curve.label }}</span><small>{{ curve.values.length }} Epochs</small></div>
                    <div class="curve-chart">
                      <div class="curve-y-axis" aria-hidden="true">
                        <span v-for="tick in curveTicks(curve.values)" :key="`y-label-${tick.value}`" :style="curveTickStyle(tick)">
                          {{ formatNumber(tick.value, 3) }}
                        </span>
                      </div>
                      <div class="curve-plot">
                        <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" :aria-label="`${curve.label} training curve by Epoch`">
                          <line class="axis" x1="10" y1="86" x2="94" y2="86" />
                          <line class="axis" x1="10" y1="18" x2="10" y2="86" />
                          <g v-for="tick in curveTicks(curve.values)" :key="`y-${tick.value}`">
                            <line class="grid-line" x1="10" :y1="tick.y" x2="94" :y2="tick.y" />
                          </g>
                          <g v-for="tick in epochTicks(curve.values)" :key="`x-${tick.label}`">
                            <line class="tick-line" :x1="tick.x" y1="86" :x2="tick.x" y2="89" />
                          </g>
                          <polyline :points="curvePoints(curve.values)" />
                        </svg>
                        <div class="curve-axis-labels" aria-hidden="true">
                          <span v-for="tick in epochTicks(curve.values)" :key="`x-label-${tick.label}`" :style="epochTickStyle(tick)">
                            {{ tick.label }}
                          </span>
                        </div>
                        <div class="curve-axis-title">Epoch</div>
                      </div>
                    </div>
                    <small>Epoch 1: {{ formatNumber(curve.values[0]) }} → Epoch {{ curve.values.length }}: {{ formatNumber(curve.values[curve.values.length - 1]) }}</small>
                  </div>
                </article>
              </div>
              <p v-else class="empty-state">当前训练结果未返回真实逐轮曲线。</p>
            </section>

            <section class="result-block compact-block">
              <header class="module-header"><h2>生成报告</h2><span>Markdown</span></header>
              <button @click="generateReport">生成报告</button>
              <a v-if="state.report" class="download" :href="reportHref" :download="`${selectedRun.id}.md`">下载 Markdown</a>
              <pre v-if="state.report">{{ state.report }}</pre>
            </section>
          </section>
        </div>
      </div>
    </section>
  </main>
</template>
