const LINES_PER_BATCH = 100000;
const LARGE_FILE_THRESHOLD_BYTES = 1024 * 1024 * 1024; // 1GB
const UPLOAD_PROGRESS_START = 5;
const UPLOAD_PROGRESS_END = 55;
const CREATE_PROGRESS_START = 55;

const state = {
  activeDatabase: null,
  activeStats: null,
  existingDatabases: [],
  loadedDatabases: [],
  existingDetails: [],
  createInProgress: false,
  createTask: null,
  createProgressTimer: null
};

const els = {
  toast: document.getElementById('toast'),
  refreshAllBtn: document.getElementById('refreshAllBtn'),

  activeDbBadge: document.getElementById('activeDbBadge'),
  statSubjects: document.getElementById('statSubjects'),
  statPredicates: document.getElementById('statPredicates'),
  statObjects: document.getElementById('statObjects'),
  statTriples: document.getElementById('statTriples'),

  existingDbSelect: document.getElementById('existingDbSelect'),
  allDatabaseStatsCards: document.getElementById('allDatabaseStatsCards'),

  createForm: document.getElementById('createForm'),
  loadForm: document.getElementById('loadForm'),
  createDbName: document.getElementById('createDbName'),
  createFileInput: document.getElementById('createFileInput'),
  chooseFileBtn: document.getElementById('chooseFileBtn'),
  createSubmitBtn: document.getElementById('createSubmitBtn'),
  loadSubmitBtn: document.getElementById('loadSubmitBtn'),
  deleteDbBtn: document.getElementById('deleteDbBtn'),
  selectedFileName: document.getElementById('selectedFileName'),

  runQueryBtn: document.getElementById('runQueryBtn'),
  queryInput: document.getElementById('queryInput'),

  metricElapsed: document.getElementById('metricElapsed'),
  metricRows: document.getElementById('metricRows'),
  metricCols: document.getElementById('metricCols'),
  metricDensity: document.getElementById('metricDensity'),

  resultDbBadge: document.getElementById('resultDbBadge'),
  resultMeta: document.getElementById('resultMeta'),
  resultTableHead: document.querySelector('#resultTable thead'),
  resultTableBody: document.querySelector('#resultTable tbody'),
  rawJson: document.getElementById('rawJson'),

  progressModal: document.getElementById('progressModal'),
  progressTitle: document.getElementById('progressTitle'),
  progressStepText: document.getElementById('progressStepText'),
  progressBarFill: document.getElementById('progressBarFill'),
  progressPercentText: document.getElementById('progressPercentText'),
  cancelCreateBtn: document.getElementById('cancelCreateBtn')
};

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value)))
    return '-';
  return new Intl.NumberFormat('zh-CN').format(Number(value));
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value)))
    return '-';
  return Number(value).toFixed(3);
}

function formatTripleValue(stats) {
  if (!stats)
    return '-';
  if (Number(stats.triple_count) === 0 && stats.triple_count_exact === false)
    return 'N/A';
  return formatNumber(stats.triple_count);
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0)
    return '-';
  if (bytes < 1024)
    return `${bytes} B`;

  const units = ['KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIdx = -1;
  while (size >= 1024 && unitIdx < units.length - 1) {
    size /= 1024;
    unitIdx += 1;
  }
  return `${size.toFixed(size >= 100 ? 0 : 2)} ${units[unitIdx]}`;
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.style.background = isError
    ? 'rgba(130, 27, 27, 0.92)'
    : 'rgba(19, 33, 47, 0.92)';
  els.toast.classList.add('show');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => els.toast.classList.remove('show'), 2600);
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData))
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';

  const response = await fetch(path, {
    headers,
    ...options
  });

  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`响应不是有效 JSON: ${text.slice(0, 200)}`);
  }

  if (!response.ok || data.ok === false)
    throw new Error(data.message || `请求失败 (${response.status})`);

  return data;
}

function setActiveDatabaseStats(stats) {
  els.statSubjects.textContent = formatNumber(stats?.subject_count);
  els.statPredicates.textContent = formatNumber(stats?.predicate_count);
  els.statObjects.textContent = formatNumber(stats?.object_count);
  els.statTriples.textContent = formatTripleValue(stats);
}

function renderDatabaseStatsCards() {
  const detailsMap = new Map(state.existingDetails.map((item) => [item.name, item]));
  els.allDatabaseStatsCards.innerHTML = state.existingDatabases.length
    ? state.existingDatabases.map((name) => {
      const detail = detailsMap.get(name) || { stats: {} };
      const stats = detail.stats || {};
      const activeTag = detail.active ? '<span class="snapshot-active">当前</span>' : '';
      const loadedTag = detail.loaded ? '<span class="snapshot-active">已加载</span>' : '';
      const tag = activeTag || loadedTag;
        return `
          <article class="snapshot-card">
            <div class="snapshot-head">
              <span class="snapshot-name">${name}</span>
              ${tag}
            </div>
            <div class="snapshot-sections">
              <section class="snapshot-section">
                <p class="snapshot-section-title">Triples</p>
                <p class="snapshot-section-value">${formatTripleValue(stats)}</p>
              </section>
              <div class="snapshot-size-grid">
                <div class="snapshot-item"><span class="k">索引总大小</span><span class="v">${formatBytes(detail.index_size_bytes)}</span></div>
                <div class="snapshot-item"><span class="k">字典总大小</span><span class="v">${formatBytes(detail.dictionary_size_bytes)}</span></div>
                <div class="snapshot-item"><span class="k">总占用空间</span><span class="v">${formatBytes(detail.total_size_bytes)}</span></div>
              </div>
            </div>
          </article>`;
      }).join('')
    : '<article class="snapshot-card">暂无数据库</article>';
}

function renderLoadSelect() {
  if (!state.existingDatabases.length) {
    els.existingDbSelect.innerHTML = '<option value="">没有可加载的数据库</option>';
    return;
  }
  els.existingDbSelect.innerHTML = state.existingDatabases
    .map((name) => `<option value="${name}">${name}</option>`)
    .join('');
}

function applyDashboardData(data) {
  state.activeDatabase = data.active_database || null;
  state.activeStats = data.active_database_stats || null;
  state.existingDatabases = data.existing_databases || [];
  state.loadedDatabases = data.loaded_databases || [];
  state.existingDetails = data.existing_database_details || [];

  els.activeDbBadge.textContent = state.activeDatabase
    ? `当前：${state.activeDatabase}`
    : '未激活数据库';

  setActiveDatabaseStats(state.activeStats);
  renderDatabaseStatsCards();
  renderLoadSelect();

  if (els.runQueryBtn.textContent !== '查询中...')
    els.runQueryBtn.disabled = state.createInProgress || !state.activeDatabase;
}

async function refreshDashboard(silent = false, force = false) {
  if (state.createInProgress && !force)
    return;

  try {
    const data = await api('/apex/databases');
    applyDashboardData(data);
    if (!silent)
      showToast('数据已刷新');
  } catch (error) {
    showToast(error.message, true);
  }
}

function resetResultPanel() {
  els.metricElapsed.textContent = '-';
  els.metricRows.textContent = '-';
  els.metricCols.textContent = '-';
  els.metricDensity.textContent = '-';
  els.resultDbBadge.textContent = '数据库：-';
  els.resultMeta.textContent = '';
  els.resultTableHead.innerHTML = '';
  els.resultTableBody.innerHTML = '';
  els.rawJson.textContent = '';
}

function renderQueryResult(data) {
  const vars = data.head?.vars || [];
  const rows = data.results?.bindings || [];
  const metrics = data.metrics || {};
  const queryStats = data.query_stats || {};
  const truncated = Boolean(data.results?.truncated ?? queryStats.truncated);
  const truncatedLimit = Number((data.results?.truncated_limit ?? queryStats.truncated_limit) || 1000);
  const totalRows = Number(queryStats.row_count ?? data.results?.count ?? rows.length);
  const returnedRows = Number(queryStats.returned_row_count ?? data.results?.returned_count ?? rows.length);

  els.resultDbBadge.textContent = `数据库：${data.database || '-'}`;
  els.metricElapsed.textContent = formatMs(metrics.elapsed_ms);
  els.metricRows.textContent = formatNumber(totalRows);
  els.metricCols.textContent = formatNumber(queryStats.column_count ?? vars.length);
  els.metricDensity.textContent = `${Number(queryStats.non_empty_ratio_percent || 0).toFixed(2)}%`;

  const visibleLimit = 1000;
  const shownRows = rows.slice(0, visibleLimit);
  const truncationText = truncated ? `（已截断至 ${formatNumber(truncatedLimit)} 行）` : '';
  els.resultMeta.textContent =
    `总行数 ${formatNumber(totalRows)}${truncationText}，展示 ${formatNumber(shownRows.length)} 行，` +
    `结果单元格 ${formatNumber(queryStats.cell_count || returnedRows * vars.length)}，` +
    `平均值长度 ${Number(queryStats.avg_value_chars || 0).toFixed(2)}`;

  els.resultTableHead.innerHTML = `<tr>${vars.map((v) => `<th>${v}</th>`).join('')}</tr>`;
  els.resultTableBody.innerHTML = shownRows
    .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(String(cell))}</td>`).join('')}</tr>`)
    .join('');

  if (!shownRows.length)
    els.resultTableBody.innerHTML = '<tr><td>无结果</td></tr>';

  els.rawJson.textContent = JSON.stringify(data, null, 2);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function isNtFilename(filename) {
  return /\.nt$/i.test(String(filename || '').trim());
}

function assertNotCreating() {
  if (!state.createInProgress)
    return true;
  showToast('创建进行中，请等待完成或取消', true);
  return false;
}

function openProgressModal(title) {
  els.progressTitle.textContent = title;
  els.progressModal.classList.remove('hidden');
  els.progressModal.setAttribute('aria-hidden', 'false');
}

function closeProgressModal() {
  els.progressModal.classList.add('hidden');
  els.progressModal.setAttribute('aria-hidden', 'true');
}

function updateProgress(percent, text) {
  const safe = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  els.progressBarFill.style.width = `${safe}%`;
  els.progressPercentText.textContent = `${safe}%`;
  els.progressStepText.textContent = text || '处理中...';
}

function setUiLocked(locked) {
  state.createInProgress = locked;
  document.body.classList.toggle('ui-locked', locked);

  els.refreshAllBtn.disabled = locked;
  els.createDbName.disabled = locked;
  els.createFileInput.disabled = locked;
  els.chooseFileBtn.disabled = locked;
  els.createSubmitBtn.disabled = locked;
  els.existingDbSelect.disabled = locked;
  els.loadSubmitBtn.disabled = locked;
  els.deleteDbBtn.disabled = locked;
  els.queryInput.disabled = locked;
  els.runQueryBtn.disabled = locked || !state.activeDatabase;
  els.cancelCreateBtn.disabled = !locked;
}

function stopCreateProgressPolling() {
  if (state.createProgressTimer) {
    clearInterval(state.createProgressTimer);
    state.createProgressTimer = null;
  }
}

function startCreateProgressPolling() {
  stopCreateProgressPolling();

  let inFlight = false;
  state.createProgressTimer = setInterval(async () => {
    if (inFlight || !state.createTask || state.createTask.phase !== 'creating')
      return;

    inFlight = true;
    try {
      const data = await api('/apex/databases/create/progress');
      const backendPercent = Number(data.progress_percent);
      const percent = Number.isFinite(backendPercent)
        ? Math.max(CREATE_PROGRESS_START, Math.min(99, backendPercent))
        : CREATE_PROGRESS_START;
      const message = data.message ? `${data.step || '创建中'}：${data.message}` : (data.step || '创建中...');
      updateProgress(percent, message);
    } catch {
      // Keep current progress UI even if one poll fails.
    } finally {
      inFlight = false;
    }
  }, 700);
}

function ensureNotAborted(signal) {
  if (signal?.aborted)
    throw new DOMException('Aborted', 'AbortError');
}

async function* fileBatchesByLines(file, linesPerBatch, signal) {
  const reader = file.stream().getReader();
  const decoder = new TextDecoder();

  let buffer = '';
  let batchText = '';
  let batchLines = 0;

  while (true) {
    ensureNotAborted(signal);
    const { value, done } = await reader.read();
    if (done)
      break;

    buffer += decoder.decode(value, { stream: true });

    let idx = buffer.indexOf('\n');
    while (idx !== -1) {
      const oneLine = buffer.slice(0, idx + 1);
      buffer = buffer.slice(idx + 1);

      batchText += oneLine;
      batchLines += 1;

      if (batchLines >= linesPerBatch) {
        yield batchText;
        batchText = '';
        batchLines = 0;
      }

      idx = buffer.indexOf('\n');
    }
  }

  buffer += decoder.decode();
  if (buffer.length > 0) {
    batchText += buffer;
    batchLines += 1;
  }

  if (batchLines > 0)
    yield batchText;
}

async function uploadNtFileInBatches(database, file, controller) {
  ensureNotAborted(controller.signal);

  if (file.size <= 0)
    throw new Error('上传文件为空，无法创建数据库');

  const splitByLines = file.size > LARGE_FILE_THRESHOLD_BYTES;
  const uploadModeText = splitByLines
    ? '文件大于 1GB，按每 10 万行分批上传...'
    : '文件不超过 1GB，单批上传...';
  updateProgress(2, uploadModeText);

  const init = await api('/apex/databases/upload/init', {
    method: 'POST',
    body: JSON.stringify({
      database,
      filename: file.name
    }),
    signal: controller.signal
  });

  const uploadId = init.upload_id;
  state.createTask.uploadId = uploadId;

  let uploadedBytes = 0;
  let batchCount = 0;
  if (splitByLines) {
    for await (const batchText of fileBatchesByLines(file, LINES_PER_BATCH, controller.signal)) {
      ensureNotAborted(controller.signal);

      const chunkBlob = new Blob([batchText], { type: 'text/plain' });
      const form = new FormData();
      form.append('upload_id', uploadId);
      form.append('chunk_index', String(batchCount));
      form.append('data_chunk', chunkBlob, `chunk_${batchCount}.nt`);

      await api('/apex/databases/upload/chunk', {
        method: 'POST',
        body: form,
        signal: controller.signal
      });

      uploadedBytes += chunkBlob.size;
      batchCount += 1;
      const ratio = Math.min(1, uploadedBytes / file.size);
      const progress = UPLOAD_PROGRESS_START + ((UPLOAD_PROGRESS_END - UPLOAD_PROGRESS_START) * ratio);
      updateProgress(progress, `正在上传第 ${batchCount} 批（每批 10 万行）`);
    }
  } else {
    const form = new FormData();
    form.append('upload_id', uploadId);
    form.append('chunk_index', '0');
    form.append('data_chunk', file, file.name || 'data.nt');

    await api('/apex/databases/upload/chunk', {
      method: 'POST',
      body: form,
      signal: controller.signal
    });
    uploadedBytes = file.size;
    batchCount = 1;
    updateProgress(UPLOAD_PROGRESS_END, '单批上传完成');
  }

  if (batchCount <= 0)
    throw new Error('上传文件为空，无法创建数据库');

  return {
    uploadId,
    totalBatches: batchCount,
    splitByLines
  };
}

function isCancelLikeError(error) {
  if (!error)
    return false;
  const msg = String(error.message || '');
  return error.name === 'AbortError' || /cancel/i.test(msg) || msg.includes('取消');
}

async function cancelCreateTask() {
  if (!state.createTask)
    return;

  if (state.createTask.cancelRequested)
    return;

  state.createTask.cancelRequested = true;
  els.cancelCreateBtn.disabled = true;
  updateProgress(Number(els.progressPercentText.textContent.replace('%', '')) || 5, '正在提交取消请求...');

  if (state.createTask.phase === 'uploading') {
    state.createTask.uploadController?.abort();
  }

  try {
    await api('/apex/databases/create/cancel', {
      method: 'POST',
      body: JSON.stringify({
        upload_id: state.createTask.uploadId || ''
      })
    });
  } catch {
    // Ignore cancel API errors.
  }

  if (state.createTask.phase === 'uploading')
    showToast('已取消上传与创建');
  else
    showToast('已提交取消请求，等待当前步骤完成');
}

async function createDatabase(event) {
  event.preventDefault();

  if (state.createInProgress)
    return;

  const database = els.createDbName.value.trim();
  const file = els.createFileInput.files?.[0] || null;

  if (!database) {
    showToast('请填写数据库名', true);
    return;
  }
  if (!file) {
    showToast('请先上传数据文件', true);
    return;
  }
  if (!isNtFilename(file.name)) {
    showToast('只支持 .nt 格式文件', true);
    return;
  }
  if (file.size <= 0) {
    showToast('上传文件为空，无法创建数据库', true);
    return;
  }

  const uploadController = new AbortController();
  state.createTask = {
    phase: 'uploading',
    uploadId: '',
    uploadController,
    cancelRequested: false
  };

  openProgressModal('创建数据库中');
  updateProgress(1, '准备开始任务...');
  setUiLocked(true);

  try {
    const uploadInfo = await uploadNtFileInBatches(database, file, uploadController);

    if (state.createTask.cancelRequested)
      throw new DOMException('Aborted', 'AbortError');

    state.createTask.phase = 'creating';
    const uploadSummary = uploadInfo.splitByLines
      ? `上传完成，共 ${formatNumber(uploadInfo.totalBatches)} 批，开始创建数据库...`
      : '上传完成，开始创建数据库...';
    updateProgress(58, uploadSummary);
    startCreateProgressPolling();

    const data = await api('/apex/databases/create', {
      method: 'POST',
      body: JSON.stringify({
        database,
        upload_id: uploadInfo.uploadId
      })
    });

    updateProgress(100, '数据库创建完成');
    showToast(data.message || '数据库创建成功');
    els.createFileInput.value = '';
    els.selectedFileName.textContent = '未选择文件';
    await refreshDashboard(true, true);
  } catch (error) {
    if (isCancelLikeError(error) || state.createTask.cancelRequested)
      showToast('创建已取消');
    else
      showToast(error.message, true);
  } finally {
    stopCreateProgressPolling();
    setUiLocked(false);
    closeProgressModal();
    state.createTask = null;
    await refreshDashboard(true, true);
  }
}

async function loadDatabase(event) {
  event.preventDefault();
  if (!assertNotCreating())
    return;

  const database = els.existingDbSelect.value;
  if (!database) {
    showToast('没有可加载数据库', true);
    return;
  }

  try {
    const data = await api('/apex/databases/load', {
      method: 'POST',
      body: JSON.stringify({ database })
    });
    showToast(data.message || '数据库加载成功');
    await refreshDashboard(true);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function deleteDatabase() {
  if (!assertNotCreating())
    return;

  const database = els.existingDbSelect.value;
  if (!database) {
    showToast('没有可删除数据库', true);
    return;
  }

  const confirmed = window.confirm(`确认删除数据库 “${database}”？该操作不可恢复。`);
  if (!confirmed)
    return;

  try {
    const data = await api('/apex/databases/delete', {
      method: 'POST',
      body: JSON.stringify({ database })
    });
    if (state.activeDatabase === database)
      resetResultPanel();
    showToast(data.message || '数据库删除成功');
    await refreshDashboard(true, true);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function runQuery() {
  if (!assertNotCreating())
    return;

  const query = els.queryInput.value.trim();
  if (!query) {
    showToast('请输入查询语句', true);
    return;
  }

  const payload = { query };

  els.runQueryBtn.disabled = true;
  els.runQueryBtn.textContent = '查询中...';

  try {
    const data = await api('/apex/query', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    renderQueryResult(data);
    await refreshDashboard(true);
    showToast(`查询完成，共 ${formatNumber(data.results?.count || 0)} 行`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    els.runQueryBtn.textContent = '运行查询';
    els.runQueryBtn.disabled = state.createInProgress || !state.activeDatabase;
  }
}

function bindEvents() {
  els.refreshAllBtn.addEventListener('click', () => refreshDashboard());
  els.createForm.addEventListener('submit', createDatabase);
  els.cancelCreateBtn.addEventListener('click', cancelCreateTask);

  els.chooseFileBtn.addEventListener('click', () => {
    if (!assertNotCreating())
      return;
    els.createFileInput.click();
  });

  els.createFileInput.addEventListener('change', () => {
    let file = els.createFileInput.files?.[0] || null;
    if (file && !isNtFilename(file.name)) {
      showToast('仅支持 .nt 文件，已取消选择', true);
      els.createFileInput.value = '';
      file = null;
    }
    els.selectedFileName.textContent = file ? file.name : '未选择文件';
  });

  els.loadForm.addEventListener('submit', loadDatabase);
  els.deleteDbBtn.addEventListener('click', deleteDatabase);
  els.runQueryBtn.addEventListener('click', runQuery);
}

function init() {
  bindEvents();
  resetResultPanel();
  refreshDashboard(true);
}

init();
