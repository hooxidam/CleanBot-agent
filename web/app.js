const ICONS = {
    bot: '<rect x="4" y="7" width="16" height="12" rx="4"/><path d="M9 7V4h6v3M8 12h.01M16 12h.01M9 16h6"/>',
    messages: '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>',
    device: '<circle cx="12" cy="12" r="8"/><path d="M8 12h8M12 8v8M7 19l-1 2M17 19l1 2"/>',
    edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4z"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    logout: '<path d="M10 17l5-5-5-5M15 12H3M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>',
    panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    close: '<path d="m15 18-6-6 6-6"/>',
    send: '<path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    alert: '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 17h.01"/>',
    bulb: '<path d="M9 18h6M10 22h4"/><path d="M8.3 15.2A7 7 0 1 1 15.7 15.2C14.7 16 14.2 16.7 14 18h-4c-.2-1.3-.7-2-1.7-2.8Z"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    wrench: '<path d="M14.7 6.3a4 4 0 0 0-5-5L12 3.6 9.6 6 7.3 3.7a4 4 0 0 0 5 5L4 17l3 3 7.7-8.3a4 4 0 0 0 5-5L17.4 9 15 6.6z"/>',
    activity: '<path d="M3 12h4l2-7 4 14 2-7h6"/>',
    fileSearch: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h7"/><path d="M14 2v6h6M16 19l4 4M17.5 17.5a3.5 3.5 0 1 1-5 0 3.5 3.5 0 0 1 5 0"/>',
    history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>',
    tag: '<path d="M20 13 11 22l-9-9V2h11z"/><circle cx="7" cy="7" r="1"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    profile: '<path d="M20 21a8 8 0 0 0-16 0M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8"/>',
    document: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    trash: '<path d="M3 6h18M8 6V4h8v2M19 6l-1 15H6L5 6M10 11v6M14 11v6"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>'
};

function icon(name, className = '') {
    return `<svg class="icon ${className}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.info}</svg>`;
}

function hydrateIcons(root = document) {
    root.querySelectorAll('[data-icon]').forEach(element => {
        const name = element.dataset.icon;
        if (!ICONS[name]) return;
        const size = element.classList.contains('brand-mark') ? 'icon-lg' : '';
        element.innerHTML = icon(name, size);
        element.removeAttribute('data-icon');
    });
}

const $ = id => document.getElementById(id);
const state = {
    currentUser: null,
    currentSessionId: null,
    users: [],
    sessions: [],
    streaming: false,
    serviceOnline: false,
    devicePanelCollapsed: false,
    toolUsage: {},
    currentSource: null
};

// The SSE protocol exposes tool names and status, but not tool outputs.
// Labels below present only facts the frontend actually receives.
const TOOL_META = {
    rag_summarize: { icon: 'fileSearch', label: '检索维修知识', done: '维修知识已检索' },
    get_device_status: { icon: 'activity', label: '查询设备状态', done: '设备状态已获取' },
    get_error_history: { icon: 'history', label: '查询故障记录', done: '故障记录已获取' },
    get_device_info: { icon: 'tag', label: '获取设备信息', done: '设备信息已获取' },
    get_my_usage_record: { icon: 'chart', label: '查询使用记录', done: '使用记录已获取' },
    update_my_profile: { icon: 'profile', label: '更新用户信息', done: '用户信息已更新' },
    fill_context_for_report: { icon: 'document', label: '补充报告上下文', done: '报告上下文已补充' }
};

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function renderInline(value) {
    const symbols = [
        ['⚠️', 'alert', 'warning', '注意'],
        ['⚠', 'alert', 'warning', '注意'],
        ['💡', 'bulb', 'tip', '提示'],
        ['✅', 'check', 'success', '完成'],
        ['✔️', 'check', 'success', '完成'],
        ['❌', 'alert', 'error', '异常'],
        ['🔍', 'search', 'info', '查询'],
        ['🤖', 'bot', 'info', '助手']
    ];
    let rendered = value;
    symbols.forEach(([symbol, iconName, type, label]) => {
        rendered = rendered.split(symbol).join(
            `<span class="answer-symbol ${type}" role="img" aria-label="${label}" title="${label}">${icon(iconName, 'icon-sm')}</span>`
        );
    });
    return rendered
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function splitTableRow(line) {
    return line.trim().replace(/^\||\|$/g, '').split('|').map(cell => renderInline(cell.trim()));
}

function renderMarkdown(text) {
    const lines = escapeHtml(text).split('\n');
    let html = '';
    let listType = null;
    const closeList = () => {
        if (listType) {
            html += `</${listType}>`;
            listType = null;
        }
    };

    for (let i = 0; i < lines.length; i += 1) {
        const current = lines[i].trim();
        if (!current) {
            closeList();
            continue;
        }
        if (current.includes('|') && lines[i + 1]?.trim().match(/^\|?\s*:?-{3,}/)) {
            closeList();
            const headers = splitTableRow(current);
            i += 2;
            const rows = [];
            while (i < lines.length && lines[i].trim().includes('|')) {
                rows.push(splitTableRow(lines[i]));
                i += 1;
            }
            i -= 1;
            html += `<div class="table-scroll"><table><thead><tr>${headers.map(cell => `<th>${cell}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
            continue;
        }

        let match;
        if ((match = current.match(/^(#{1,4})\s+(.*)$/))) {
            closeList();
            const level = match[1].length;
            html += `<h${level}>${renderInline(match[2])}</h${level}>`;
        } else if ((match = current.match(/^[-*]\s+(.*)$/))) {
            if (listType !== 'ul') {
                closeList();
                html += '<ul>';
                listType = 'ul';
            }
            html += `<li>${renderInline(match[1])}</li>`;
        } else if ((match = current.match(/^(\d+)\.\s+(.*)$/))) {
            if (listType !== 'ol') {
                closeList();
                html += '<ol>';
                listType = 'ol';
            }
            html += `<li value="${match[1]}">${renderInline(match[2])}</li>`;
        } else if ((match = current.match(/^&gt;\s?(.*)$/))) {
            closeList();
            html += `<blockquote>${renderInline(match[1])}</blockquote>`;
        } else if (/^---+$/.test(current)) {
            closeList();
            html += '<hr>';
        } else {
            closeList();
            html += `<p>${renderInline(current)}</p>`;
        }
    }
    closeList();
    return html;
}

async function fetchJson(url, options = {}, fallback = '请求失败') {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || fallback);
    return data;
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `${icon(type === 'error' ? 'alert' : 'check')}<span>${escapeHtml(message)}</span>`;
    $('toastRegion').appendChild(toast);
    setTimeout(() => toast.remove(), type === 'error' ? 5200 : 3200);
}

function setServiceStatus(online) {
    state.serviceOnline = online;
    const element = $('serviceStatus');
    element.className = `service-status ${online ? 'online' : 'offline'}`;
    element.innerHTML = `<span class="status-dot"></span><span>${online ? '服务在线' : '服务不可用'}</span>`;
}

async function checkHealth() {
    try {
        const data = await fetchJson('/health', {}, '服务不可用');
        setServiceStatus(data.status === 'ok');
    } catch {
        setServiceStatus(false);
    }
}

async function loadUsers() {
    const grid = $('identityGrid');
    try {
        const data = await fetchJson('/users', {}, '无法加载演示用户');
        state.users = data.users || [];
        if (!state.users.length) {
            grid.innerHTML = '<div class="error-card">当前没有可用的演示用户，请检查模拟用户数据。</div>';
            return;
        }
        grid.innerHTML = state.users.map(user => `
            <button class="identity-option" type="button" data-user-id="${escapeHtml(user.user_id)}" title="${escapeHtml(user.feature)}">
                <span class="identity-avatar">${escapeHtml(String(user.user_id).slice(-2))}</span>
                <span style="min-width:0"><strong>用户 ${escapeHtml(user.user_id)}</strong><small>${escapeHtml(user.feature || '暂无使用环境')}</small></span>
            </button>`).join('');
        grid.querySelectorAll('.identity-option').forEach(button => {
            button.onclick = () => selectUser(state.users.find(user => String(user.user_id) === button.dataset.userId));
        });
    } catch (error) {
        grid.innerHTML = `<div class="error-card">${escapeHtml(error.message)}。请确认 FastAPI 服务已启动后刷新页面。</div>`;
    }
}

async function selectUser(user) {
    if (!user) return;
    state.currentUser = user;
    state.currentSessionId = null;
    state.toolUsage = {};
    localStorage.setItem('robocare_user_id', user.user_id);
    $('currentUserId').textContent = `用户 ${user.user_id}`;
    $('currentAvatar').textContent = String(user.user_id).slice(-2);
    $('identityScreen').classList.add('hidden');
    $('app').classList.remove('hidden');
    renderDevicePanel();
    await loadSessions(true);
}

function switchUser() {
    if (state.streaming) {
        showToast('请等待当前回答完成后再退出身份', 'error');
        return;
    }
    localStorage.removeItem('robocare_user_id');
    state.currentUser = null;
    state.currentSessionId = null;
    state.sessions = [];
    state.toolUsage = {};
    $('sessionList').innerHTML = '';
    $('messageList').innerHTML = '';
    $('app').classList.add('hidden');
    $('identityScreen').classList.remove('hidden');
    closeMobilePanels();
}

function formatSessionTime(value) {
    if (!value) return '刚刚';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    const today = new Date();
    if (date.toDateString() === today.toDateString()) {
        return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
}

function renderSessionList() {
    const keyword = $('sessionSearch').value.trim().toLowerCase();
    const sessions = state.sessions.filter(session => String(session.title || '').toLowerCase().includes(keyword));
    if (!sessions.length) {
        $('sessionList').innerHTML = `<div class="rail-empty">${keyword ? '没有匹配的会话' : '还没有对话，点击上方按钮开始诊断'}</div>`;
        return;
    }
    $('sessionList').innerHTML = sessions.map(session => `
        <div class="session-item ${session.session_id === state.currentSessionId ? 'active' : ''}" role="button" tabindex="0" data-session-id="${escapeHtml(session.session_id)}" title="${escapeHtml(session.title)}">
            <span class="session-icon">${icon('messages', 'icon-sm')}</span>
            <span class="session-copy"><span class="session-title">${escapeHtml(session.title || '新对话')}</span><span class="session-time">${escapeHtml(formatSessionTime(session.updated_at || session.created_at))}</span></span>
            <span class="icon-btn session-delete" role="button" tabindex="0" aria-label="删除会话" title="删除会话">${icon('trash', 'icon-sm')}</span>
        </div>`).join('');

    $('sessionList').querySelectorAll('.session-item').forEach(item => {
        item.onclick = event => {
            if (event.target.closest('.session-delete')) {
                deleteSession(item.dataset.sessionId);
                return;
            }
            selectSession(item.dataset.sessionId);
        };
        item.onkeydown = event => {
            if ((event.key === 'Enter' || event.key === ' ') && !event.target.closest('.session-delete')) {
                event.preventDefault();
                selectSession(item.dataset.sessionId);
            }
        };
        item.querySelector('.session-delete').onkeydown = event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                deleteSession(item.dataset.sessionId);
            }
        };
    });
}

async function loadSessions(autoSelect = false) {
    if (!state.currentUser) return;
    $('sessionList').innerHTML = '<div class="load-state"><span class="spinner"></span>正在加载会话</div>';
    try {
        const data = await fetchJson(`/sessions?user_id=${encodeURIComponent(state.currentUser.user_id)}`, {}, '无法加载会话');
        state.sessions = data.sessions || [];
        if (autoSelect && !state.currentSessionId) {
            if (state.sessions.length) await selectSession(state.sessions[0].session_id);
            else await newSession();
        } else {
            renderSessionList();
        }
    } catch (error) {
        $('sessionList').innerHTML = `<div class="rail-empty">${escapeHtml(error.message)}<br>请稍后重试</div>`;
        showToast(error.message, 'error');
    }
}

async function selectSession(sessionId) {
    if (state.streaming || !sessionId) return;
    state.currentSessionId = sessionId;
    state.toolUsage = {};
    renderSessionList();
    closeMobilePanels();
    const list = $('messageList');
    list.innerHTML = '<div class="load-state"><span class="spinner"></span>正在加载对话</div>';
    try {
        const data = await fetchJson(`/sessions/${encodeURIComponent(sessionId)}/messages`, {}, '无法加载对话记录');
        list.innerHTML = '';
        if (!(data.messages || []).length) renderEmptyState();
        else data.messages.forEach(message => addMessage(message.role, message.content, message.role === 'bot'));
        scrollMessages();
        renderDevicePanel();
    } catch (error) {
        list.innerHTML = `<div class="error-card">${escapeHtml(error.message)}</div>`;
    }
}

async function newSession() {
    if (!state.currentUser || state.streaming) return;
    try {
        const session = await fetchJson('/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.currentUser.user_id })
        }, '新建对话失败');
        state.currentSessionId = session.session_id;
        state.toolUsage = {};
        $('messageList').innerHTML = '';
        renderEmptyState();
        await loadSessions(false);
        $('input').focus();
        closeMobilePanels();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function askConfirm(title, message) {
    return new Promise(resolve => {
        const modal = $('confirmModal');
        $('confirmTitle').textContent = title;
        $('confirmMessage').textContent = message;
        modal.classList.remove('hidden');
        const settle = value => {
            modal.classList.add('hidden');
            resolve(value);
        };
        $('confirmOk').onclick = () => settle(true);
        $('confirmCancel').onclick = () => settle(false);
    });
}

async function deleteSession(sessionId) {
    if (state.streaming) {
        showToast('请等待当前回答完成', 'error');
        return;
    }
    if (!await askConfirm('删除这场对话？', '删除后聊天记录无法恢复。')) return;
    try {
        await fetchJson(`/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }, '删除会话失败');
        if (state.currentSessionId === sessionId) state.currentSessionId = null;
        await loadSessions(true);
        showToast('会话已删除');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function quickCard(iconName, title, note, prompt) {
    return `<button class="quick-card" type="button" data-prompt="${escapeHtml(prompt)}"><span class="quick-card-icon">${icon(iconName)}</span><span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(note)}</small></span></button>`;
}

function renderEmptyState() {
    $('messageList').innerHTML = `
        <div class="empty-state"><div class="empty-inner">
            <div class="device-hero-icon">${icon('device', 'icon-xl')}</div>
            <h2>设备遇到什么问题？</h2>
            <p>描述故障现象，我会结合设备状态、近期故障记录和维修知识协助排查。</p>
            <div class="quick-grid">
                ${quickCard('activity', '设备突然不工作了', '查询状态并定位可能故障', '我的扫地机器人突然不工作了，帮我检查一下')}
                ${quickCard('alert', '为什么一直提示 E5？', '分析错误码与处理方法', '机器人一直提示 E5，应该怎么处理？')}
                ${quickCard('history', '最近有哪些故障？', '查询近期故障记录', '帮我查询最近 7 天的故障记录')}
                ${quickCard('wrench', '怎么清理主刷？', '检索维护与清理知识', '扫地机器人的主刷应该怎么清理？')}
            </div>
        </div></div>`;
    bindQuickActions();
}

function bindQuickActions(root = document) {
    root.querySelectorAll('[data-prompt]').forEach(button => {
        button.onclick = () => sendQuickPrompt(button.dataset.prompt);
    });
}

function messageFrame(role) {
    const element = document.createElement('article');
    element.className = `message ${role === 'user' ? 'user' : 'agent'}`;
    element.innerHTML = `<span class="message-avatar">${icon(role === 'user' ? 'user' : 'bot', 'icon-sm')}</span><div class="message-body"><div class="message-label">${role === 'user' ? '您' : 'RoboCare'}</div><div class="answer"></div></div>`;
    $('messageList').appendChild(element);
    return element;
}

function addMessage(role, text, markdown = false) {
    const element = messageFrame(role === 'user' ? 'user' : 'agent');
    const answer = element.querySelector('.answer');
    if (markdown) answer.innerHTML = renderMarkdown(text);
    else answer.textContent = text;
    scrollMessages();
    return answer;
}

function addStreamingMessage() {
    const element = messageFrame('agent');
    const body = element.querySelector('.message-body');
    const answer = body.querySelector('.answer');
    const trace = document.createElement('section');
    trace.className = 'agent-trace hidden';
    trace.innerHTML = `<button class="trace-toggle" type="button" aria-expanded="true"><span class="trace-status-icon">${icon('wrench', 'icon-sm')}</span><span class="trace-title">Agent 正在准备</span><span class="trace-chevron">${icon('chevron', 'icon-sm')}</span></button><div class="trace-list"></div>`;
    body.insertBefore(trace, answer);
    trace.querySelector('.trace-toggle').onclick = () => {
        trace.classList.toggle('collapsed');
        trace.querySelector('.trace-toggle').setAttribute('aria-expanded', String(!trace.classList.contains('collapsed')));
    };
    return { answer, trace, traceList: trace.querySelector('.trace-list') };
}

function argumentSummary(name, args) {
    if (!args) return '';
    if (name === 'rag_summarize' && args.query) return `检索主题：${args.query}`;
    if (name === 'get_error_history' && args.days) return `范围：近 ${args.days} 天`;
    if (name === 'get_my_usage_record' && args.month) return `月份：${args.month}`;
    return '';
}

function addToolStep(trace, traceList, name, args) {
    const meta = TOOL_META[name] || { icon: 'wrench', label: '执行业务操作', done: '业务操作已完成' };
    trace.classList.remove('hidden', 'collapsed');
    const step = document.createElement('div');
    step.className = 'tool-step running';
    step.dataset.toolName = name;
    step.dataset.startedAt = String(Date.now());
    step.innerHTML = `<span class="tool-node"><span class="spinner"></span></span><div class="tool-step-head"><span class="tool-kind">${icon(meta.icon, 'icon-sm')}</span><strong>${escapeHtml(meta.label)}</strong></div><div class="tool-meta">${escapeHtml(argumentSummary(name, args) || '正在执行')}</div>`;
    traceList.appendChild(step);
    updateTraceSummary(trace, false);
    scrollMessages();
    return step;
}

function finishToolStep(step, name) {
    if (!step) return;
    const meta = TOOL_META[name] || { done: '业务操作已完成' };
    const duration = Math.max(0, Date.now() - Number(step.dataset.startedAt || Date.now()));
    step.classList.remove('running');
    step.classList.add('done');
    step.querySelector('.tool-node').innerHTML = icon('check', 'icon-sm');
    step.querySelector('.tool-meta').textContent = `${meta.done} · ${(duration / 1000).toFixed(1)}s`;
    state.toolUsage[name] = { completedAt: new Date(), duration };
    renderDevicePanel();
}

function failRunningSteps(traceList) {
    traceList.querySelectorAll('.tool-step.running').forEach(step => {
        step.classList.remove('running');
        step.classList.add('error');
        step.querySelector('.tool-node').innerHTML = icon('alert', 'icon-sm');
        step.querySelector('.tool-meta').textContent = '操作未完成，请稍后重试';
    });
}

function updateTraceSummary(trace, complete) {
    const total = trace.querySelectorAll('.tool-step').length;
    const done = trace.querySelectorAll('.tool-step.done').length;
    const errors = trace.querySelectorAll('.tool-step.error').length;
    const title = trace.querySelector('.trace-title');
    if (complete) {
        title.textContent = errors ? `Agent 完成 ${done} 个操作，${errors} 个未完成` : `Agent 已完成 ${done} 个操作`;
    } else {
        title.textContent = total ? `Agent 正在执行 · ${done}/${total}` : 'Agent 正在准备';
    }
}

function setStreaming(streaming) {
    state.streaming = streaming;
    $('input').disabled = streaming;
    $('sendBtn').disabled = streaming;
    $('sendBtn').innerHTML = streaming
        ? '<span class="spinner" style="border-color:rgba(255,255,255,.45);border-top-color:#fff"></span>'
        : icon('send');
    $('newChatBtn').disabled = streaming;
}

function sendQuickPrompt(prompt) {
    if (state.streaming) return;
    $('input').value = prompt;
    autoResizeInput();
    send();
}

function send() {
    const query = $('input').value.trim();
    if (!query || !state.currentSessionId || !state.currentUser || state.streaming) return;
    if ($('messageList').querySelector('.empty-state')) $('messageList').innerHTML = '';
    addMessage('user', query);
    $('input').value = '';
    autoResizeInput();
    setStreaming(true);

    const { answer, trace, traceList } = addStreamingMessage();
    answer.classList.add('stream-cursor');
    const pending = {};
    let raw = '';
    let hasToolCall = false;
    let finished = false;
    const url = `/chat/stream?query=${encodeURIComponent(query)}&session_id=${encodeURIComponent(state.currentSessionId)}&user_id=${encodeURIComponent(state.currentUser.user_id)}`;
    const source = new EventSource(url);
    state.currentSource = source;

    const finish = (errorMessage = '') => {
        if (finished) return;
        finished = true;
        source.close();
        state.currentSource = null;
        answer.classList.remove('stream-cursor');
        if (errorMessage) {
            failRunningSteps(traceList);
            if (!raw.trim()) answer.innerHTML = `<div class="error-card">${escapeHtml(errorMessage)}</div>`;
        } else {
            traceList.querySelectorAll('.tool-step.running').forEach(step => finishToolStep(step, step.dataset.toolName));
        }
        if (!trace.classList.contains('hidden')) {
            updateTraceSummary(trace, true);
            trace.classList.add('collapsed');
            trace.querySelector('.trace-toggle').setAttribute('aria-expanded', 'false');
        }
        setStreaming(false);
        $('input').focus();
        loadSessions(false);
        scrollMessages();
    };

    source.onmessage = event => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch {
            return;
        }
        if (data.type === 'token') {
            raw += data.content || '';
            answer.innerHTML = renderMarkdown(raw);
            scrollMessages();
        } else if (data.type === 'tool_start') {
            // Text before a tool call is transient model narration. It is not an auditable tool fact,
            // so remove it instead of presenting it as hidden reasoning.
            hasToolCall = true;
            raw = '';
            answer.innerHTML = '';
            (pending[data.name] ||= []).push(addToolStep(trace, traceList, data.name, data.args || {}));
        } else if (data.type === 'tool_end') {
            const step = (pending[data.name] || []).shift();
            finishToolStep(step, data.name);
            updateTraceSummary(trace, false);
        } else if (data.type === 'done') {
            if (hasToolCall && !raw.trim()) answer.textContent = '操作已完成。';
            finish();
        } else if (data.type === 'error') {
            finish('本次处理没有完成，请稍后重试。');
        }
    };
    source.onerror = () => finish('与售后服务的连接已中断，请检查服务后重试。');
}

function scrollMessages() {
    const container = $('messages');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

function autoResizeInput() {
    const input = $('input');
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
}

function deviceAction(iconName, label, prompt) {
    return `<button class="device-action" type="button" data-prompt="${escapeHtml(prompt)}">${icon(iconName, 'icon-sm')}<span>${escapeHtml(label)}</span>${icon('arrow', 'icon-sm')}</button>`;
}

function renderDevicePanel() {
    if (!state.currentUser) {
        $('deviceContent').innerHTML = '';
        return;
    }
    const capabilities = [
        ['get_device_status', 'activity', '设备实时状态'],
        ['get_error_history', 'history', '近期故障记录'],
        ['get_device_info', 'tag', '设备档案与保修'],
        ['get_my_usage_record', 'chart', '设备使用记录']
    ];
    $('deviceContent').innerHTML = `
        <section class="support-card">
            <div class="device-identity"><span class="device-icon">${icon('device', 'icon-lg')}</span><span><strong>用户 ${escapeHtml(state.currentUser.user_id)} 的绑定设备</strong><small>型号与设备编号将在 Agent 查询后由回答展示</small></span></div>
            <p class="environment-text"><strong>使用环境</strong><br>${escapeHtml(state.currentUser.feature || '暂无已知环境信息')}</p>
        </section>
        <section class="support-card"><h3>本次会话查询</h3><div class="capability-list">
            ${capabilities.map(([name, iconName, label]) => {
                const used = Boolean(state.toolUsage[name]);
                return `<div class="capability-row"><span class="capability-icon">${icon(iconName, 'icon-sm')}</span><span class="capability-copy"><strong>${label}</strong><small>${used ? 'Agent 已完成查询' : '由 Agent 按需获取'}</small></span><span class="capability-state ${used ? 'done' : ''}">${used ? '已查询' : '待查询'}</span></div>`;
            }).join('')}
        </div></section>
        <section class="support-card"><h3>快速诊断</h3><div class="quick-actions">
            ${deviceAction('activity', '检查设备当前状态', '帮我查询设备当前状态')}
            ${deviceAction('history', '查询最近 7 天故障', '帮我查询最近 7 天的故障记录')}
            ${deviceAction('tag', '查看设备与保修信息', '帮我查询设备型号和保修状态')}
        </div></section>
        <div class="data-note">${icon('info', 'icon-sm')}<span>设备数据只在需要时由 Agent 调用业务工具查询。本页面不会根据回答文本猜测或补造设备字段。</span></div>`;
    bindQuickActions($('deviceContent'));
}

function toggleDevicePanel() {
    if (window.innerWidth <= 980) {
        toggleDeviceDrawer();
        return;
    }
    state.devicePanelCollapsed = !state.devicePanelCollapsed;
    $('workspace').classList.toggle('device-collapsed', state.devicePanelCollapsed);
}

function toggleConversationDrawer() {
    const rail = $('conversationRail');
    const opening = !rail.classList.contains('mobile-open');
    closeMobilePanels();
    rail.classList.toggle('mobile-open', opening);
    $('mobileBackdrop').classList.toggle('hidden', !opening);
}

function toggleDeviceDrawer() {
    const panel = $('devicePanel');
    const opening = !panel.classList.contains('mobile-open');
    closeMobilePanels();
    panel.classList.toggle('mobile-open', opening);
    $('mobileBackdrop').classList.toggle('hidden', !opening);
}

function closeMobilePanels() {
    $('conversationRail').classList.remove('mobile-open');
    $('devicePanel').classList.remove('mobile-open');
    $('mobileBackdrop').classList.add('hidden');
}

$('sendBtn').onclick = send;
$('newChatBtn').onclick = newSession;
$('switchUserBtn').onclick = switchUser;
$('sessionSearch').oninput = renderSessionList;
$('input').addEventListener('input', autoResizeInput);
$('input').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        send();
    }
});
document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
        closeMobilePanels();
        if (!$('confirmModal').classList.contains('hidden')) $('confirmCancel').click();
    }
});
window.addEventListener('resize', () => {
    if (window.innerWidth > 980) closeMobilePanels();
});

(async function init() {
    hydrateIcons();
    await Promise.all([loadUsers(), checkHealth()]);
    const savedId = localStorage.getItem('robocare_user_id');
    const savedUser = state.users.find(user => String(user.user_id) === savedId);
    if (savedUser) await selectUser(savedUser);
})();
