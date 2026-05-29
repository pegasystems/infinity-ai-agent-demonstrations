'use strict';

const InspectorPanel = (() => {
  let accessToken = null;
  let agentCard = null;
  let requestHistory = [];
  let _historySeq = 0;

  function init() {
    document.getElementById('get-agent-info-btn').addEventListener('click', handleGetAgentInfo);
  }

  async function handleGetAgentInfo() {
    const validation = window.ConfigPanel.validate();
    if (!validation.valid) {
      window.setStatusBar('Validation errors: ' + validation.errors.join('; '), 'error');
      return;
    }

    const creds = window.ConfigPanel.getCredentials();
    requestHistory = [];
    _historySeq = 0;
    const btn = document.getElementById('get-agent-info-btn');
    btn.disabled = true;
    btn.textContent = 'Loading…';
    window.setStatusBar('Fetching OAuth token…', '');

    // Step 1: Get token
    let tokenResult;
    const tokenReqStart = Date.now();
    try {
      tokenResult = await window.ApiClient.getToken({
        accessTokenEndpoint: creds.accessTokenEndpoint,
        clientId: creds.clientId,
        clientSecret: creds.clientSecret,
      });
      accessToken = tokenResult.access_token;
      appendHistory({
        method: 'POST',
        endpoint: '/api/auth/token',
        status: 200,
        durationMs: Date.now() - tokenReqStart,
        request: { accessTokenEndpoint: creds.accessTokenEndpoint, clientId: creds.clientId, clientSecret: '***' },
        response: { ...tokenResult, access_token: tokenResult.access_token ? '***' : null },
      });
    } catch (err) {
      appendHistory({
        method: 'POST',
        endpoint: '/api/auth/token',
        status: err.status || 0,
        durationMs: Date.now() - tokenReqStart,
        request: { accessTokenEndpoint: creds.accessTokenEndpoint, clientId: creds.clientId, clientSecret: '***' },
        response: { error: err.error, message: err.message },
      });
      window.setStatusBar('Token fetch failed: ' + err.message, 'error');
      btn.disabled = false;
      btn.textContent = 'Get Agent Information';
      return;
    }

    // Step 2: Get agent info
    window.setStatusBar('Fetching agent information…', '');
    const infoReqStart = Date.now();
    const infoParams = {
      accessToken,
      protocol: creds.protocol,
      ...(creds.protocol === 'api' ? { baseUrl: creds.baseUrl, agentId: creds.agentId } : { baseUrl: creds.baseUrl, agentCardUrl: creds.agentCardUrl }),
    };

    try {
      const agentInfo = await window.ApiClient.getAgentInfo(infoParams);

      // Store agent card execute URL for A2A chat
      if (creds.protocol === 'a2a') {
        agentCard = { executeUrl: agentInfo.executeUrl, capabilities: agentInfo.capabilities };
      }

      appendHistory({
        method: 'POST',
        endpoint: '/api/agent/info',
        status: 200,
        durationMs: Date.now() - infoReqStart,
        request: { ...infoParams, accessToken: '***' },
        response: agentInfo,
      });

      renderAgentInfo(agentInfo);
      window.setStatusBar('Agent information loaded successfully.', 'success');
    } catch (err) {
      appendHistory({
        method: 'POST',
        endpoint: '/api/agent/info',
        status: err.status || 0,
        durationMs: Date.now() - infoReqStart,
        request: { ...infoParams, accessToken: '***' },
        response: { error: err.error, message: err.message },
      });
      window.setStatusBar('Agent info fetch failed: ' + err.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Get Agent Information';
  }

  function renderAgentInfo(info) {
    renderVisualization(info);

    const container = document.getElementById('agent-info-display');
    container.innerHTML = '';

    // ── Overview ──────────────────────────────────────────────────
    container.appendChild(_textSection('Agent Name', info.agentName));
    container.appendChild(_textSection('Description', info.description));

    const meta = [
      info.ruleName     && `<span class="meta-badge">${_esc(info.ruleName)}</span>`,
      info.className    && `<span class="meta-badge">${_esc(info.className)}</span>`,
      info.coachMode    && `<span class="meta-badge">Coach: ${_esc(info.coachMode)}</span>`,
      info.enableExternalAccess !== undefined && `<span class="meta-badge meta-badge-${info.enableExternalAccess ? 'green' : 'muted'}">External access: ${info.enableExternalAccess ? 'on' : 'off'}</span>`,
    ].filter(Boolean);
    if (meta.length) container.appendChild(_rawSection('Details', meta.join('')));

    // ── Model ─────────────────────────────────────────────────────
    if (info.model && info.model.name) {
      const m = info.model;
      const body = `
        <div class="kv-row"><span class="kv-key">Model</span><span class="kv-val">${_esc(m.name)}</span></div>
        <div class="kv-row"><span class="kv-key">Provider</span><span class="kv-val">${_esc(m.provider)}</span></div>
        <div class="kv-row"><span class="kv-key">Model ID</span><span class="kv-val">${_esc(m.modelId)}</span></div>
        ${m.description ? `<div class="kv-row"><span class="kv-key">Description</span><span class="kv-val">${_esc(m.description)}</span></div>` : ''}
      `;
      container.appendChild(_rawSection('Model Configuration', body));
    }

    // ── Prompts (HTML content from Pega) ──────────────────────────
    const p = info.prompts || {};
    if (p.user)    container.appendChild(_collapsibleSection('User Prompt',          _htmlBody(p.user)));
    if (p.initial) container.appendChild(_collapsibleSection('Initial Instructions', _htmlBody(p.initial)));
    if (p.system)  container.appendChild(_collapsibleSection('System Prompt',        _htmlBody(p.system), true));
    if (p.responseStyle) container.appendChild(_collapsibleSection('Response Style', _htmlBody(p.responseStyle), true));
    if (p.guardrails)    container.appendChild(_collapsibleSection('Guardrails',     _htmlBody(p.guardrails), true));

    // ── Examples ──────────────────────────────────────────────────
    if (info.examples && info.examples.length > 0) {
      const body = info.examples.map((ex, i) =>
        `<div class="example-row">
           <div class="example-num">${i + 1}</div>
           <div class="example-text">${_esc(ex.example || ex.instruction)}</div>
         </div>`
      ).join('');
      container.appendChild(_collapsibleSection(`Examples (${info.examples.length})`, body));
    }

    // ── Tool groups ───────────────────────────────────────────────
    const tools = info.tools || {};
    const toolGroups = [
      { label: 'Case Type Tools',  list: tools.caseTypes },
      { label: 'Knowledge Tools',  list: tools.knowledge },
      { label: 'Agent Tools',      list: tools.agents },
      { label: 'External Agents',  list: tools.externalAgents },
      { label: 'MCP Clients',      list: tools.mcpClients },
    ];
    toolGroups.forEach(({ label, list }) => {
      if (list && list.length > 0) {
        container.appendChild(_toolListSection(label, list));
      }
    });
  }

  // Plain escaped text section
  function _textSection(title, content) {
    const div = document.createElement('div');
    div.className = 'agent-info-section';
    const isEmpty = !content;
    div.innerHTML = `
      <div class="agent-info-section-title">${_esc(title)}</div>
      <div class="agent-info-section-body${isEmpty ? ' empty' : ''}">${_esc(content || '(none)')}</div>
    `;
    return div;
  }

  // Section with pre-built HTML body string (trusted internal content only)
  function _rawSection(title, bodyHtml) {
    const div = document.createElement('div');
    div.className = 'agent-info-section';
    div.innerHTML = `<div class="agent-info-section-title">${_esc(title)}</div>`;
    const body = document.createElement('div');
    body.className = 'agent-info-section-body';
    body.innerHTML = bodyHtml;
    div.appendChild(body);
    return div;
  }

  // Collapsible <details> section — starts closed unless openByDefault=false
  function _collapsibleSection(title, bodyHtml, startCollapsed = false) {
    const details = document.createElement('details');
    details.className = 'agent-info-section agent-info-collapsible';
    if (!startCollapsed) details.open = true;
    const summary = document.createElement('summary');
    summary.className = 'agent-info-section-title';
    summary.textContent = title;
    details.appendChild(summary);
    const body = document.createElement('div');
    body.className = 'agent-info-section-body';
    body.innerHTML = bodyHtml;
    details.appendChild(body);
    return details;
  }

  // Wrap Pega HTML prompt content in a styled container.
  // DOMPurify sanitizes before injection to prevent stored XSS from Pega prompt fields.
  function _htmlBody(html) {
    const clean = window.DOMPurify ? window.DOMPurify.sanitize(html) : html;
    return `<div class="prompt-html">${clean}</div>`;
  }

  // Tool list displayed as a compact table
  function _toolListSection(title, tools) {
    const rows = tools.map((t) => {
      const badges = [
        t.available !== undefined
          ? `<span class="meta-badge meta-badge-${t.available ? 'green' : 'red'}">${t.available ? 'Available' : 'Unavailable'}</span>`
          : '',
        t.askConfirmation
          ? `<span class="meta-badge meta-badge-yellow">Confirm</span>`
          : '',
        t.protocol
          ? `<span class="meta-badge">${_esc(t.protocol)}</span>`
          : '',
      ].filter(Boolean).join('');

      return `<div class="tool-row">
        <div class="tool-row-name">${_esc(t.name)}</div>
        <div class="tool-row-meta">
          <span class="tool-row-category">${_esc(t.category)}</span>
          ${badges}
        </div>
      </div>`;
    }).join('');

    return _collapsibleSection(`${title} (${tools.length})`, rows);
  }

  // ── Agent Visualization ──────────────────────────────────────────

  function renderVisualization(info) {
    const container = document.getElementById('agent-viz-display');
    container.innerHTML = '';

    const svg = _buildVizSVG(info);
    if (!svg) return;

    const details = document.createElement('details');
    details.className = 'viz-section';
    details.open = true;

    const summary = document.createElement('summary');
    summary.textContent = 'Agent Visualization';
    details.appendChild(summary);

    const scroll = document.createElement('div');
    scroll.className = 'viz-scroll';
    scroll.innerHTML = svg;
    details.appendChild(scroll);

    container.appendChild(details);
  }

  function _buildVizSVG(info) {
    // Tool group definitions: label, key in info.tools, colour
    const GROUPS = [
      { label: 'Case Type Tools',  key: 'caseTypes',      color: '#7c3aed' },
      { label: 'Knowledge Tools',  key: 'knowledge',      color: '#0891b2' },
      { label: 'Agent Tools',      key: 'agents',         color: '#059669' },
      { label: 'External Agents',  key: 'externalAgents', color: '#d97706' },
      { label: 'MCP Clients',      key: 'mcpClients',     color: '#dc2626' },
    ].filter(g => info.tools && info.tools[g.key] && info.tools[g.key].length > 0);

    if (!GROUPS.length) return null;

    // Layout constants
    const PAD    = 24;
    const NODE_W = 168;
    const NODE_H = 36;
    const H_GAP  = 72;   // horizontal gap between columns
    const V_GAP  = 10;   // vertical gap between nodes in same column
    const RX     = 6;    // corner radius

    const COL_X = [
      PAD,
      PAD + NODE_W + H_GAP,
      PAD + 2 * (NODE_W + H_GAP),
    ];

    // ── Build type-node and tool-node data ───────────────────────
    let curY = PAD;
    const typeNodes = [];
    const toolNodes = [];

    GROUPS.forEach(group => {
      const tools = info.tools[group.key];
      const groupStartY = curY;

      tools.forEach(tool => {
        toolNodes.push({
          x: COL_X[2],
          y: curY,
          label: tool.name || '(unnamed)',
          color: group.color,
        });
        curY += NODE_H + V_GAP;
      });

      const groupEndY = curY - V_GAP;
      typeNodes.push({
        x: COL_X[1],
        y: (groupStartY + groupEndY) / 2 - NODE_H / 2,
        label: group.label,
        color: group.color,
        firstTool: toolNodes.length - tools.length,
        toolCount: tools.length,
      });
    });

    const svgH = Math.max(curY - V_GAP + PAD, NODE_H + PAD * 2);
    const svgW = PAD + 3 * NODE_W + 2 * H_GAP + PAD;
    const agentY = svgH / 2 - NODE_H / 2;

    // ── SVG helpers ──────────────────────────────────────────────
    function _s(str) {
      return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function truncate(str, max) {
      return str.length > max ? str.slice(0, max - 1) + '…' : str;
    }

    // Cubic-bezier connector from right-centre of A to left-centre of B
    function connector(ax, ay, bx, by, color) {
      const x1 = ax + NODE_W, y1 = ay + NODE_H / 2;
      const x2 = bx,          y2 = by + NODE_H / 2;
      const cx = (x1 + x2) / 2;
      return `<path d="M${x1} ${y1} C${cx} ${y1} ${cx} ${y2} ${x2} ${y2}" `
           + `stroke="${_s(color)}" stroke-width="1.5" fill="none" opacity="0.45"/>`;
    }

    // Rounded-rect node with centred label
    function node(x, y, label, color, bold) {
      const tx = x + NODE_W / 2;
      const ty = y + NODE_H / 2;
      const fw = bold ? '600' : '400';
      const text = truncate(label, 24);
      return `<rect x="${x}" y="${y}" width="${NODE_W}" height="${NODE_H}" rx="${RX}" `
           + `fill="${_s(color)}" fill-opacity="0.12" stroke="${_s(color)}" stroke-width="1.5"/>`
           + `<text x="${tx}" y="${ty}" text-anchor="middle" dominant-baseline="middle" `
           + `fill="${_s(color)}" font-size="11" font-weight="${fw}" `
           + `font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">${_s(text)}</text>`;
    }

    // ── Assemble SVG ─────────────────────────────────────────────
    const parts = [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${svgW}" height="${svgH}" `
      + `viewBox="0 0 ${svgW} ${svgH}">`,
    ];

    // Connectors (drawn first, behind nodes)
    typeNodes.forEach(tn => {
      parts.push(connector(COL_X[0], agentY, tn.x, tn.y, tn.color));
      for (let i = tn.firstTool; i < tn.firstTool + tn.toolCount; i++) {
        const tl = toolNodes[i];
        parts.push(connector(tn.x, tn.y, tl.x, tl.y, tn.color));
      }
    });

    // Agent node
    parts.push(node(COL_X[0], agentY, info.agentName || 'Agent', '#2563eb', true));

    // Type-group nodes
    typeNodes.forEach(tn => parts.push(node(tn.x, tn.y, tn.label, tn.color, true)));

    // Tool leaf nodes
    toolNodes.forEach(tl => parts.push(node(tl.x, tl.y, tl.label, tl.color, false)));

    parts.push('</svg>');
    return parts.join('');
  }

  function appendHistory(entry) {
    entry._id = ++_historySeq;
    entry.timestamp = new Date().toLocaleTimeString();
    requestHistory.push(entry);
    renderHistory();
  }

  function renderHistory() {
    const log = document.getElementById('request-history-log');
    log.innerHTML = '';

    // Render newest first
    [...requestHistory].reverse().forEach((entry) => {
      const id = entry._id;
      const isOk = entry.status >= 200 && entry.status < 300;
      const entryEl = document.createElement('div');
      entryEl.className = 'history-entry';

      const statusClass = isOk ? 'ok' : 'err';
      const statusText = entry.status || '---';

      entryEl.innerHTML = `
        <div class="history-entry-header">
          <span class="history-status ${statusClass}">${statusText}</span>
          <span class="history-method">${_esc(entry.method)}</span>
          <span class="history-endpoint" title="${_esc(entry.endpoint)}">${_esc(entry.endpoint)}</span>
          <span class="history-time">${_esc(entry.timestamp)}</span>
          <span class="history-duration">${entry.durationMs}ms</span>
        </div>
        <div class="history-entry-body">
          <div class="history-json-block">
            <div class="history-json-label">
              <span>Request</span>
              <button class="btn-icon copy-btn" data-target="req-${id}">Copy</button>
            </div>
            <pre id="req-${id}">${_esc(JSON.stringify(entry.request, null, 2))}</pre>
          </div>
          <div class="history-json-block">
            <div class="history-json-label">
              <span>Response</span>
              <button class="btn-icon copy-btn" data-target="res-${id}">Copy</button>
            </div>
            <pre id="res-${id}">${_esc(JSON.stringify(entry.response, null, 2))}</pre>
          </div>
        </div>
      `;

      // Toggle body on header click
      entryEl.querySelector('.history-entry-header').addEventListener('click', () => {
        entryEl.querySelector('.history-entry-body').classList.toggle('open');
      });

      // Copy buttons
      entryEl.querySelectorAll('.copy-btn').forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const pre = document.getElementById(btn.dataset.target);
          if (pre) {
            navigator.clipboard.writeText(pre.textContent).then(() => {
              btn.textContent = 'Copied!';
              setTimeout(() => (btn.textContent = 'Copy'), 1500);
            });
          }
        });
      });

      log.appendChild(entryEl);
    });
  }

  function _esc(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function getAccessToken() { return accessToken; }
  function getAgentCard() { return agentCard; }

  return { init, getAccessToken, getAgentCard, appendHistory };
})();

window.InspectorPanel = InspectorPanel;
