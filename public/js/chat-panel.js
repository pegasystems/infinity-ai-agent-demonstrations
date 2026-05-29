'use strict';

const ChatPanel = (() => {
  let conversationId = null;  // API protocol: Pega conversation ID
  let contextId = null;       // A2A protocol: context ID for multi-turn continuity
  let messageHistory = [];
  let _configPanel = null;
  let _inspectorPanel = null;

  function init(configPanel, inspectorPanel) {
    _configPanel = configPanel;
    _inspectorPanel = inspectorPanel;

    const input = document.getElementById('chat-input');
    document.getElementById('send-btn').addEventListener('click', handleSend);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });
    input.addEventListener('input', () => _autoResize(input));
    document.getElementById('new-conversation-btn').addEventListener('click', handleNewConversation);

    renderMessages(); // show empty state
  }

  async function handleSend() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    const accessToken = _inspectorPanel.getAccessToken();
    if (!accessToken) {
      _appendErrorBubble('No access token — please click "Get Agent Information" first to authenticate.');
      return;
    }

    const creds = _configPanel.getCredentials();

    // Append user message immediately
    messageHistory.push({ role: 'user', text: message, timestamp: _now() });
    renderMessages();
    input.value = '';
    _autoResize(input);

    _setSending(true);
    window.setStatusBar('Sending message…', '');

    try {
      let params;
      if (creds.protocol === 'api') {
        params = {
          protocol: 'api',
          accessToken,
          baseUrl: creds.baseUrl,
          agentId: creds.agentId,
          message,
          conversationId,
        };
      } else {
        const card = _inspectorPanel.getAgentCard();
        const executeUrl = card && card.executeUrl;
        if (!executeUrl) {
          _appendErrorBubble('No A2A execute URL found. Please click "Get Agent Information" first.');
          _setSending(false);
          return;
        }
        params = {
          protocol: 'a2a',
          accessToken,
          executeUrl,
          message,
          contextId,
        };
      }

      const reqStart = Date.now();
      // Redact the token from the history entry
      const loggedParams = { ...params, accessToken: '***' };

      let result;
      try {
        result = await window.ApiClient.sendMessage(params);
        _inspectorPanel.appendHistory({
          method: 'POST',
          endpoint: '/api/chat/send',
          status: 200,
          durationMs: Date.now() - reqStart,
          request: loggedParams,
          response: result,
        });
      } catch (err) {
        _inspectorPanel.appendHistory({
          method: 'POST',
          endpoint: '/api/chat/send',
          status: err.status || 0,
          durationMs: Date.now() - reqStart,
          request: loggedParams,
          response: { error: err.error, message: err.message },
        });
        throw err;
      }

      if (creds.protocol === 'api') {
        conversationId = result.conversationId || conversationId;
      } else {
        contextId = result.contextId || contextId;
      }

      const reply = result.reply || '(no response text)';
      messageHistory.push({ role: 'agent', text: reply, timestamp: _now() });
      renderMessages();
      window.setStatusBar('', '');
    } catch (err) {
      const detail = err.upstreamBody ? '\n\nPega response: ' + JSON.stringify(err.upstreamBody, null, 2) : '';
      _appendErrorBubble((err.message || 'An error occurred sending the message.') + detail);
      window.setStatusBar('Send failed: ' + err.message, 'error');
    }

    _setSending(false);
  }

  async function handleNewConversation() {
    conversationId = null;
    contextId = null;
    messageHistory = [];
    renderMessages();

    const creds = _configPanel.getCredentials();
    try {
      await window.ApiClient.newConversation(creds.protocol);
    } catch {
      // Non-critical; state already reset in frontend
    }

    window.setStatusBar('New conversation started.', 'success');
  }

  function renderMessages() {
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';

    if (messageHistory.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'chat-empty';
      empty.textContent = 'Configure the connection and fetch agent info, then start chatting.';
      container.appendChild(empty);
      return;
    }

    messageHistory.forEach((msg) => {
      const wrapper = document.createElement('div');
      wrapper.className = `chat-message ${msg.role}`;

      const bubble = document.createElement('div');
      bubble.className = 'message-bubble';
      if (msg.role === 'agent') {
        const raw = window.marked ? window.marked.parse(msg.text) : msg.text;
        bubble.innerHTML = window.DOMPurify ? window.DOMPurify.sanitize(raw) : raw;
      } else {
        bubble.textContent = msg.text;
      }

      const meta = document.createElement('div');
      meta.className = 'message-meta';
      meta.textContent = (msg.role === 'user' ? 'You' : 'Agent') + ' · ' + msg.timestamp;

      wrapper.appendChild(bubble);
      wrapper.appendChild(meta);
      container.appendChild(wrapper);
    });

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
  }

  function _appendErrorBubble(text) {
    messageHistory.push({ role: 'error', text, timestamp: _now() });
    renderMessages();
  }

  function _setSending(isSending) {
    const btn = document.getElementById('send-btn');
    const label = btn.querySelector('.send-label');
    const spinner = btn.querySelector('.spinner');
    btn.disabled = isSending;
    label.classList.toggle('hidden', isSending);
    spinner.classList.toggle('hidden', !isSending);
  }

  function _autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }

  function _now() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  return { init };
})();

window.ChatPanel = ChatPanel;
