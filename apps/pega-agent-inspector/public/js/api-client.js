'use strict';

const ApiClient = (() => {
  async function _post(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    let data;
    try {
      data = await res.json();
    } catch {
      data = { error: 'parse_error', message: `Server returned non-JSON response (${res.status})` };
    }
    if (!res.ok) {
      const err = new Error(data.message || `Request failed with status ${res.status}`);
      err.status = res.status;
      err.error = data.error || 'unknown_error';
      err.upstreamBody = data.upstreamBody;
      throw err;
    }
    return data;
  }

  return {
    getToken(credentials) {
      return _post('/api/auth/token', credentials);
    },
    getAgentInfo(params) {
      return _post('/api/agent/info', params);
    },
    sendMessage(params) {
      return _post('/api/chat/send', params);
    },
    newConversation(protocol) {
      return _post('/api/chat/new', { protocol });
    },
  };
})();

window.ApiClient = ApiClient;
