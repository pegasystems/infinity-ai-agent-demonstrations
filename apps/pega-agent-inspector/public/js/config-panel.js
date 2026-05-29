'use strict';

const ConfigPanel = (() => {
  function init() {
    const protocolSelect = document.getElementById('protocol-select');
    protocolSelect.addEventListener('change', handleProtocolToggle);
    handleProtocolToggle(); // set initial state
  }

  function handleProtocolToggle() {
    const protocol = document.getElementById('protocol-select').value;
    document.getElementById('api-fields').classList.toggle('hidden', protocol !== 'api');
    document.getElementById('a2a-fields').classList.toggle('hidden', protocol !== 'a2a');
  }

  function getCredentials() {
    return {
      baseUrl: document.getElementById('base-url').value.trim(),
      accessTokenEndpoint: document.getElementById('token-endpoint').value.trim(),
      clientId: document.getElementById('client-id').value.trim(),
      clientSecret: document.getElementById('client-secret').value,
      protocol: document.getElementById('protocol-select').value,
      agentId: document.getElementById('agent-id').value.trim(),
      agentCardUrl: document.getElementById('agent-card-url').value.trim(),
    };
  }

  function validate() {
    const creds = getCredentials();
    const errors = [];

    if (!creds.accessTokenEndpoint) errors.push('Access Token Endpoint is required');
    else if (!_isValidUrl(creds.accessTokenEndpoint)) errors.push('Access Token Endpoint must be a valid URL');

    if (!creds.clientId) errors.push('Client ID is required');
    if (!creds.clientSecret) errors.push('Client Secret is required');

    if (creds.protocol === 'api') {
      if (!creds.baseUrl) errors.push('Base Application URL is required');
      else if (!_isValidUrl(creds.baseUrl)) errors.push('Base Application URL must be a valid URL');
      if (!creds.agentId) errors.push('Agent ID is required for API protocol');
    }

    if (creds.protocol === 'a2a') {
      if (!creds.agentCardUrl) errors.push('Agent Card URL is required for A2A protocol');
      else if (!_isValidUrl(creds.agentCardUrl)) errors.push('Agent Card URL must be a valid URL');
    }

    return { valid: errors.length === 0, errors };
  }

  function _isValidUrl(str) {
    try { new URL(str); return true; } catch { return false; }
  }

  return { init, getCredentials, validate };
})();

window.ConfigPanel = ConfigPanel;
