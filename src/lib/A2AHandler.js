'use strict';

const { randomUUID } = require('crypto');

class A2AError extends Error {
  constructor(message, statusCode, upstreamBody) {
    super(message);
    this.name = 'A2AError';
    this.statusCode = statusCode;
    this.upstreamBody = upstreamBody;
  }
}

class A2AHandler {
  constructor(accessToken) {
    this.accessToken = accessToken;
  }

  // GET: no Content-Type (no body)
  _getHeaders() {
    return {
      Authorization: `Bearer ${this.accessToken}`,
      Accept: 'application/json',
    };
  }

  // POST: include Content-Type
  _postHeaders() {
    return {
      Authorization: `Bearer ${this.accessToken}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
  }

  async fetchAgentCard(agentCardUrl) {
    let res;
    try {
      res = await fetch(agentCardUrl, {
        method: 'GET',
        headers: this._getHeaders(),
      });
    } catch (err) {
      throw new A2AError(`Network error fetching agent card: ${err.message}`, 0, null);
    }

    let raw;
    try {
      raw = await res.json();
    } catch {
      const text = await res.text().catch(() => '');
      throw new A2AError(`Agent card response is not valid JSON: ${text}`, res.status, text);
    }

    if (!res.ok) {
      throw new A2AError(`Agent card endpoint returned ${res.status}`, res.status, raw);
    }

    // The execute endpoint is the top-level "url" field in the A2A agent card spec.
    const executeUrl = raw.url || null;

    return {
      name: raw.name || '',
      description: raw.description || '',
      executeUrl,
      capabilities: raw.capabilities || {},
      skills: raw.skills || [],
      raw,
    };
  }

  // Send a message to the agent via JSON-RPC 2.0 (A2A 0.3 "message/send").
  // contextId is null on the first turn; subsequent turns pass back the contextId
  // returned by the previous response to maintain conversation continuity.
  async sendMessage(executeUrl, message, contextId = null) {
    const jsonRpcId = randomUUID();

    // A2A 0.3 message shape: parts use "kind" (not "type"), message requires "kind" and "metadata"
    const msgPayload = {
      role: 'user',
      kind: 'message',
      messageId: randomUUID(),
      metadata: {},
      parts: [{ kind: 'text', text: message }],
    };
    if (contextId) {
      msgPayload.contextId = contextId;
    }

    const payload = {
      jsonrpc: '2.0',
      id: jsonRpcId,
      method: 'message/send',
      params: {
        message: msgPayload,
      },
    };

    let res;
    try {
      res = await fetch(executeUrl, {
        method: 'POST',
        headers: this._postHeaders(),
        body: JSON.stringify(payload),
      });
    } catch (err) {
      throw new A2AError(`Network error calling A2A execute endpoint: ${err.message}`, 0, null);
    }

    let raw;
    try {
      raw = await res.json();
    } catch {
      const text = await res.text().catch(() => '');
      throw new A2AError(`A2A response is not valid JSON: ${text}`, res.status, text);
    }

    if (!res.ok) {
      throw new A2AError(`A2A execute endpoint returned ${res.status}`, res.status, raw);
    }

    if (raw.error) {
      throw new A2AError(
        `A2A JSON-RPC error ${raw.error.code}: ${raw.error.message}`,
        res.status,
        raw
      );
    }

    // A2A 0.3 response: result IS the agent message object directly.
    // { role, kind, parts: [{kind:"text", text}], contextId }
    const result = raw.result || {};
    const returnedContextId = result.contextId || contextId;
    const reply = _extractReply(result);

    return { contextId: returnedContextId, reply, raw };
  }
}

// Extract the agent's reply text from the A2A 0.3 result object.
// result is the agent message directly: { role, kind, parts:[{kind:"text", text}], contextId }
function _extractReply(result) {
  // Primary: parts array with kind:"text" (A2A 0.3)
  if (Array.isArray(result.parts)) {
    const part = result.parts.find((p) => p.kind === 'text' || p.type === 'text');
    if (part?.text) return part.text;
  }

  // Fallback: nested artifacts from older spec shapes
  if (Array.isArray(result.artifacts) && result.artifacts.length > 0) {
    for (const artifact of result.artifacts) {
      const part = (artifact.parts || []).find((p) => p.kind === 'text' || p.type === 'text');
      if (part?.text) return part.text;
    }
  }

  return '';
}

module.exports = { A2AHandler, A2AError };
