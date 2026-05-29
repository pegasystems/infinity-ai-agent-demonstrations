'use strict';

const express = require('express');
const { ApiChatHandler } = require('../lib/ApiChatHandler');
const { A2AHandler, A2AError } = require('../lib/A2AHandler');
const { PegaDXApiError } = require('../lib/PegaDXApi');
const { assertSafeUrl } = require('../lib/assertSafeUrl');

const router = express.Router();

router.post('/send', async (req, res, next) => {
  const { protocol, accessToken, message } = req.body || {};

  if (!protocol || !['api', 'a2a'].includes(protocol)) {
    return res.status(400).json({ error: 'invalid_protocol', message: 'protocol must be "api" or "a2a"' });
  }
  if (!accessToken) {
    return res.status(400).json({ error: 'missing_params', message: 'accessToken is required' });
  }
  if (!message || typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'missing_params', message: 'message is required' });
  }

  try {
    if (protocol === 'api') {
      const { baseUrl, conversationId } = req.body;
      const agentId = (req.body.agentId || '').replace(/\/+$/, '');
      if (!baseUrl || !agentId) {
        return res.status(400).json({ error: 'missing_params', message: 'baseUrl and agentId are required for API protocol' });
      }
      try { assertSafeUrl(baseUrl); } catch (err) {
        return res.status(400).json({ error: 'invalid_url', message: err.message });
      }
      const handler = new ApiChatHandler(baseUrl, accessToken, agentId);
      const result = await handler.send(message, conversationId || null);
      return res.json({ protocol: 'api', ...result });
    }

    // A2A protocol
    const { executeUrl, contextId } = req.body;
    if (!executeUrl) {
      return res.status(400).json({ error: 'missing_params', message: 'executeUrl is required for A2A protocol' });
    }
    try { assertSafeUrl(executeUrl); } catch (err) {
      return res.status(400).json({ error: 'invalid_url', message: err.message });
    }
    const handler = new A2AHandler(accessToken);
    const result = await handler.sendMessage(executeUrl, message, contextId || null);
    return res.json({ protocol: 'a2a', ...result });
  } catch (err) {
    if (err instanceof PegaDXApiError || err instanceof A2AError) {
      return res.status(502).json({
        error: 'upstream_error',
        message: err.message,
        statusCode: err.statusCode,
        upstreamBody: err.upstreamBody,
      });
    }
    next(err);
  }
});

// Conversation reset signal. No server-side state to clear today, but the endpoint
// exists as a hook for future session management (e.g. server-side conversation caching).
router.post('/new', (_req, res) => {
  res.json({ reset: true });
});

module.exports = router;
