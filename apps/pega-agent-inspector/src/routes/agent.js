'use strict';

const express = require('express');
const { PegaDXApi, PegaDXApiError } = require('../lib/PegaDXApi');
const { A2AHandler, A2AError } = require('../lib/A2AHandler');
const { assertSafeUrl } = require('../lib/assertSafeUrl');

const router = express.Router();

// Parse "AppliesTo!RuleName" agent ID format used by the API protocol.
function parseAgentId(agentId) {
  const bang = agentId.indexOf('!');
  if (bang < 1 || bang === agentId.length - 1) {
    throw Object.assign(new Error('Agent ID must be in the form AppliesTo!RuleName (e.g. MyOrg-MyApp-Work-Agent!MyAgentRule)'), { statusCode: 400 });
  }
  return { appliesTo: agentId.slice(0, bang).replace(/\/+$/, ''), ruleName: agentId.slice(bang + 1).replace(/\/+$/, '') };
}

// Parse an A2A agent card URL to extract AppliesTo, RuleName, and the Pega base URL.
// Expected format: https://<host>/<context>/.../.../ai-agents/<AppliesTo>!<RuleName>/.well-known/agent.json
function parseA2ACardUrl(agentCardUrl) {
  const urlObj = new URL(agentCardUrl);

  // The agent identity sits in the segment before /.well-known/agent.json
  const match = urlObj.pathname.match(/\/ai-agents\/([^/]+)\//);
  if (!match) {
    throw Object.assign(
      new Error('Cannot parse agent identity from Agent Card URL — expected path segment /ai-agents/<AppliesTo>!<RuleName>/'),
      { statusCode: 400 }
    );
  }

  const { appliesTo, ruleName } = parseAgentId(decodeURIComponent(match[1]));

  // Derive the Pega base URL: origin + first non-empty path segment (the context root, e.g. /prweb)
  const firstSegment = urlObj.pathname.split('/').find((s) => s.length > 0);
  const baseUrl = firstSegment ? `${urlObj.origin}/${firstSegment}` : urlObj.origin;

  return { appliesTo, ruleName, baseUrl };
}

router.post('/info', async (req, res, next) => {
  const { baseUrl, accessToken, protocol, agentId, agentCardUrl } = req.body || {};

  if (!accessToken) {
    return res.status(400).json({ error: 'missing_params', message: 'accessToken is required' });
  }
  if (!protocol || !['api', 'a2a'].includes(protocol)) {
    return res.status(400).json({ error: 'invalid_protocol', message: 'protocol must be "api" or "a2a"' });
  }

  try {
    if (protocol === 'api') {
      if (!baseUrl || !agentId) {
        return res.status(400).json({ error: 'missing_params', message: 'baseUrl and agentId are required for API protocol' });
      }
      try { assertSafeUrl(baseUrl); } catch (err) {
        return res.status(400).json({ error: 'invalid_url', message: err.message });
      }

      let parsed;
      try { parsed = parseAgentId(agentId); } catch (err) {
        return res.status(400).json({ error: 'invalid_agent_id', message: err.message });
      }

      const api = new PegaDXApi(baseUrl, accessToken);
      const result = await api.getAgentInfo(parsed.appliesTo, parsed.ruleName);
      return res.json(result);
    }

    // A2A: parse agent card URL for identity + base URL, then fetch card and agent info in parallel
    if (!agentCardUrl) {
      return res.status(400).json({ error: 'missing_params', message: 'agentCardUrl is required for A2A protocol' });
    }
    try { assertSafeUrl(agentCardUrl); } catch (err) {
      return res.status(400).json({ error: 'invalid_url', message: err.message });
    }

    let parsed;
    try { parsed = parseA2ACardUrl(agentCardUrl); } catch (err) {
      return res.status(400).json({ error: 'invalid_agent_card_url', message: err.message });
    }

    const handler = new A2AHandler(accessToken);
    // Use caller-supplied baseUrl if valid (handles non-standard Pega context paths);
    // fall back to deriving it from the agent card URL.
    let resolvedBaseUrl = parsed.baseUrl;
    if (baseUrl) {
      try { assertSafeUrl(baseUrl); resolvedBaseUrl = baseUrl; } catch { /* use derived */ }
    }
    const api = new PegaDXApi(resolvedBaseUrl, accessToken);

    // Fetch the agent card (for executeUrl) and the DX API agent info in parallel
    const [card, agentInfo] = await Promise.all([
      handler.fetchAgentCard(agentCardUrl),
      api.getAgentInfo(parsed.appliesTo, parsed.ruleName),
    ]);

    return res.json({
      ...agentInfo,
      executeUrl: card.executeUrl,
      capabilities: card.capabilities,
    });
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

module.exports = router;
