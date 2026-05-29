'use strict';

const { randomUUID } = require('crypto');

class PegaDXApiError extends Error {
  constructor(message, statusCode, endpoint, upstreamBody) {
    super(message);
    this.name = 'PegaDXApiError';
    this.statusCode = statusCode;
    this.endpoint = endpoint;
    this.upstreamBody = upstreamBody;
  }
}

class PegaDXApi {
  constructor(baseUrl, accessToken) {
    // Strip trailing slash
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.accessToken = accessToken;
  }

  async _request(method, url, body) {
    const headers = {
      Authorization: `Bearer ${this.accessToken}`,
      Accept: 'application/json',
    };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }

    const opts = { method, headers };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
    }

    let res;
    try {
      console.log(`[PegaDXApi] ${method} ${url}`);
      res = await fetch(url, opts);
    } catch (err) {
      throw new PegaDXApiError(`Network error: ${err.message}`, 0, url, null);
    }

    let responseBody;
    try {
      responseBody = await res.json();
    } catch {
      responseBody = await res.text().catch(() => null);
    }

    if (!res.ok) {
      throw new PegaDXApiError(
        `Pega DX API returned ${res.status} for ${url}`,
        res.status,
        url,
        responseBody
      );
    }

    return responseBody;
  }

  async getAgentInfo(appliesTo, ruleName) {
    const dataViewParams = JSON.stringify({ AppliesTo: appliesTo, RuleName: ruleName });
    const url = `${this.baseUrl}/api/application/v2/data_views/D_GetAIAgentRule?dataViewParameters=${encodeURIComponent(dataViewParams)}`;
    const raw = await this._request('GET', url);

    return this._normalizeAgentInfo(raw);
  }

  _normalizeAgentInfo(raw) {
    const def = raw.pyGenAIDef || {};
    const modelCfg = (raw.pyGenAIConfig || {}).pyModelConfiguration || {};

    return {
      agentName: raw.pyLabel || raw.pyRuleName || '',
      description: raw.pyDescription || '',
      ruleName: raw.pyRuleName || '',
      className: raw.pyClassName || '',
      coachMode: raw.pyCoachMode || '',
      enableExternalAccess: raw.pyEnableExternalAccess || false,
      agentCardUrl: raw.pyAgentCardURL || '',
      model: {
        name: modelCfg.pyModelName || '',
        provider: modelCfg.pyProvider || '',
        modelId: modelCfg.pyModelId || '',
        description: modelCfg.pyDescription || '',
      },
      prompts: {
        system: def.pySystemPrompt || '',
        responseStyle: def.pyResponseStylePrompt || '',
        guardrails: def.pyGuardrailsPrompt || '',
        user: def.pyUserPrompt || '',
        initial: raw.pyInitialPromptInstructions || '',
      },
      examples: (def.pyExamples || []).map((e) => ({
        instruction: e.pyInstruction || '',
        example: e.pyExample || '',
      })),
      tools: {
        caseTypes: this._normalizeToolList(raw.pzCaseTypeTools),
        knowledge: this._normalizeToolList(raw.pzKnowledgeTools),
        agents: this._normalizeToolList(raw.pzAgentTools),
        externalAgents: this._normalizeToolList(raw.pzExternalAgents),
        mcpClients: this._normalizeToolList(raw.pzMCPClients),
      },
      raw,
    };
  }

  _normalizeToolList(tools) {
    if (!Array.isArray(tools)) return [];
    return tools.map((t) => ({
      name: t.pyPurpose || t.pyServiceName || '',
      category: t.pyCategory || '',
      askConfirmation: t.pyAskConfirmation || false,
      available: t.pyRuleAvailable === 'Yes',
      protocol: t.pyAgentProtocol || '',
    }));
  }

  // POST /ai-agents/{agentID}/conversations
  // Creates a new conversation and returns its ID along with an optional initial response.
  // interactionID is required by Pega when the caller is a non-Pega application.
  async initiateConversation(agentId) {
    const url = `${this.baseUrl}/api/application/v2/ai-agents/${_pathSegment(agentId)}/conversations`;
    const raw = await this._request('POST', url, { interactionID: randomUUID() });

    return {
      conversationId: raw.ID || null,
      initialResponse: raw.response || '',
      initialInstruction: raw.initialInstruction || '',
      messageId: raw.messageID || null,
      suggestedPrompts: raw.suggestedPrompts || [],
      raw,
    };
  }

  // PATCH /ai-agents/{agentID}/conversations/{conversationID}
  // Sends a user message and returns the agent's response.
  async sendMessage(agentId, conversationId, message) {
    const url = `${this.baseUrl}/api/application/v2/ai-agents/${_pathSegment(agentId)}/conversations/${_pathSegment(conversationId)}`;
    const raw = await this._request('PATCH', url, { Request: message });

    return {
      conversationId,
      reply: raw.response || '',
      messageId: raw.messageID || null,
      aiGuidedQuestions: raw.aiGuidedQuestions || [],
      raw,
    };
  }
}

// Encode a value for use in a URL path segment.
// Uses encodeURIComponent but restores characters that are valid in path segments
// (RFC 3986 §3.3 sub-delimiters) and that Pega's Tomcat router expects unencoded.
// Notably, encoding '@' as '%40' causes Pega to return 500 instead of routing correctly.
function _pathSegment(str) {
  return encodeURIComponent(str)
    .replace(/%40/g, '@')  // @
    .replace(/%21/g, '!'); // !
}

module.exports = { PegaDXApi, PegaDXApiError };
