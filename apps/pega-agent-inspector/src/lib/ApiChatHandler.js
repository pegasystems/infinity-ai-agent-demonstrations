'use strict';

const { PegaDXApi } = require('./PegaDXApi');

class ApiChatHandler {
  constructor(baseUrl, accessToken, agentId) {
    this.agentId = agentId;
    this.api = new PegaDXApi(baseUrl, accessToken);
  }

  // On the first message (no conversationId yet):
  //   1. POST /ai-agents/{agentID}/conversations  → get conversationId
  //   2. PATCH /ai-agents/{agentID}/conversations/{conversationId}  → send message, get reply
  //
  // On subsequent messages (conversationId already known):
  //   1. PATCH /ai-agents/{agentID}/conversations/{conversationId}  → send message, get reply
  async send(message, conversationId = null) {
    if (!conversationId) {
      const initiated = await this.api.initiateConversation(this.agentId);
      conversationId = initiated.conversationId;
      if (!conversationId) {
        throw new Error('Pega did not return a conversation ID from the initiate call');
      }
    }

    const result = await this.api.sendMessage(this.agentId, conversationId, message);
    return result;
  }
}

module.exports = { ApiChatHandler };
