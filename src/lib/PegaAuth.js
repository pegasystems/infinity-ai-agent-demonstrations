'use strict';

class PegaAuthError extends Error {
  constructor(message, statusCode, upstreamBody) {
    super(message);
    this.name = 'PegaAuthError';
    this.statusCode = statusCode;
    this.upstreamBody = upstreamBody;
  }
}

class PegaAuth {
  constructor(accessTokenEndpoint, clientId, clientSecret) {
    this.accessTokenEndpoint = accessTokenEndpoint;
    this.clientId = clientId;
    this.clientSecret = clientSecret;
  }

  async getToken() {
    const body = new URLSearchParams({
      grant_type: 'client_credentials',
      client_id: this.clientId,
      client_secret: this.clientSecret,
    });

    let res;
    try {
      res = await fetch(this.accessTokenEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
      });
    } catch (err) {
      throw new PegaAuthError(
        `Network error reaching token endpoint: ${err.message}`,
        0,
        null
      );
    }

    let responseBody;
    try {
      responseBody = await res.json();
    } catch {
      responseBody = await res.text().catch(() => null);
    }

    if (!res.ok) {
      throw new PegaAuthError(
        `Token endpoint returned ${res.status}`,
        res.status,
        responseBody
      );
    }

    return responseBody;
  }
}

module.exports = { PegaAuth, PegaAuthError };
