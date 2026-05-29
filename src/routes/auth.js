'use strict';

const express = require('express');
const { PegaAuth, PegaAuthError } = require('../lib/PegaAuth');
const { assertSafeUrl } = require('../lib/assertSafeUrl');

const router = express.Router();

router.post('/token', async (req, res, next) => {
  const { accessTokenEndpoint, clientId, clientSecret } = req.body || {};

  if (!accessTokenEndpoint || !clientId || !clientSecret) {
    return res.status(400).json({
      error: 'missing_params',
      message: 'accessTokenEndpoint, clientId, and clientSecret are required',
    });
  }

  try {
    assertSafeUrl(accessTokenEndpoint);
  } catch (err) {
    return res.status(400).json({
      error: 'invalid_url',
      message: err.message,
    });
  }

  try {
    const auth = new PegaAuth(accessTokenEndpoint, clientId, clientSecret);
    const token = await auth.getToken();
    return res.json(token);
  } catch (err) {
    if (err instanceof PegaAuthError) {
      return res.status(502).json({
        error: 'token_request_failed',
        message: err.message,
        statusCode: err.statusCode,
        upstreamBody: err.upstreamBody,
      });
    }
    next(err);
  }
});

module.exports = router;
