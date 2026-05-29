'use strict';

require('dotenv').config();

module.exports = Object.freeze({
  port: process.env.PORT || 3000,
  nodeEnv: process.env.NODE_ENV || 'development',
  corsOrigin: process.env.CORS_ORIGIN || '*',
});
