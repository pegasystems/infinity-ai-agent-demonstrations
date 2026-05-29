'use strict';

const path = require('path');
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const helmet = require('helmet');

const config = require('./config');
const authRouter = require('./routes/auth');
const agentRouter = require('./routes/agent');
const chatRouter = require('./routes/chat');

const app = express();

// Security headers — relax CSP for a local dev tool serving its own scripts
app.use(
  helmet({
    contentSecurityPolicy: false,
  })
);
// corsOrigin defaults to '*' — intentional for a localhost dev tool; restrict in production
app.use(cors({ origin: config.corsOrigin }));
app.use(morgan(config.nodeEnv === 'development' ? 'dev' : 'combined'));
app.use(express.json());

// API routes
app.use('/api/auth', authRouter);
app.use('/api/agent', agentRouter);
app.use('/api/chat', chatRouter);

// Serve frontend static files
app.use(express.static(path.join(__dirname, '../public')));

// Catch-all: send index.html for any non-API route
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public', 'index.html'));
});

// Global error handler
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, _next) => {
  console.error('[server error]', err);
  res.status(500).json({
    error: 'internal_error',
    message: err.message || 'An unexpected error occurred',
  });
});

app.listen(config.port, () => {
  console.log(`Pega Agent Inspector running at http://localhost:${config.port}`);
});

module.exports = app;
