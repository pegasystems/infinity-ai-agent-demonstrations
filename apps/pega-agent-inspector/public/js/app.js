'use strict';

// Global status bar helper — used by all panels
window.setStatusBar = function (message, type) {
  const bar = document.getElementById('status-bar');
  if (!bar) return;
  bar.textContent = message;
  bar.className = 'status-bar' + (type ? ' ' + type : '');

  // Auto-clear success messages after 4 seconds
  if (type === 'success') {
    clearTimeout(bar._clearTimer);
    bar._clearTimer = setTimeout(() => {
      bar.textContent = '';
      bar.className = 'status-bar';
    }, 4000);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.ConfigPanel.init();
  window.InspectorPanel.init();
  window.ChatPanel.init(window.ConfigPanel, window.InspectorPanel);

  // ── Theme toggle ────────────────────────────────────────────────
  const themeBtn = document.getElementById('theme-toggle');
  const applyTheme = (isLight) => {
    document.body.classList.toggle('light', isLight);
    themeBtn.textContent = isLight ? '🌙' : '☀️';
    themeBtn.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
  };

  // Restore saved preference, defaulting to dark
  const savedTheme = localStorage.getItem('theme');
  applyTheme(savedTheme === 'light');

  themeBtn.addEventListener('click', () => {
    const isLight = !document.body.classList.contains('light');
    applyTheme(isLight);
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
  });

  // Surface any unhandled promise rejections in the status bar
  window.addEventListener('unhandledrejection', (event) => {
    const msg = event.reason?.message || String(event.reason) || 'Unhandled error';
    window.setStatusBar('Unexpected error: ' + msg, 'error');
    console.error('[unhandledrejection]', event.reason);
  });
});
