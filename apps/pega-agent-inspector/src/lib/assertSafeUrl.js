'use strict';

// Asserts that a URL is syntactically valid and uses HTTPS.
// Throws on failure so routes can return a 400 or silently discard unsafe overrides.
function assertSafeUrl(raw) {
  let u;
  try {
    u = new URL(raw);
  } catch {
    throw new Error(`"${raw}" is not a valid URL`);
  }
  if (u.protocol !== 'https:') {
    throw new Error(`URL must use HTTPS (received "${u.protocol.replace(/:$/, '')}")`);
  }
}

module.exports = { assertSafeUrl };
