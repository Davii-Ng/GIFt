const recent = new Set();
const RECENT_LIMIT = 100;

// Log only the message, never request bodies, tokens, form values or account data.
export function logSystem(value, { level = 'info', context = 'app', once = false } = {}) {
  if (value?.name === 'AbortError') return;
  const message = typeof value === 'string' ? value.trim() : value?.message;
  if (!message) return;
  const severity = ['info', 'warn', 'error'].includes(level) ? level : 'info';
  const key = `${severity}:${context}:${message}`;
  if (once) {
    if (recent.has(key)) return;
    recent.add(key);
    if (recent.size > RECENT_LIMIT) recent.delete(recent.values().next().value);
  }
  console[severity](`[gifly:${context}] ${message}`);
}

export function consoleOnlyValidation(document) {
  // Keep native validation blocking invalid submissions, without browser bubbles.
  document.addEventListener('invalid', event => {
    event.preventDefault();
    logSystem(event.target.validationMessage, { level: 'warn', context: 'validation' });
  }, true);
}
