import { logSystem } from './system-log.js';

export function feedbackMemory(model, updated) {
  const key = 'gifly-feedback-' + model;
  try {
    if (updated !== undefined) {
      localStorage.setItem(key, JSON.stringify(updated));
      return updated;
    }
    const items = JSON.parse(localStorage.getItem(key) || '[]');
    return Array.isArray(items) ? items.slice(-100) : [];
  } catch (error) { logSystem(error, { level: 'warn', context: 'feedback-memory', once: true }); return updated || []; }
}

function text(tag, value) {
  const element = document.createElement(tag);
  element.textContent = value;
  return element;
}

export function videoCards(results, base) {
  return results.map(item => {
    const card = document.createElement('article');
    if (item.url) {
      const video = document.createElement('video');
      video.src = base + item.url;
      video.controls = true;
      video.preload = 'metadata';
      video.addEventListener('error', () => {
        video.hidden = true; logSystem('Video preview could not be loaded.', { level: 'error', context: 'video-preview', once: true });
      }, { once: true });
      video.addEventListener('error', () => video.replaceWith(text('p', 'This video format could not be played.')), { once: true });
      card.append(video);
    } else logSystem('Video file unavailable.', { level: 'warn', context: 'video-preview', once: true });
    card.append(text('p', item.name), text('p', 'Similarity ' + item.score.toFixed(2)));
    return card;
  });
}
