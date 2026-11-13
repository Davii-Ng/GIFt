import { observeAccount, authenticate, logout, saveFavorite, accountToken, syncContribution } from './firebase-account.js?v=contribution-studio';
import { initContributions } from './contributions.js?v=console-only';
import { logSystem, consoleOnlyValidation } from './system-log.js';
import { fetchGIFBlob, saveGIFBlob } from './gif-actions.js?v=gif-actions-relevance';
import { apiBaseFor, requestJSON } from './api-client.js';
import { feedbackMemory, videoCards } from './tools.js?v=console-only';

const $ = selector => document.querySelector(selector);
const API_BASE = apiBaseFor(window.location);
const PAGE_SIZE = 24;
const HOME_SEED = crypto.getRandomValues(new Uint32Array(1))[0];
const HOME_VOTE_LIMIT = 30;
const state = {
  catalog: [], results: [], known: new Map(), total: 0, hasMore: false,
  mode: 'gif', feedback: null, searchPage: null,
  homeVotes: new Map(feedbackMemory('home').filter(entry => Array.isArray(entry) && entry[1]?.choice)),
  homeResults: null,
  query: '', category: 'all', view: 'browse', saved: new Set(), source: 'GIF',
  paused: matchMedia('(prefers-reduced-motion: reduce)').matches,
  user: null, accountLoading: true, accountRevision: 0, authMode: 'login', pendingAuth: null, pendingSavedView: false, authBusy: false,
  request: 0, controller: null, loading: false,
};
const categoryNames = { reactions: 'Reactions', animals: 'Animals', celebration: 'Celebration', mood: 'Mood' };
let contributionUI;
let previewGIF;
consoleOnlyValidation(document);

async function downloadGIF(gif, button) {
  const revision = state.accountRevision;
  const label = button.textContent; button.disabled = true; button.textContent = '…';
  try {
    const token = gif.id?.startsWith('contrib_') && state.user ? await accountToken(state.user.id) : undefined;
    if (revision !== state.accountRevision) return;
    const blob = await fetchGIFBlob(gif, { base: API_BASE, token });
    if (revision !== state.accountRevision) return;
    saveGIFBlob(blob, gif); toast('Your GIF is downloading.');
  } catch (error) { if (revision === state.accountRevision) logSystem(error, { level: 'error', context: 'download' }); }
  finally { button.disabled = false; button.textContent = label; }
}

function toast(message) {
  logSystem(message);
  $('#toast').textContent = ''; $('#toast').hidden = true;
}
function notice(message) {
  logSystem(message, { level: 'error' });
  $('#notice').textContent = ''; $('#notice').hidden = true;
}
function remember(gifs) {
  for (const gif of gifs) {
    if (gif.url?.startsWith('/api/contributions/')) gif.url = API_BASE + gif.url;
    state.known.set(gif.id, gif);
  }
}
function updateSavedCount() { $('#saved-count').textContent = state.saved.size; }
function applyAccount(data) {
  if (state.user?.id !== data.user?.id) {
    for (const id of state.saved) state.known.delete(id);
    state.accountRevision++;
    if (state.view === "saved") { state.controller?.abort(); state.request++; finishRequest(state.request); }
  }
  state.accountLoading = Boolean(data.loading);
  state.user = data.user; state.saved = new Set(data.saved || []);
  contributionUI?.setAccount(state.user);
  $('#account-button').disabled = state.accountLoading;
  $('#account-button').textContent = state.accountLoading ? 'Loading account…' : 'Log in / Sign up';
  $('#saved-nav').disabled = state.accountLoading;
  $('#logout-button').disabled = state.accountLoading;
  $('#saved-count').textContent = state.saved.size;
  $('#account-button').hidden = Boolean(state.user);
  $('#account-name').hidden = !state.user;
  $('#account-name').textContent = state.user?.username || '';
  $('#logout-button').hidden = !state.user;
  updateSavedCount();
}
let authSubmitted = false;
const authFields = ['username', 'password', 'confirm'];
function showAuthErrors(errors = {}, message = '') {
  for (const field of authFields) {
    const input = $(`#auth-${field}`);
    const error = $(`#${field}-error`);
    input.setAttribute('aria-invalid', Boolean(errors[field]));
    error.textContent = ''; error.hidden = true;
  }
  logSystem(message || Object.values(errors).join(' '), { level: 'error', context: 'account' });
  $('#auth-error').textContent = ''; $('#auth-error').hidden = true;
}
function validateAuthForm(focus = false) {
  const errors = {};
  const username = $('#auth-username').value.trim();
  const password = $('#auth-password').value;
  const register = state.authMode === 'register';
  if (!username) errors.username = 'Enter your username.';
  else if (!/^[A-Za-z0-9_]{3,24}$/.test(username)) errors.username = 'Use 3–24 letters, numbers or underscores.';
  if (!password) errors.password = 'Enter your password.';
  else if (password.length < 8 || password.length > 128) errors.password = 'Your password must contain 8–128 characters.';
  if (register) {
    if (!$('#auth-confirm').value) errors.confirm = 'Confirm your password.';
    else if ($('#auth-confirm').value !== password) errors.confirm = 'Passwords do not match.';
  }
  showAuthErrors(errors);
  if (focus && Object.keys(errors).length) $(`#auth-${Object.keys(errors)[0]}`).focus();
  return !Object.keys(errors).length;
}
function setAuthMode(mode) {
  state.authMode = mode;
  const register = mode === 'register';
  $('#auth-title').textContent = register ? 'Make it yours.' : 'Welcome back.';
  $('#auth-description').textContent = register ? 'Create an account to save the GIFs you love.' : 'Log in to keep your favorite GIFs together.';
  $('#auth-submit').textContent = register ? 'Sign up' : 'Log in';
  $('#auth-password').autocomplete = register ? 'new-password' : 'current-password';
  $('#auth-confirm-group').hidden = !register; $('#auth-confirm').required = register;
  $('#auth-confirm').disabled = !register;
  $('#username-hint').hidden = !register; $('#password-hint').hidden = !register;
  $('#auth-success').hidden = true; authSubmitted = false; showAuthErrors();
  for (const action of ['login', 'register']) {
    $(`#${action}-tab`).classList.toggle('active', mode === action);
    $(`#${action}-tab`).setAttribute('aria-pressed', mode === action);
  }
}
function openAuth(mode = 'login', pending = null) {
  state.pendingAuth = pending;
  $('#auth-form').reset(); setAuthMode(mode);
  if (!$('#auth-dialog').open) $('#auth-dialog').showModal();
  $('#auth-username').focus();
}
$('#account-button').addEventListener('click', () => openAuth());
$('#login-tab').addEventListener('click', () => setAuthMode('login'));
$('#register-tab').addEventListener('click', () => setAuthMode('register'));
$('#auth-dialog').addEventListener('close', () => {
  state.pendingAuth = null; $('#auth-form').reset();
});
for (const field of authFields) {
  $(`#auth-${field}`).addEventListener('input', () => {
    if (authSubmitted) validateAuthForm();
  });
}
$('#auth-form').addEventListener('submit', async event => {
  event.preventDefault(); if (state.authBusy) return;
  authSubmitted = true;
  $('#auth-success').hidden = true;
  if (!validateAuthForm(true)) return;
  const password = $('#auth-password').value;
  const mode = state.authMode;
  const pending = state.pendingAuth;
  state.authBusy = true;
  for (const id of ['auth-submit', 'login-tab', 'register-tab']) $(`#${id}`).disabled = true;
  try {
    const result = await authenticate(mode, $('#auth-username').value, password);
    state.pendingSavedView = pending?.view === 'saved';
    applyAccount({ user: result.user, saved: state.user?.id === result.user.id ? [...state.saved] : [], loading: state.accountLoading });
    $('#auth-dialog').close(); render();
    toast(result.warning || `Welcome, ${result.username}!`);
    if (pending?.gif) {
      const button = document.createElement('button');
      if (!state.saved.has(pending.gif.id)) await toggleSaved(pending.gif, button);
      render();
    }
    if (state.pendingSavedView && !state.accountLoading) {
      state.pendingSavedView = false;
      await showSaved();
    }
  } catch (error) {
    const fields = error.status === 409 ? { username: error.message }
      : error.status === 401 ? { username: 'Check your username.', password: 'Check your password.' } : {};
    showAuthErrors(fields, error.message);
    if (Object.keys(fields).length) $(`#auth-${Object.keys(fields)[0]}`).focus();
  } finally {
    state.authBusy = false;
    for (const id of ['auth-submit', 'login-tab', 'register-tab']) $(`#${id}`).disabled = false;
  }
});
$('#logout-button').addEventListener('click', async () => {
  $('#logout-button').disabled = true;
  try {
    await logout();
    toast('You have logged out.');
  } catch (error) { logSystem(error, { level: 'error', context: 'logout' }); }
  finally { $('#logout-button').disabled = false; }
});
function syncSave(button, gif) {
  const saved = state.saved.has(gif.id);
  button.classList.toggle('saved', saved); button.textContent = saved ? '♥' : '♡';
  button.setAttribute('aria-label', `${saved ? 'Unsave' : 'Save'} GIF ${gif.title}`);
  button.setAttribute('aria-pressed', saved);
}
async function toggleSaved(gif, button) {
  if (state.accountLoading && !state.authBusy) { toast('Đang tải tài khoản. Vui lòng chờ.'); return; }
  if (!state.user) { openAuth('login', { gif }); return; }
  const revision = state.accountRevision;
  const saved = !state.saved.has(gif.id);
  button.disabled = true;
  try {
    await saveFavorite(state.user.id, gif.id, saved);
    if (revision !== state.accountRevision) return;
    if (saved) state.saved.add(gif.id); else state.saved.delete(gif.id);
    remember([gif]); syncSave(button, gif); updateSavedCount();
    if (state.view === 'saved') render();
  } catch (error) {
    if (revision !== state.accountRevision) return;
    if (error.status === 401) { applyAccount({ user: null, saved: [] }); openAuth('login', { gif }); }
    else logSystem(error, { level: 'error', context: 'favorites' });
  } finally { button.disabled = false; }
}
function pauseImage(image, frame) {
  if (!image.complete || !image.naturalWidth || !image.isConnected || !state.paused) return;
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  canvas.setAttribute('aria-hidden', 'true');
  const context = canvas.getContext('2d'); if (!context) return;
  context.drawImage(image, 0, 0); frame.append(canvas); image.hidden = true;
}
function makeCard(gif, index) {
  const article = document.createElement('article'); article.className = 'gif-card';
  article.style.setProperty('--ratio', gif.ratio || 1.33);
  const open = document.createElement('button'); open.type = 'button'; open.className = 'gif-open';
  open.setAttribute('aria-label', `View GIF: ${gif.title}`);
  const image = document.createElement('img'); image.src = gif.url; image.alt = gif.description;
  image.loading = index < 8 ? 'eager' : 'lazy'; image.decoding = 'async'; image.referrerPolicy = 'no-referrer';
  image.addEventListener('load', () => pauseImage(image, open), { once: true });
  image.addEventListener('error', () => {
    image.hidden = true; article.classList.add('unavailable');
    logSystem('GIF preview could not be loaded.', { level: 'error', context: `gif-preview:${gif.id}`, once: true });
  }, { once: true });
  open.append(image); open.addEventListener('click', () => openGif(gif));
  const info = document.createElement('div'); info.className = 'card-info';
  const title = document.createElement('h3'); title.textContent = gif.title;
  const tag = document.createElement('p');
  tag.textContent = [categoryNames[gif.category] || gif.source || 'GIF',
    state.view === 'browse' && state.query && typeof gif.score === 'number' ? `Similarity ${gif.score.toFixed(2)}` : ''].filter(Boolean).join(' · ');
  info.append(title, tag);
  const favorite = gif.favorite_id === undefined ? gif : state.known.get(gif.favorite_id)
    || (gif.favorite_id ? { ...gif, id: gif.favorite_id } : null);
  const arrow = document.createElement('span'); arrow.className = 'card-arrow'; arrow.textContent = '↗';
  arrow.setAttribute('aria-hidden', 'true'); article.append(open, info, arrow);
  if (favorite) {
    const save = document.createElement('button'); save.type = 'button'; save.className = 'save-button';
    syncSave(save, favorite); save.addEventListener('click', () => toggleSaved(favorite, save));
    article.append(save);
  }
  if (state.view !== 'browse' || (state.query && !state.feedback)) return article;
  const wrapper = document.createElement('div');
  wrapper.className = 'feedback-card';
  const actions = document.createElement('div'); actions.className = 'tools-actions';
  for (const [choice, label] of [['liked', '👍 Like'], ['disliked', '👎 Dislike']]) {
    const button = document.createElement('button'); button.type = 'button'; button.textContent = label;
    const vote = state.query ? state.feedback.votes.get(gif.id) : state.homeVotes.get(gif.id)?.choice;
    button.setAttribute('aria-pressed', vote === choice);
    button.addEventListener('click', () => {
      if (!state.loading) {
        if (state.query) changeVote(gif.id, choice);
        else changeHomeVote(gif, choice);
      }
    });
    actions.append(button);
  }
  wrapper.append(article, actions);
  return wrapper;
}
function beginRequest() {
  state.controller?.abort(); state.controller = new AbortController();
  state.loading = true; $('#search-button').disabled = true; $('#load-more').disabled = true;
  notice(''); return ++state.request;
}
function finishRequest(request) {
  if (request !== state.request) return;
  state.loading = false; $('#search-button').disabled = false; $('#load-more').disabled = false;
}
async function getJSON(url, options) {
  return requestJSON(url, { base: API_BASE, signal: state.controller.signal, ...options });
}
function skeleton() {
  $('#empty-state').hidden = true; $('#collection-end').hidden = true; $('#load-more').hidden = true;
  $('#search-more-status').hidden = true;
  $('#gif-grid').setAttribute('aria-busy', 'true');
  $('#gif-grid').replaceChildren(...Array.from({ length: 12 }, () => {
    const element = document.createElement('div'); element.className = 'gif-card skeleton';
    element.setAttribute('aria-hidden', 'true'); return element;
  }));
}
function renderLoadMore() {
  const searching = state.view === 'browse' && state.query && state.feedback && state.searchPage;
  const browseMore = state.view === 'browse' && !state.query && !state.homeResults && state.hasMore;
  $('#load-more').hidden = !(searching ? !state.searchPage.ended : browseMore);
  $('#search-more-status').textContent = searching ? state.searchPage.message || '' : '';
  $('#search-more-status').hidden = !$('#search-more-status').textContent;
}
function render() {
  if (contributionUI && !contributionUI.isExplore()) return;
  if (state.mode !== 'gif') return;
  const feedback = state.view === 'browse' ? state.feedback : null;
  const home = state.view === 'browse' && !state.query;
  const homeVotes = home ? [...state.homeVotes.values()] : [];
  const homeHidden = homeVotes.filter(vote => vote.choice === 'disliked').length;
  $('#feedback-actions').hidden = !feedback && !homeVotes.length;
  if (feedback) {
    const liked = [...feedback.votes.values()].filter(vote => vote === 'liked').length;
    $('#feedback-summary').textContent = `${liked} liked · ${feedback.votes.size - liked} disliked · ${feedback.hidden} hidden`;
    $('#feedback-reset').disabled = !feedback.votes.size;
    $('#feedback-unhide').disabled = !feedback.hidden;
  } else if (homeVotes.length) {
    $('#feedback-summary').textContent = `${homeVotes.length - homeHidden} liked · ${homeHidden} hidden from home`;
    $('#feedback-reset').disabled = false;
    $('#feedback-unhide').disabled = !homeHidden;
  }
  const source = state.view === 'saved' ? [...state.saved].map(id => state.known.get(id)).filter(Boolean)
    : state.query ? state.results : state.homeResults ?? state.catalog;
  const items = source.filter(gif => (state.category === 'all' || gif.category === state.category)
    && (!home || state.homeVotes.get(gif.id)?.choice !== 'disliked'));
  $('#gif-grid').classList.toggle('ranked', Boolean(state.view === 'browse' && (state.query || state.homeResults)));
  $('#gif-grid').replaceChildren(...items.map(makeCard)); $('#gif-grid').setAttribute('aria-busy', 'false');
  $('#empty-state').hidden = items.length > 0; $('#collection-end').hidden = items.length === 0;
  renderLoadMore();
  $('#empty-title').textContent = state.view === 'saved' ? 'Save a little feeling' : 'No matching GIFs found';
  $('#empty-description').textContent = state.view === 'saved'
    ? 'Tap the heart on a GIF to add it to your collection.' : home && homeHidden ? 'Load more GIFs or show the hidden GIFs again.'
    : 'Try another search or a different category.';
  $('#results-heading').textContent = state.view === 'saved' ? 'Your favorite GIFs'
    : state.query ? `Results for “${state.query}”` : state.homeResults ? 'More like your liked GIFs' : 'A little inspiration ✦';
  $('#section-eyebrow').textContent = state.view === 'saved' ? 'FEELINGS WORTH KEEPING'
    : state.query ? 'SEARCH BY MEANING, FIND THE FEELING' : 'A LITTLE SCROLL, A LITTLE JOY';
  $('#result-count').textContent = state.view === 'saved' ? `${items.length} saved GIFs`
    : state.query ? `${items.length} closest GIFs` : state.homeResults ? `${items.length} recommended GIFs`
    : `${items.length} / ${state.total.toLocaleString('en-US')} GIF`;
  for (const view of ['browse', 'saved']) {
    $(`#${view}-nav`).classList.toggle('active', state.view === view);
    $(`#${view}-nav`).setAttribute('aria-pressed', state.view === view);
  }
}
function syncFilters() {
  document.querySelectorAll('[data-category]').forEach(button => {
    const selected = button.dataset.category === state.category;
    button.classList.toggle('active', selected); button.setAttribute('aria-pressed', selected);
  });
}
async function loadCatalog(append = false) {
  contributionUI?.showExplore();
  state.view = 'browse'; state.query = ''; state.feedback = null; state.searchPage = null; $('#search-input').value = '';
  const request = beginRequest(); const offset = append ? state.catalog.length : 0;
  if (!append) { state.catalog = []; state.homeResults = null; skeleton(); }
  try {
    const params = new URLSearchParams({ limit: PAGE_SIZE, offset, category: state.category, seed: HOME_SEED });
    const data = await getJSON(`/api/catalog?${params}`); if (request !== state.request) return;
    remember(data.gifs); state.catalog = append ? [...state.catalog, ...data.gifs] : data.gifs;
    state.total = data.total; state.hasMore = data.has_more; state.source = data.source;
    $('#category-filters').hidden = !data.categories?.length;
    document.querySelectorAll('[data-category]').forEach(button => {
      button.hidden = button.dataset.category !== 'all' && !data.categories.includes(button.dataset.category);
    });
    $('#dataset-label').textContent = `GIFs from original sources · ${state.source === 'Sample' ? 'Sample collection' : state.source}`;
    render();
  } catch (error) {
    if (error.name === 'AbortError' || request !== state.request) return;
    render(); notice(error.message);
    if (!state.catalog.length) $('#empty-state').hidden = true;
  } finally { finishRequest(request); }
}
function reset() {
  state.category = 'all'; syncFilters();
  syncSearchMode('gif');
  return loadCatalog();
}

async function search(query, update = {}) {
  contributionUI?.showExplore();
  query = query.trim(); if (!query) return reset();
  const previous = state.query === query ? state.feedback : null;
  const append = Boolean(update.append && previous && state.searchPage);
  if (append && state.loading) return;
  if (append && !state.searchPage.hasMore) {
    state.searchPage.ended = true;
    state.searchPage.message = 'No more matching GIFs. All results with similarity above 0 are shown.';
    renderLoadMore(); return;
  }
  const votes = update.votes ?? previous?.votes ?? new Map();
  const memory = update.memory ?? previous?.memory ?? feedbackMemory('local');
  state.query = query; state.view = 'browse'; $('#search-input').value = query;
  const request = beginRequest();
  if (append) {
    state.searchPage.message = ''; renderLoadMore();
    $('#load-more').textContent = 'Loading GIFs…'; $('#gif-grid').setAttribute('aria-busy', 'true');
  } else {
    state.searchPage = null;
    $('#feedback-actions').hidden = true;
    $('#result-count').textContent = 'Searching by meaning…';
    skeleton();
  }
  try {
    const idsFor = choice => [...votes].filter(([, vote]) => vote === choice).map(([id]) => id);
    const data = await getJSON('/api/tools/feedback', { method: 'POST', body: {
      query, model: 'auto', limit: Number($('#result-limit').value), category: state.category,
      offset: append ? state.searchPage.nextOffset : 0,
      liked: idsFor('liked'), disliked: idsFor('disliked'), memory, unhide: Boolean(update.unhide),
    } });
    if (request !== state.request) return;
    const nextMemory = data.memory ?? memory;
    state.feedback = { votes, memory: nextMemory, vector: data.vector, hidden: data.hidden };
    state.results = [...new Map([...(append ? state.results : []), ...data.results].map(gif => [gif.id, gif])).values()];
    state.searchPage = { nextOffset: data.next_offset, hasMore: data.has_more,
      ended: append && !data.results.length && !data.has_more,
      message: append && !data.results.length && !data.has_more
        ? 'No more matching GIFs. All results with similarity above 0 are shown.' : '' };
    remember(data.results);
    if (update.memory || update.unhide) feedbackMemory('local', nextMemory);
    render();
  } catch (error) {
    if (error.name === 'AbortError' || request !== state.request) return;
    state.feedback = previous;
    logSystem(error, { level: 'error', context: 'search' });
    if (append) {
      state.searchPage.message = ''; render();
    } else {
      if (!previous) state.results = [];
      render(); $('#result-count').textContent = '';
      $('#empty-state').hidden = true; $('#collection-end').hidden = true;
    }
  } finally {
    if (request === state.request) $('#load-more').textContent = 'Load more GIFs ↓';
    finishRequest(request);
  }
}

async function applyHomeVotes(votes, recommend = true) {
  const liked = [...votes].filter(([, vote]) => vote.choice === 'liked');
  const disliked = [...votes].filter(([, vote]) => vote.choice === 'disliked').map(([id]) => id);
  const seed = liked.at(-1)?.[1].description;
  let results = null;
  if (seed && recommend) {
    const request = beginRequest();
    $('#feedback-actions').hidden = true;
    $('#result-count').textContent = 'Finding similar GIFs…';
    skeleton();
    try {
      const data = await getJSON('/api/tools/feedback', { method: 'POST', body: {
        query: seed, model: 'local', limit: PAGE_SIZE + disliked.length, category: state.category,
        liked: liked.map(([id]) => id), disliked,
      } });
      if (request !== state.request) return;
      results = data.results.filter(gif => votes.get(gif.id)?.choice !== 'disliked').slice(0, PAGE_SIZE);
      remember(results);
    } catch (error) {
      if (error.name === 'AbortError' || request !== state.request) return;
      render(); notice(error.message);
      return;
    } finally { finishRequest(request); }
  }
  state.homeVotes = votes;
  state.homeResults = results;
  feedbackMemory('home', [...votes]);
  render();
}

function changeHomeVote(gif, choice) {
  const votes = new Map(state.homeVotes);
  const previous = votes.get(gif.id)?.choice;
  votes.delete(gif.id);
  if (previous !== choice) votes.set(gif.id, { choice, description: gif.description.trim().slice(0, 300) });
  if (votes.size > HOME_VOTE_LIMIT) votes.delete(votes.keys().next().value);
  return applyHomeVotes(votes, choice === 'liked' || state.homeResults !== null);
}

function changeVote(id, choice) {
  const { votes: current, memory: past, vector } = state.feedback;
  const votes = new Map(current);
  if (votes.get(id) === choice) votes.delete(id); else votes.set(id, choice);
  if (votes.size > 100) votes.delete(votes.keys().next().value);
  const memory = past.filter(item => item.gif_id !== id || item.query !== state.query);
  if (votes.get(id) === 'disliked') memory.push({ gif_id: id, query: state.query, query_vector: vector });
  return search(state.query, { votes, memory: memory.slice(-100) });
}

async function searchVideo({ submit = false } = {}) {
  const query = $('#search-input').value.trim();
  if (submit && !query) { $('#search-input').focus(); return; }
  const request = beginRequest();
  const status = $('#search-output-status');
  status.hidden = false;
  $('#video-results').replaceChildren();
  try {
    if (submit) {
      status.textContent = 'Searching videos…';
      const { results } = await getJSON('/api/tools/videos?' + new URLSearchParams({ q: query }));
      if (request !== state.request) return;
      $('#video-results').replaceChildren(...videoCards(results, API_BASE));
      status.textContent = results.length ? `${results.length} videos` : 'No matching videos.';
    } else {
      status.textContent = 'Checking video index…';
      const info = await getJSON('/api/tools/status');
      if (request === state.request) {
        logSystem(info.video, { level: 'warn', context: 'video-search', once: true });
        status.textContent = 'Describe a moment to search indexed video clips.';
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError' && request === state.request) {
      logSystem(error, { level: 'error', context: 'video-search' }); status.textContent = ''; status.hidden = true;
    }
  } finally { finishRequest(request); }
}

function syncSearchMode(mode) {
  state.mode = mode;
  $('#search-mode').value = mode;
  const video = mode === 'video';
  $('#search-input').placeholder = video ? 'Try “a dancing cartoon character”' : 'Try “a dog being confused”';
  $('#search-label').textContent = video ? 'Describe the video you are looking for' : 'Describe the GIF you are looking for';
  $('#search-button').textContent = video ? 'Find videos ↗' : 'Find GIFs ↗';
  $('#search-suggestions').hidden = video;
  $('#search-hint').hidden = video;
  $('#gif-collection').hidden = video;
  $('#search-output').hidden = !video;
  $('#search-output-title').textContent = 'Video results';
  $('#video-results').hidden = !video;
}

async function showSaved() {
  contributionUI?.showExplore();
  if (state.accountLoading) return;
  if (!state.user) { openAuth('login', { view: 'saved' }); return; }
  syncSearchMode('gif');
  state.view = 'saved'; const request = beginRequest(); render();
  try {
    const missing = [...state.saved].filter(id => !state.known.has(id));
    for (let offset = 0; offset < missing.length; offset += 60) {
      const params = new URLSearchParams(); missing.slice(offset, offset + 60).forEach(id => params.append('ids', id));
      const data = await getJSON(`/api/catalog?${params}`); if (request !== state.request) return;
      remember(data.gifs);
    }
    render();
    if ([...state.saved].some(id => !state.known.has(id))) notice('Some saved GIFs are no longer in the current collection.');
  } catch (error) {
    if (error.name !== 'AbortError' && request === state.request) notice(error.message);
  } finally { finishRequest(request); }
}
function openGif(gif) {
  previewGIF = gif;
  const image = document.createElement('img'); image.src = gif.url; image.alt = gif.description; image.referrerPolicy = 'no-referrer';
  image.addEventListener('load', () => pauseImage(image, $('#dialog-preview')), { once: true });
  image.addEventListener('error', () => {
    image.hidden = true;
    logSystem('GIF preview could not be loaded.', { level: 'error', context: `gif-preview:${gif.id}`, once: true });
  }, { once: true });
  $('#dialog-preview').replaceChildren(image); $('#gif-title').textContent = gif.title;
  $('#gif-description').textContent = gif.description; $('#original-link').href = gif.url;
  $('#gif-url').value = new URL(gif.url, location.href).href; $('#copy-button').textContent = 'Copy GIF link ↗'; $('#gif-dialog').showModal();
}

$('#search-form').addEventListener('submit', event => {
  event.preventDefault();
  if (state.mode === 'gif') search($('#search-input').value);
  else searchVideo({ submit: true });
});
$('#search-mode').addEventListener('change', event => {
  const mode = event.target.value;
  syncSearchMode(mode);
  if (mode === 'gif') loadCatalog();
  else searchVideo();
});
$('#feedback-reset').addEventListener('click', () => {
  if (!state.query) return applyHomeVotes(new Map());
  const { votes, memory } = state.feedback;
  search(state.query, { votes: new Map(), memory: memory.filter(item =>
    item.query !== state.query || votes.get(item.gif_id) !== 'disliked') });
});
$('#feedback-unhide').addEventListener('click', () => {
  if (!state.query) {
    const votes = new Map([...state.homeVotes].filter(([, vote]) => vote.choice !== 'disliked'));
    return applyHomeVotes(votes, state.homeResults !== null);
  }
  return search(state.query, { unhide: true });
});
$('#search-input').addEventListener('input', () => { if (state.mode === 'gif' && !$('#search-input').value.trim() && state.query) reset(); });
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => search(button.dataset.query)));
document.querySelectorAll('[data-category]').forEach(button => button.addEventListener('click', () => {
  state.category = button.dataset.category; syncFilters();
  if (state.view === 'saved') render(); else if (state.query) search(state.query); else loadCatalog();
}));
$('#result-limit').addEventListener('change', () => { if (state.query && state.view === 'browse') search(state.query); });
$('#browse-nav').addEventListener('click', reset); $('#reset-button').addEventListener('click', reset);
$('#load-more').addEventListener('click', () => state.query ? search(state.query, { append: true }) : loadCatalog(true));
$('#saved-nav').addEventListener('click', showSaved);
$('#how-button').addEventListener('click', () => $('#how-dialog').showModal());
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
});
$('#copy-button').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('#gif-url').value); logSystem('GIF link copied.', { context: 'clipboard' }); }
  catch (error) { $('#gif-url').focus(); $('#gif-url').select(); logSystem(error, { level: 'error', context: 'clipboard' }); }
});
$('#download-gif-button').addEventListener('click', () => previewGIF && downloadGIF(previewGIF, $('#download-gif-button')));
function syncMotion() {
  $('#motion-button').innerHTML = state.paused ? '<span aria-hidden="true">▷</span> Play GIFs' : '<span aria-hidden="true">Ⅱ</span> Pause GIFs';
  $('#motion-button').setAttribute('aria-pressed', state.paused);
}
$('#motion-button').addEventListener('click', () => { state.paused = !state.paused; syncMotion(); if (!state.loading) render(); });
async function init() {
  syncMotion(); updateSavedCount();
  observeAccount(data => {
    const changed = state.user?.id !== data.user?.id;
    applyAccount(data);
    if (!data.user) state.pendingSavedView = false;
    if (state.pendingSavedView && !data.loading && data.user) {
      state.pendingSavedView = false;
      showSaved();
    }
    else if (changed && state.view === 'saved' && !data.user) reset();
    else if (state.view === 'saved' && !data.loading) showSaved();
    else if (!state.loading) render();
  }, message => {
    state.accountLoading = false;
    $('#account-button').textContent = 'Log in / Sign up';
    $('#account-button').disabled = false;
    $('#saved-nav').disabled = false;
    $('#logout-button').disabled = false;
    notice(message);
  }).catch(error => { applyAccount({ user: null, saved: [] }); logSystem(error, { level: 'error', context: 'account' }); });
  if (state.mode !== 'gif' || state.request) return;
  await loadCatalog();
  try {
    const health = await requestJSON('/api/health', { base: API_BASE });
    if (!health.ready && state.view === 'browse' && !state.query) notice(health.message);
  } catch { /* The catalog error already explains connection failures. */ }
}
contributionUI = initContributions({
  base: API_BASE, getToken: accountToken, syncCloud: syncContribution, toast,
  requestLogin: () => openAuth('login'),
  onNavigate: () => {
    state.controller?.abort(); state.request++; state.view = 'browse';
    finishRequest(state.request);
  },
});
init();
