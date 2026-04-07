const CACHE_NAME = 'audiometria-ai-v1';
const ASSETS = [
  './index.html',
  './style.css',
  './script.js',
  './manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Network-only for API calls
  if (event.request.url.includes('generativelanguage.googleapis.com')) {
    event.respondWith(fetch(event.request));
    return;
  }
  // Cache first, then network for app assets
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
