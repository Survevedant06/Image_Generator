// sw.js - Service Worker for StudioAI PWA
const CACHE_NAME = 'studioai-cache-v1';
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/static/index.html',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn('[SW] Pre-caching warning:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls, streaming events, uploads, or dynamic image generations
  if (
    url.pathname.startsWith('/api/') ||
    event.request.method !== 'GET' ||
    url.pathname.startsWith('/inputs/') ||
    url.pathname.startsWith('/processed_images/') ||
    url.pathname.startsWith('/generated_scenes/') ||
    url.pathname.startsWith('/exports/') ||
    url.pathname.startsWith('/output_angles/')
  ) {
    return;
  }

  // Network first with cache fallback for shell assets
  event.respondWith(
    fetch(event.request)
      .then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      })
      .catch(() => caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) return cachedResponse;
        if (event.request.mode === 'navigate') {
          return caches.match('/');
        }
      }))
  );
});
