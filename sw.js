/* ============================================================
   MÖBIUS — Service Worker
   Estrategia: Cache-First para assets de la app.
   Cache separada y persistente para opencv.js (~8 MB WASM).
   ============================================================ */
'use strict';

const CACHE_VERSION    = 'v6';
const APP_CACHE_NAME   = `mobius-app-${CACHE_VERSION}`;
const CV_CACHE_NAME    = `mobius-opencv-${CACHE_VERSION}`;

const APP_ASSETS = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
];

const OPENCV_ORIGIN = 'https://docs.opencv.org';

// ── INSTALL: pre-cachear assets de la app ───────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(APP_CACHE_NAME).then(cache => {
      return cache.addAll(APP_ASSETS).catch(err => {
        // En dev (file://) algunos assets pueden fallar; no bloquear.
        console.warn('[SW] Pre-cache parcial:', err);
      });
    })
  );
  self.skipWaiting();
});

// ── ACTIVATE: limpiar caches obsoletas ──────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== APP_CACHE_NAME && k !== CV_CACHE_NAME)
          .map(k => {
            console.log('[SW] Eliminando cache obsoleta:', k);
            return caches.delete(k);
          })
      )
    ).then(() => self.clients.claim())
  );
});

// ── FETCH: lógica de cache por tipo de recurso ───────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // ① opencv.js: cache separada y durable
  if (url.origin === OPENCV_ORIGIN || request.url.includes('opencv.js')) {
    event.respondWith(cacheFirst(request, CV_CACHE_NAME));
    return;
  }

  // ② Assets de la propia app: cache-first
  if (url.origin === self.location.origin) {
    event.respondWith(cacheFirst(request, APP_CACHE_NAME));
    return;
  }

  // ③ Cualquier otra request: network normal (fonts de Google, etc.)
  // No cacheamos fuentes externas aquí; el navegador lo hace solo.
});

/**
 * Estrategia Cache-First con actualización en background.
 * @param {Request} request
 * @param {string}  cacheName
 */
async function cacheFirst(request, cacheName) {
  const cache  = await caches.open(cacheName);
  const cached = await cache.match(request);

  if (cached) {
    // Actualizar en background (stale-while-revalidate silencioso)
    fetchAndCache(request, cache).catch(() => {});
    return cached;
  }

  return fetchAndCache(request, cache);
}

async function fetchAndCache(request, cache) {
  const response = await fetch(request);
  if (response.ok && request.method === 'GET') {
    cache.put(request, response.clone());
  }
  return response;
}
