/* Service worker: tệp giao diện ưu tiên bộ nhớ đệm, dữ liệu ưu tiên mạng (có dự phòng ngoại tuyến). */
const V = "vnv-__VERSION__";
const CORE = ["./", "index.html", "style.css", "app.js", "thesis.json", "lib/chart.umd.min.js", "manifest.webmanifest"];
self.addEventListener("install", (e) => { e.waitUntil(caches.open(V).then((c) => c.addAll(CORE)).then(() => self.skipWaiting())); });
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== V).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin) return;
  if (u.pathname.includes("/data/")) {
    e.respondWith(fetch(e.request).then((r) => { const c = r.clone(); caches.open(V).then((x) => x.put(u.pathname, c)); return r; })
      .catch(() => caches.match(u.pathname)));
    return;
  }
  e.respondWith(caches.match(e.request).then((m) => m || fetch(e.request).then((r) => {
    if (r.ok) { const c = r.clone(); caches.open(V).then((x) => x.put(e.request, c)); }
    return r;
  })));
});
