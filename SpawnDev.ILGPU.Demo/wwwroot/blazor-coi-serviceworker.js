/*! coi-serviceworker - Cross-Origin Isolation via Service Worker */
/*
 * This service worker intercepts all responses and adds the required
 * Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
 * to enable SharedArrayBuffer support in the browser.
 *
 * Works locally and on static hosts like GitHub Pages where you
 * cannot set server response headers.
 *
 * Registration: Add <script src="blazor-coi-serviceworker.js"></script> to index.html
 */

if (typeof window !== 'undefined') {
    // --- Running as a regular script in the page context ---

    // Set to true to enable verbose logging of service worker registration and activation flow. Useful for debugging but can be noisy in normal use.
    var verbose = false;
    function consoleLog(...args) {
        if (!verbose) return;
        console.log("[COI]", ...args);
    }
    
    // Helper to check if Blazor script is loaded
    function isBlazorRunning() {
        return !!document.querySelector('script[src*="blazor.webassembly.js"]');
    }

    // Start Blazor immediately if we're already cross-origin isolated.
    if (window.crossOriginIsolated && !isBlazorRunning()) {
        // Already cross-origin isolated — nothing to do
        consoleLog("[COI] Cross-origin isolated ✓");
        var s = document.createElement("script");
        s.src = "_framework/blazor.webassembly.js";
        document.body.appendChild(s);
    }

    // First check if service workers are supported before trying to register one.
    if ("serviceWorker" in navigator) {
        // Register the service worker, which will add the necessary headers to enable
        // cross-origin isolation. The SW will reload the page once activated to apply the headers.
        consoleLog("[COI] Registering service worker...");
        navigator.serviceWorker
            .register(window.document.currentScript.src)
            .then(function (reg) {
                consoleLog("[COI] Service worker registered:", reg.scope);
                if (!isBlazorRunning()) {
                    if (navigator.serviceWorker.controller) {
                        // SW is controlling the page but we're not isolated.
                        // This can happen if the SW was just updated or headers aren't applied yet.
                        // Reload to pick up the headers from the active SW.
                        consoleLog("[COI] Controlled by SW but not isolated. Reloading...");
                        window.location.reload();
                    } else {
                        // SW registered but not controlling the page yet. Wait for it to activate and take control, then reload.
                        function waitForActivation(worker) {
                            worker.addEventListener("statechange", function () {
                                consoleLog("[COI] Worker statechange — .", worker.state);
                                if (worker.state === "activated") {
                                    consoleLog("[COI] Worker activated — reloading.");
                                    window.location.reload();
                                }
                            });
                        }

                        if (reg.installing) {
                            // SW is installing — wait for it to activate
                            waitForActivation(reg.installing);
                        } else if (reg.waiting) {
                            // SW installed but waiting — wait for activation
                            waitForActivation(reg.waiting);
                        } else if (reg.active) {
                            // SW is active but not controlling this page yet.
                            // clients.claim() in the SW will trigger "controllerchange".
                            navigator.serviceWorker.addEventListener("controllerchange", function () {
                                consoleLog("[COI] Controller changed — reloading.");
                                window.location.reload();
                            });
                        }

                        // Fallback reload in case something goes wrong with the activation flow
                        setTimeout(function () {
                            if (document.querySelector('script[src*="blazor.webassembly.js"]')) return;
                            consoleLog("[COI] Reloading as fallback in case activation flow fails.");
                            window.location.reload();
                        }, 1000);
                    }
                }
            })
            .catch(function (err) {
                console.error("[COI] Service worker registration failed:", err);
            });
    }
} else {
    // --- Running as a Service Worker ---
    self.addEventListener("install", function () { self.skipWaiting(); });
    self.addEventListener("activate", function (event) { event.waitUntil(self.clients.claim()); });

    self.addEventListener("fetch", function (event) {
        // Skip requests the SW can't meaningfully intercept
        if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") {
            return;
        }

        // Don't intercept cross-origin requests — we can't add useful
        // headers to them and they are the main source of "Failed to fetch" errors
        var url = new URL(event.request.url);
        if (url.origin !== self.location.origin) {
            return; // Let the browser handle it natively
        }

        event.respondWith(
            fetch(event.request)
                .then(function (response) {
                    var newHeaders = new Headers(response.headers);
                    newHeaders.set("Cross-Origin-Embedder-Policy", "credentialless");
                    newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");

                    return new Response(response.body, {
                        status: response.status,
                        statusText: response.statusText,
                        headers: newHeaders,
                    });
                })
                .catch(function (e) {
                    consoleLog("[COI] Fetch failed for:", event.request.url, e.message);
                    return new Response("Service Worker fetch failed", {
                        status: 502,
                        statusText: "Service Worker Fetch Failed",
                    });
                })
        );
    });
}
