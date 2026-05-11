/* Companion Bench — shared header/footer partials.
 *
 * Every page hosts <div id="site-header"></div> and <div id="site-footer"></div>;
 * this script injects the chrome so we don't ship the same HTML 10 times.
 *
 * Active-page highlighting is derived from <body data-page="...">.
 */

(function () {
  "use strict";

  const NAV = [
    { href: "leaderboard.html", label: "Leaderboard", page: "leaderboard" },
    { href: "compare.html", label: "Compare", page: "compare" },
    { href: "methodology.html", label: "Methodology", page: "methodology" },
    { href: "scenarios.html", label: "Scenarios", page: "scenarios" },
    { href: "judges.html", label: "Judges", page: "judges" },
    { href: "submit.html", label: "Submit", page: "submit" },
    { href: "governance.html", label: "Governance", page: "governance" },
    { href: "about.html", label: "About", page: "about" },
  ];

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
    }[c]));
  }

  function pathPrefix() {
    // Pages under /results/ or other subdirs need to walk up.
    const depth = window.location.pathname.split("/").filter((s) => s && !s.endsWith(".html")).length - 1;
    return depth > 0 ? "../".repeat(depth) : "";
  }

  function buildHeader(activePage) {
    const prefix = pathPrefix();
    const links = NAV
      .map((item) => {
        const cls = item.page === activePage ? "active" : "";
        return `<a href="${prefix}${item.href}" class="${cls}">${escapeHtml(item.label)}</a>`;
      })
      .join("");
    return `
      <div class="container">
        <div class="brand">
          <a href="${prefix}index.html">Companion Bench</a>
          <span class="version" id="cb-version">v1.0</span>
        </div>
        <nav>${links}</nav>
        <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
          </svg>
        </button>
      </div>
    `;
  }

  function buildFooter() {
    return `
      <div class="container">
        <div>
          <p>Companion Bench v1.0 reference implementation
             &middot; Apache 2.0 (code) &middot; CC BY 4.0 (RFC + docs)</p>
          <p>Public scenarios reproducible by SHA-256; held-out scenarios are private.</p>
        </div>
        <div>
          <p>
            <a href="https://github.com/VolvenceZero/companion-bench" rel="noopener" target="_blank">GitHub</a> &middot;
            <a href="https://github.com/VolvenceZero/companion-bench/blob/main/docs/external/companion-bench-rfc-v0.md" rel="noopener" target="_blank">RFC v0.1</a> &middot;
            <a href="https://github.com/VolvenceZero/companion-bench/issues" rel="noopener" target="_blank">Issues</a>
          </p>
          <p>Previously circulated as <strong>LSCB</strong>.</p>
        </div>
      </div>
    `;
  }

  function injectChrome() {
    const activePage = document.body && document.body.dataset
      ? document.body.dataset.page || ""
      : "";
    const headerHost = document.getElementById("site-header");
    const footerHost = document.getElementById("site-footer");
    if (headerHost) {
      headerHost.className = "site-header";
      headerHost.innerHTML = buildHeader(activePage);
      const toggle = document.getElementById("theme-toggle");
      if (toggle) toggle.addEventListener("click", toggleTheme);
    }
    if (footerHost) {
      footerHost.className = "site-footer";
      footerHost.innerHTML = buildFooter();
    }
  }

  function applyStoredTheme() {
    const stored = localStorage.getItem("cb-theme");
    if (stored === "dark") {
      document.body.classList.add("theme-dark");
    } else if (stored === "light") {
      document.body.classList.remove("theme-dark");
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.body.classList.add("theme-dark");
    }
  }

  function toggleTheme() {
    const isDark = document.body.classList.toggle("theme-dark");
    localStorage.setItem("cb-theme", isDark ? "dark" : "light");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      applyStoredTheme();
      injectChrome();
    });
  } else {
    applyStoredTheme();
    injectChrome();
  }

  // Expose for debugging.
  window.CB_PARTIALS = { toggleTheme, NAV };
})();
