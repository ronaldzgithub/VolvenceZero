/* Companion Bench — leaderboard renderer.
 *
 * Reads site/data/aggregate_results.json (built by
 * scripts/companion_bench/build_site.py) and renders the leaderboard
 * table + an axis breakdown panel.
 *
 * Two modes:
 *   - data-mode="full"   → renders the full leaderboard with filter/sort/search.
 *   - data-mode="preview" → top-N preview for the landing page.
 */

(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function escapeHtml(s) {
    return (window.CB_CHARTS && CB_CHARTS.escapeText)
      ? CB_CHARTS.escapeText(s)
      : String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
          "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
        }[c]));
  }

  function fmtAxis(v) {
    return Number.isFinite(v) ? v.toFixed(1) : "—";
  }
  function fmtScore(v) {
    return Number.isFinite(v) ? v.toFixed(2) : "—";
  }
  function categoryLabel(cat) {
    return {
      "open-weight": "Open weight",
      "closed-api": "Closed API",
      "bespoke": "Bespoke",
    }[cat] || cat;
  }

  function detailHref(submissionId) {
    const u = new URL("results/", window.location.href.replace(/[^/]*$/, ""));
    u.searchParams.set("s", submissionId);
    return u.pathname + u.search;
  }

  function sortValue(row, key) {
    switch (key) {
      case "companionbench_final": return row.companionbench_final;
      case "trueskill": return row.trueskill_conservative;
      case "bradley_terry": return row.bradley_terry_score;
      case "A1":
      case "A2":
      case "A3":
      case "A4":
      case "A5":
      case "A6":
        return (row.axis_means && row.axis_means[key]) != null ? row.axis_means[key] : 0;
      default:
        return 0;
    }
  }

  function applyFilters(systems, opts) {
    const cat = opts.category;
    const search = (opts.search || "").trim().toLowerCase();
    const sortKey = opts.sort;
    let filtered = systems.slice();
    if (cat && cat !== "all") {
      filtered = filtered.filter((s) => s.leaderboard_category === cat);
    }
    if (search) {
      filtered = filtered.filter((s) => {
        const hay = `${s.system_name || ""} ${s.model_identifier || ""} ${s.submission_id || ""}`.toLowerCase();
        return hay.includes(search);
      });
    }
    filtered.sort((a, b) => sortValue(b, sortKey) - sortValue(a, sortKey));
    return filtered;
  }

  function renderRow(row, i, mode) {
    const detail = detailHref(row.submission_id);
    const axisCells = ["A1", "A2", "A3", "A4", "A5", "A6"]
      .map((a) => `<td class="numeric">${fmtAxis(row.axis_means && row.axis_means[a])}</td>`).join("");
    const cap = row.a6_cap_applied ? `<span class="cap-tag" title="Final score capped at 50 because A6 < 60">A6 cap</span>` : "";
    const idLine = row.model_identifier
      ? `<span class="system-id">${escapeHtml(row.model_identifier)}</span>`
      : `<span class="system-id">${escapeHtml(row.submission_id)}</span>`;
    if (mode === "preview") {
      return `
        <tr>
          <td>${i + 1}</td>
          <td><a href="${detail}"><span class="system-name">${escapeHtml(row.system_name)}</span>${idLine}</a></td>
          <td><span class="category-tag">${escapeHtml(categoryLabel(row.leaderboard_category))}</span></td>
          <td class="numeric companionbench-cell">${fmtScore(row.companionbench_final)}</td>
          <td class="numeric">${fmtAxis(row.axis_means && row.axis_means.A3)}</td>
          <td class="numeric">${fmtAxis(row.axis_means && row.axis_means.A6)}</td>
          <td>${cap}</td>
        </tr>
      `;
    }
    return `
      <tr>
        <td>${i + 1}</td>
        <td><a href="${detail}"><span class="system-name">${escapeHtml(row.system_name)}</span>${idLine}</a></td>
        <td><span class="category-tag">${escapeHtml(categoryLabel(row.leaderboard_category))}</span></td>
        <td class="numeric companionbench-cell">${fmtScore(row.companionbench_final)}</td>
        ${axisCells}
        <td class="numeric">${fmtScore(row.trueskill_conservative)}</td>
        <td class="numeric">${fmtScore(row.bradley_terry_score)}</td>
        <td>${cap}</td>
      </tr>
    `;
  }

  function renderTable(host, systems, mode) {
    const tbody = $("tbody", host);
    if (!tbody) return;
    if (!systems.length) {
      tbody.innerHTML = `<tr><td colspan="13" class="text-muted center">No systems match the current filters.</td></tr>`;
      return;
    }
    tbody.innerHTML = systems.map((row, i) => renderRow(row, i, mode)).join("");
  }

  function renderMeta(payload) {
    const metaEl = $("#aggregate-meta");
    if (metaEl) {
      const parts = [];
      if (payload.companion_bench_version || payload.companionbench_version) {
        parts.push(`Companion Bench v${payload.companion_bench_version || payload.companionbench_version}`);
      }
      if (payload.generated_at) parts.push(`generated ${payload.generated_at.replace("T", " ").split(".")[0]} UTC`);
      if (Array.isArray(payload.systems)) parts.push(`${payload.systems.length} systems`);
      metaEl.textContent = parts.join(" · ");
    }
    const versionEl = $("#cb-version");
    if (versionEl && (payload.companion_bench_version || payload.companionbench_version)) {
      versionEl.textContent = `v${payload.companion_bench_version || payload.companionbench_version}`;
    }
    const banner = $("#demo-banner");
    if (banner) banner.hidden = !payload.demo;
  }

  async function fetchData(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  function initFor(host, dataUrl) {
    const mode = host.dataset.mode || "full";
    fetchData(dataUrl).then((payload) => {
      renderMeta(payload);
      const systems = Array.isArray(payload.systems) ? payload.systems : [];
      const stateSelectors = {
        category: $("#category-filter"),
        sort: $("#sort-by"),
        search: $("#search"),
      };
      const readState = () => ({
        category: stateSelectors.category ? stateSelectors.category.value : "all",
        sort: stateSelectors.sort ? stateSelectors.sort.value : "companionbench_final",
        search: stateSelectors.search ? stateSelectors.search.value : "",
      });
      const rerender = () => {
        const filtered = applyFilters(systems, readState());
        const limited = mode === "preview" ? filtered.slice(0, 10) : filtered;
        renderTable(host, limited, mode);
      };
      if (stateSelectors.category) stateSelectors.category.addEventListener("change", rerender);
      if (stateSelectors.sort) stateSelectors.sort.addEventListener("change", rerender);
      if (stateSelectors.search) stateSelectors.search.addEventListener("input", rerender);
      rerender();
    }).catch((err) => {
      const tbody = $("tbody", host);
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="13" class="text-bad">
          Could not load <code>${escapeHtml(dataUrl)}</code>: ${escapeHtml(String(err.message || err))}
        </td></tr>`;
      }
    });
  }

  async function renderForest() {
    const host = document.getElementById("trueskill-forest");
    const empty = document.getElementById("forest-empty");
    if (!host || !window.CB_CHARTS || !CB_CHARTS.forestPlot) return;
    let data;
    try {
      const resp = await fetch("data/pairwise.json", { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
    } catch (e) {
      host.innerHTML = "";
      if (empty) empty.hidden = false;
      return;
    }
    const ts = (data.elo && data.elo.trueskill) || [];
    if (!ts.length) {
      host.innerHTML = "";
      if (empty) empty.hidden = false;
      return;
    }
    const items = ts.slice().sort((a, b) => (b.conservative || 0) - (a.conservative || 0))
      .map((r) => ({ label: r.system, mu: r.mu, sigma: r.sigma }));
    host.innerHTML = CB_CHARTS.forestPlot(items, { aria: "TrueSkill mu plus or minus 3 sigma" });
  }

  function init() {
    const tables = $$("[data-leaderboard]");
    tables.forEach((tbl) => {
      const url = tbl.dataset.url || "data/aggregate_results.json";
      initFor(tbl, url);
    });
    renderForest();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
