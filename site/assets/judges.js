/* Companion Bench — judges page renderer.
 *
 * Reads site/data/judge_calibration.json with shape:
 *   {
 *     "rotations": [
 *       { "quarter": "2026Q2", "perturn": "anthropic/claude-3.7-sonnet",
 *         "arc": "openai/gpt-5", "chair_signoff": "alice@example",
 *         "notes": "..." }
 *     ],
 *     "agreement": [
 *       { "axis": "A1", "spearman": 0.78 }, ...
 *     ],
 *     "calibration_points": [
 *       { "axis": "A3", "judge": 76.4, "human": 71.2, "label": "..." }, ...
 *     ]
 *   }
 *
 * If the file is missing or empty, the page renders empty-state copy gracefully.
 */

(function () {
  "use strict";
  const $ = (sel, root) => (root || document).querySelector(sel);
  const escape = (s) => (window.CB_CHARTS && CB_CHARTS.escapeText)
    ? CB_CHARTS.escapeText(s)
    : String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
      }[c]));

  function renderRotation(rotations) {
    const tbody = $("#rotation-table tbody");
    if (!rotations || !rotations.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="text-muted center">
        No rotations recorded yet. The first quarterly rotation will be
        signed off by the working group on formation.
      </td></tr>`;
      return;
    }
    tbody.innerHTML = rotations.map((r) => `
      <tr>
        <td>${escape(r.quarter)}</td>
        <td><code>${escape(r.perturn)}</code></td>
        <td><code>${escape(r.arc)}</code></td>
        <td>${escape(r.chair_signoff || "—")}</td>
        <td class="text-muted">${escape(r.notes || "")}</td>
      </tr>
    `).join("");
  }

  function renderAgreement(agreement) {
    const host = $("#agreement-bars");
    if (!agreement || !agreement.length) {
      host.innerHTML = `<p class="text-muted">
        No agreement data yet. The first golden-set calibration is scheduled
        to coincide with the v1.0 release run.
      </p>`;
      return;
    }
    if (!window.CB_CHARTS || !CB_CHARTS.axisBars) {
      host.innerHTML = `<p class="text-muted">Charts unavailable.</p>`;
      return;
    }
    const items = agreement.map((a) => ({
      label: a.axis,
      value: (a.spearman || 0) * 100,
      fill: (a.spearman || 0) >= 0.6 ? "var(--good)" : "var(--bad)",
    }));
    host.innerHTML = CB_CHARTS.axisBars(items, { width: 420, decimals: 0, aria: "judge agreement" })
      + `<p class="meta">Spearman ρ × 100 (per axis). Threshold for re-rotation: 60.</p>`;
  }

  function renderCalibration(points) {
    const host = $("#calibration-scatter");
    if (!points || !points.length) {
      host.innerHTML = `<p class="text-muted">
        No calibration points yet. Once the human-eval golden set is
        established, scatter will populate here.
      </p>`;
      return;
    }
    if (!window.CB_CHARTS || !CB_CHARTS.calibrationScatter) {
      host.innerHTML = `<p class="text-muted">Charts unavailable.</p>`;
      return;
    }
    host.innerHTML = CB_CHARTS.calibrationScatter(
      points.map((p) => ({ x: p.judge, y: p.human, label: `${p.axis}` })),
      { aria: "judge vs human calibration" }
    );
  }

  function renderEmptyMeta(payload) {
    const meta = $("#judges-meta");
    if (!payload || (!payload.rotations || !payload.rotations.length)) {
      meta.innerHTML = `
        <strong>Bootstrap state:</strong> the v1.0 release run uses the default
        rotation (per-turn = anthropic/claude-3.7-sonnet, arc = openai/gpt-5).
        Quarterly rotations begin once the working group is formed.
      `;
    } else if (payload.generated_at) {
      meta.textContent = `Last updated ${payload.generated_at.replace("T", " ").split(".")[0]} UTC`;
    }
  }

  async function init() {
    let payload = null;
    try {
      const resp = await fetch("data/judge_calibration.json", { cache: "no-store" });
      if (resp.ok) payload = await resp.json();
    } catch (_) { /* graceful */ }
    payload = payload || {};
    renderEmptyMeta(payload);
    renderRotation(payload.rotations || []);
    renderAgreement(payload.agreement || []);
    renderCalibration(payload.calibration_points || []);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
