/* Companion Bench — scenario browser. Renders site/data/scenarios.json
 * (built from packages/companion-bench/src/companion_bench/scenarios/public/
 * by scripts/companion_bench/build_site.py).
 */

(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const escape = (s) => (window.CB_CHARTS && CB_CHARTS.escapeText)
    ? CB_CHARTS.escapeText(s)
    : String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
      }[c]));

  const FAMILY_LABELS = {
    F1: "Continuity",
    F2: "Repair",
    F3: "Personalization",
    F4: "Long absence",
    F5: "Boundary pressure",
    F6: "Goal drift",
  };

  function renderScenario(scn) {
    const fsm = (scn.user_simulator && scn.user_simulator.fsm) || [];
    const fsmRows = fsm.map((step) => `
      <tr>
        <td>${step.session}</td>
        <td>${step.turn}</td>
        <td><code>${escape(step.action)}</code></td>
        <td class="text-muted">${escape(step.payload || "")}</td>
      </tr>
    `).join("");
    const axes = (scn.expected_axes && scn.expected_axes.primary || []).map((a) => `<span class="axis-tag">${escape(a)}</span>`).join(" ");
    const secondary = (scn.expected_axes && scn.expected_axes.secondary || []).map((a) => `<span class="axis-tag" style="opacity:.6">${escape(a)}</span>`).join(" ");
    const hard = scn.expected_axes && scn.expected_axes.hard_constraint
      ? `<span class="cap-tag">hard: ${escape(scn.expected_axes.hard_constraint)}</span>`
      : "";
    const dq = (scn.disqualifiers || []).map((d) => `<code>${escape(d.kind)}</code>`).join(", ");
    return `
      <article class="card" id="${escape(scn.scenario_id)}" data-family="${escape(scn.family)}">
        <h3>
          <span class="family-tag">${escape(scn.family)} ${escape(FAMILY_LABELS[scn.family] || "")}</span>
          ${escape(scn.scenario_id)}
        </h3>
        <p class="meta">
          ${scn.arc_length_sessions} sessions × ${scn.session_turn_range[0]}–${scn.session_turn_range[1]} turns
          · gaps: ${(scn.inter_session_gap_days || []).map(String).join(" / ")} days
          · ${scn.paraphrase_seed_count} paraphrase seeds
        </p>
        <p><strong>Persona:</strong> <em>${escape(scn.user_simulator && scn.user_simulator.persona)}</em></p>
        <p><strong>Axes:</strong> ${axes} ${secondary} ${hard}</p>
        ${dq ? `<p class="text-muted"><strong>Disqualifiers:</strong> ${dq}</p>` : ""}
        ${fsm.length ? `<details>
          <summary>FSM (${fsm.length} steps)</summary>
          <div class="table-wrap" style="margin-top:.5rem;">
            <table>
              <thead><tr><th>S</th><th>T</th><th>Action</th><th>Payload</th></tr></thead>
              <tbody>${fsmRows}</tbody>
            </table>
          </div>
        </details>` : ""}
        <p class="meta" style="margin-top:.6rem;">
          <code style="font-size:.78em;">sha256: ${escape(scn.scenario_hash)}</code>
        </p>
      </article>
    `;
  }

  function applyFilters(scenarios, opts) {
    let filtered = scenarios.slice();
    if (opts.family && opts.family !== "all") {
      filtered = filtered.filter((s) => s.family === opts.family);
    }
    if (opts.search) {
      const q = opts.search.toLowerCase();
      filtered = filtered.filter((s) => {
        const hay = [
          s.scenario_id,
          s.user_simulator && s.user_simulator.persona,
          (s.user_simulator && s.user_simulator.goals || []).join(" "),
        ].join(" ").toLowerCase();
        return hay.includes(q);
      });
    }
    return filtered;
  }

  async function init() {
    const list = $("#scenario-list");
    const familySel = $("#family-filter");
    const searchEl = $("#scenario-search");
    const metaEl = $("#scenarios-meta");
    let payload;
    try {
      const resp = await fetch("data/scenarios.json", { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      payload = await resp.json();
    } catch (e) {
      list.innerHTML = `<p class="text-bad">Could not load <code>data/scenarios.json</code>: ${escape(String(e.message || e))}</p>
        <p class="meta">Generate this file with
          <code>python scripts/companion_bench/build_site.py --scenarios-only --output site/</code>.</p>`;
      return;
    }
    const scenarios = payload.scenarios || [];
    const rerender = () => {
      const filtered = applyFilters(scenarios, {
        family: familySel.value,
        search: (searchEl.value || "").trim(),
      });
      filtered.sort((a, b) => a.scenario_id.localeCompare(b.scenario_id));
      list.innerHTML = filtered.length
        ? `<div class="card-grid">${filtered.map(renderScenario).join("")}</div>`
        : `<p class="text-muted center">No scenarios match.</p>`;
      metaEl.textContent = `${filtered.length} of ${scenarios.length} scenarios shown.`;
    };
    familySel.addEventListener("change", rerender);
    searchEl.addEventListener("input", rerender);
    rerender();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
