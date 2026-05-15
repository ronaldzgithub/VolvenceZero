/* Companion Bench — pairwise comparison viewer.
 *
 * Reads:
 *   data/aggregate_results.json  → list of systems for the selectors
 *   data/pairwise.json           → per-arc winners + axis margins + ELO
 *   data/submissions/<id>.json   → full transcripts for each side
 *
 * Behaviour:
 *   - Pick two systems → list of shared scenarios automatically populates.
 *   - Select a scenario → side-by-side multi-session transcripts render.
 *   - Per-axis margin bars highlight which side won (or tied).
 *   - Hovering a (session, turn) cell highlights the matching cell on the
 *     other side so the eye can follow the divergence point.
 *
 * URL params: ?a=<id>&b=<id>&s=<scenario_id> are deep-linkable.
 */

(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const escape = (s) => (window.CB_CHARTS && CB_CHARTS.escapeText)
    ? CB_CHARTS.escapeText(s)
    : String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
      }[c]));
  const fmt = (v, d) => Number.isFinite(v) ? v.toFixed(d == null ? 1 : d) : "—";

  const AXES = ["A1", "A2", "A3", "A4", "A5", "A6"];

  let SYSTEMS = [];
  let PAIRWISE = null;          // { arcs: [...], elo: {...} }
  const SUB_CACHE = new Map();  // submission_id -> detail json

  function param(k) {
    return new URLSearchParams(window.location.search).get(k);
  }
  function setParam(k, v) {
    const u = new URL(window.location.href);
    if (v == null || v === "") u.searchParams.delete(k);
    else u.searchParams.set(k, v);
    window.history.replaceState({}, "", u.toString());
  }

  function fillSystemSelect(sel) {
    sel.innerHTML = SYSTEMS
      .map((s) => `<option value="${escape(s.submission_id)}">${escape(s.system_name)} · ${escape(s.leaderboard_category)}</option>`)
      .join("");
  }

  async function loadDetail(id) {
    if (SUB_CACHE.has(id)) return SUB_CACHE.get(id);
    const url = `data/submissions/${encodeURIComponent(id)}.json`;
    try {
      const resp = await fetch(url, { cache: "no-store" });
      if (!resp.ok) {
        SUB_CACHE.set(id, null);
        return null;
      }
      const data = await resp.json();
      SUB_CACHE.set(id, data);
      return data;
    } catch (_) {
      SUB_CACHE.set(id, null);
      return null;
    }
  }

  function findArc(detail, scenarioId) {
    if (!detail || !Array.isArray(detail.arcs)) return null;
    return detail.arcs.find((a) => a.scenario_id === scenarioId) || null;
  }

  function arcMatch(systemA, systemB, scenarioId) {
    if (!PAIRWISE || !Array.isArray(PAIRWISE.arcs)) return null;
    return PAIRWISE.arcs.find((a) =>
      a.scenario_id === scenarioId &&
      ((a.system_a === systemA && a.system_b === systemB) ||
       (a.system_b === systemA && a.system_a === systemB))
    );
  }

  function fillScenarioSelect(systemA, systemB) {
    const sel = $("#scenario-pick");
    if (!PAIRWISE || !Array.isArray(PAIRWISE.arcs)) {
      sel.innerHTML = `<option value="">(no shared arcs)</option>`;
      return null;
    }
    const sharedScenarios = new Set();
    PAIRWISE.arcs.forEach((a) => {
      if ((a.system_a === systemA && a.system_b === systemB) ||
          (a.system_b === systemA && a.system_a === systemB)) {
        sharedScenarios.add(a.scenario_id);
      }
    });
    const sorted = Array.from(sharedScenarios).sort();
    if (!sorted.length) {
      sel.innerHTML = `<option value="">(no shared arcs)</option>`;
      return null;
    }
    sel.innerHTML = sorted.map((s) => `<option value="${escape(s)}">${escape(s)}</option>`).join("");
    return sorted[0];
  }

  function renderHeadToHead(systemA, systemB, scenarioId) {
    if (!PAIRWISE || !Array.isArray(PAIRWISE.arcs)) {
      return `<p class="text-muted">No pairwise data available yet.</p>`;
    }
    const overall = PAIRWISE.arcs.filter((a) =>
      (a.system_a === systemA && a.system_b === systemB) ||
      (a.system_b === systemA && a.system_a === systemB)
    );
    if (!overall.length) {
      return `<p class="text-muted">No shared scenarios scored between these two systems yet.</p>`;
    }
    let aWins = 0, bWins = 0, ties = 0;
    overall.forEach((m) => {
      const margin = m.system_a === systemA ? m.score_margin : -m.score_margin;
      if (margin > 0.5) aWins++;
      else if (margin < -0.5) bWins++;
      else ties++;
    });
    const elo = (PAIRWISE.elo && PAIRWISE.elo.trueskill) || [];
    const tsLookup = Object.fromEntries(elo.map((e) => [e.system, e]));
    const tsA = tsLookup[systemA];
    const tsB = tsLookup[systemB];
    const tsLine = (tsA && tsB)
      ? ` · TrueSkill conservative: A ${fmt(tsA.conservative, 1)} · B ${fmt(tsB.conservative, 1)}`
      : "";
    return `
      <p class="meta">
        <strong>Overall (across all shared arcs):</strong>
        ${overall.length} arcs · A wins ${aWins} · ties ${ties} · B wins ${bWins}${tsLine}
      </p>
      ${renderScenarioMargin(systemA, systemB, scenarioId)}
    `;
  }

  function renderScenarioMargin(systemA, systemB, scenarioId) {
    const match = arcMatch(systemA, systemB, scenarioId);
    if (!match) return "";
    // Normalise so margin is always (A − B).
    const flip = match.system_a !== systemA;
    const scoreA = flip ? match.score_b : match.score_a;
    const scoreB = flip ? match.score_a : match.score_b;
    const margin = scoreA - scoreB;
    const verdict = margin > 0.5 ? `A leads by ${fmt(margin, 2)}`
                  : margin < -0.5 ? `B leads by ${fmt(-margin, 2)}`
                  : "tie within 0.5";
    const axisItems = AXES.map((a) => {
      const raw = (match.axis_margins && match.axis_margins[a]) || 0;
      const m = flip ? -raw : raw;
      return {
        label: a,
        value: 50 + Math.max(-50, Math.min(50, m)),  // shift to 0..100 for the bar
        fill: m > 0 ? "var(--good)" : m < 0 ? "var(--bad)" : "var(--text-muted)",
        meta: m,
      };
    });
    const barsHtml = axisItems.map((it) => {
      const fillPct = (it.value / 100) * 100;
      const dir = it.meta > 0 ? "A" : it.meta < 0 ? "B" : "tie";
      return `
        <div class="axis-row">
          <span class="label">${it.label}</span>
          <span class="bar"><span class="fill" style="width:${fillPct.toFixed(1)}%; background:${it.fill};"></span></span>
          <span class="value" title="margin = A − B">${it.meta > 0 ? "+" : ""}${fmt(it.meta, 1)} ${it.meta > 0 ? "(A)" : it.meta < 0 ? "(B)" : ""}</span>
        </div>
      `;
    }).join("");
    return `
      <div class="card" style="margin-top:1rem;">
        <h3 class="mt-0">Scenario margin · ${escape(scenarioId)}</h3>
        <p class="meta">
          A score: <strong>${fmt(scoreA, 2)}</strong> · B score: <strong>${fmt(scoreB, 2)}</strong> · ${verdict}
        </p>
        ${barsHtml}
      </div>
    `;
  }

  function turnKey(s, t) { return `S${s}T${t}`; }

  function renderTranscriptColumn(arc, label, side) {
    if (!arc) {
      return `<div><h3>${escape(label)}</h3><p class="text-muted">No data for this scenario.</p></div>`;
    }
    const sessions = (arc.sessions || []).map((session) => {
      const turns = (session.turns || []).map((t) => `
        <div class="turn user" data-turn-key="${turnKey(session.session_index, t.turn_index)}" data-side="${side}">
          <div class="turn-label">
            <span>User · S${session.session_index} T${t.turn_index}</span>
            ${t.fsm_action ? `<span class="fsm-marker">${escape(t.fsm_action)}</span>` : ""}
          </div>
          <div class="turn-text">${escape(t.user_text)}</div>
        </div>
        <div class="turn asst" data-turn-key="${turnKey(session.session_index, t.turn_index)}" data-side="${side}">
          <div class="turn-label"><span>Assistant</span></div>
          <div class="turn-text">${escape(t.assistant_text)}</div>
        </div>
      `).join("");
      return `
        <div class="session open">
          <div class="session-header"><span class="toggle"></span>
            <span>Session ${session.session_index} · ${session.turns.length} turns
                  ${session.inter_session_gap_days ? "· +" + session.inter_session_gap_days + " day(s)" : ""}</span>
            <span class="meta">${escape(session.session_id)}</span>
          </div>
          <div class="session-body">${turns}</div>
        </div>
      `;
    }).join("");
    return `<div><h3>${escape(label)}</h3>${sessions || "<p class='text-muted'>No sessions.</p>"}</div>`;
  }

  function bindTurnCorrelation() {
    $$("#compare-grid .turn").forEach((el) => {
      const key = el.dataset.turnKey;
      el.addEventListener("mouseenter", () => {
        $$(`#compare-grid .turn[data-turn-key="${key}"]`).forEach((t) => t.classList.add("turn-hover"));
      });
      el.addEventListener("mouseleave", () => {
        $$(`#compare-grid .turn[data-turn-key="${key}"]`).forEach((t) => t.classList.remove("turn-hover"));
      });
    });
    $$("#compare-grid .session-header").forEach((h) =>
      h.addEventListener("click", () => h.parentElement.classList.toggle("open"))
    );
  }

  async function rerender() {
    const a = $("#system-a").value;
    const b = $("#system-b").value;
    const scn = $("#scenario-pick").value;
    setParam("a", a);
    setParam("b", b);
    setParam("s", scn);
    $("#compare-overview").innerHTML = renderHeadToHead(a, b, scn);
    if (!a || !b || !scn) {
      $("#compare-grid").innerHTML = "";
      return;
    }
    const [detailA, detailB] = await Promise.all([loadDetail(a), loadDetail(b)]);
    const arcA = findArc(detailA, scn);
    const arcB = findArc(detailB, scn);
    const sysAName = (SYSTEMS.find((s) => s.submission_id === a) || {}).system_name || a;
    const sysBName = (SYSTEMS.find((s) => s.submission_id === b) || {}).system_name || b;
    $("#compare-grid").innerHTML = `
      ${renderTranscriptColumn(arcA, sysAName + " (A)", "A")}
      ${renderTranscriptColumn(arcB, sysBName + " (B)", "B")}
    `;
    bindTurnCorrelation();
  }

  async function init() {
    let agg;
    try {
      agg = await (await fetch("data/aggregate_results.json", { cache: "no-store" })).json();
    } catch (e) {
      $("#compare-grid").innerHTML = `<p class="text-bad">Could not load <code>data/aggregate_results.json</code>: ${escape(String(e.message || e))}</p>`;
      return;
    }
    SYSTEMS = (agg.systems || []).slice().sort((x, y) => (y.companionbench_final || 0) - (x.companionbench_final || 0));
    if (SYSTEMS.length < 2) {
      $("#compare-meta").innerHTML = `Need at least two systems on the leaderboard to compare. Add submissions and re-run <code>build_site.py</code>.`;
      return;
    }
    try {
      const resp = await fetch("data/pairwise.json", { cache: "no-store" });
      if (resp.ok) PAIRWISE = await resp.json();
    } catch (_) { /* optional */ }

    const initialA = param("a") || SYSTEMS[0].submission_id;
    const initialB = param("b") || (SYSTEMS[1] || SYSTEMS[0]).submission_id;
    fillSystemSelect($("#system-a"));
    fillSystemSelect($("#system-b"));
    $("#system-a").value = initialA;
    $("#system-b").value = initialB;

    const firstScn = fillScenarioSelect(initialA, initialB);
    const initialScn = param("s") || firstScn;
    if (initialScn) $("#scenario-pick").value = initialScn;

    const onSysChange = () => {
      const newScn = fillScenarioSelect($("#system-a").value, $("#system-b").value);
      if (newScn) $("#scenario-pick").value = newScn;
      rerender();
    };
    $("#system-a").addEventListener("change", onSysChange);
    $("#system-b").addEventListener("change", onSysChange);
    $("#scenario-pick").addEventListener("change", rerender);

    rerender();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
