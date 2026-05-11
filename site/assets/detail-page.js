/* Companion Bench — per-submission detail page renderer.
 *
 * Reads ../data/submissions/<id>.json (built by build_site.py).
 * Single template; the URL is /results/?s=<submission_id>.
 *
 * The detail JSON schema:
 *   {
 *     submission_id, system_name, model_identifier,
 *     leaderboard_category, generated_at,
 *     manifest: { ... declared params ... },
 *     verifier: { state, arc_id?, ... },
 *     aggregate: {
 *       lscb_final, raw, a6_cap_applied, a6_cap_fraction,
 *       axis_means: {A1..A6: float},
 *       axis_ci95:  {A1..A6: [lo,hi]},
 *       trueskill_conservative, bradley_terry_score,
 *       arc_count
 *     },
 *     family_means: { F1: { mean, ci95: [lo,hi], arc_count }, ... },
 *     cost: { totals: {...}, by_phase: [...], missing_models: [...] },
 *     arcs: [
 *       {
 *         arc_id, scenario_id, family, paraphrase_seed,
 *         a6_cap_applied, fabrication_count,
 *         axis_scores: {A1..A6: float},
 *         per_turn_rubric: { criteria: [...], turns: [{session, turn, scores: {crit: 0-5}}] },
 *         judge_notes: { A1..A6: "..."},
 *         callback_ledger: [{ claim, claimed_when, matched, evidence_session, evidence_turn, evidence_text, fabricated }],
 *         sessions: [{ session_index, session_id, inter_session_gap_days,
 *                      turns: [{ user_text, assistant_text, fsm_action, fsm_payload, ... }] }]
 *       }, ...
 *     ]
 *   }
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

  const FAMILY_LABELS = {
    F1: "Continuity", F2: "Repair", F3: "Personalization",
    F4: "Long absence", F5: "Boundary pressure", F6: "Goal drift",
  };
  const AXES = ["A1", "A2", "A3", "A4", "A5", "A6"];

  function getSubmissionId() {
    return new URLSearchParams(window.location.search).get("s") || "";
  }

  async function loadDetail(id) {
    const url = `../data/submissions/${encodeURIComponent(id)}.json`;
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`could not load ${url} (${resp.status})`);
    return resp.json();
  }

  function renderHero(detail) {
    document.title = `${detail.system_name} — Companion Bench`;
    $("#hero-name").textContent = detail.system_name;
    const metaParts = [];
    if (detail.model_identifier) metaParts.push(`<code>${escape(detail.model_identifier)}</code>`);
    if (detail.leaderboard_category) metaParts.push(`<span class="category-tag">${escape(detail.leaderboard_category)}</span>`);
    if (detail.generated_at) metaParts.push(escape(detail.generated_at.replace("T", " ").split(".")[0]) + " UTC");
    if (detail.aggregate && detail.aggregate.arc_count) metaParts.push(`${detail.aggregate.arc_count} arcs`);
    $("#hero-meta").innerHTML = metaParts.join(" · ");

    const agg = detail.aggregate || {};
    $("#hero-score").textContent = fmt(agg.lscb_final, 2);
    if (agg.a6_cap_applied) {
      $("#hero-cap").style.display = "inline-block";
    }
    const aggParts = [];
    if (Number.isFinite(agg.raw) && agg.a6_cap_applied) aggParts.push(`raw geometric mean ${fmt(agg.raw, 2)} (capped to 50 for safety)`);
    if (Number.isFinite(agg.trueskill_conservative)) aggParts.push(`TrueSkill ${fmt(agg.trueskill_conservative, 1)}`);
    if (Number.isFinite(agg.bradley_terry_score)) aggParts.push(`BT ${fmt(agg.bradley_terry_score, 2)}`);
    if (Number.isFinite(agg.a6_cap_fraction)) aggParts.push(`A6 cap fired in ${(agg.a6_cap_fraction * 100).toFixed(0)}% of arcs`);
    $("#hero-aggregation").innerHTML = aggParts.join(" · ");

    const items = AXES.map((a) => {
      const v = agg.axis_means && agg.axis_means[a];
      const ci = agg.axis_ci95 && agg.axis_ci95[a];
      return {
        label: a,
        value: v,
        low: ci ? ci[0] : null,
        high: ci ? ci[1] : null,
        fill: a === "A6" ? (v < 60 ? "var(--bad)" : "var(--good)") : "var(--accent)",
      };
    });
    if (window.CB_CHARTS) {
      $("#hero-axis-bars").innerHTML = CB_CHARTS.axisBars(items, { width: 420, decimals: 1, aria: "axis means" });
    }
  }

  function renderManifest(detail) {
    const dl = $("#manifest-dl");
    const m = detail.manifest || {};
    const rows = [
      ["Submission ID", detail.submission_id],
      ["System name", detail.system_name],
      ["Model identifier", detail.model_identifier],
      ["Endpoint", m.base_url],
      ["Generation config", m.generation_config ? JSON.stringify(m.generation_config) : "—"],
      ["Attestation", m.attestation_summary || "all clauses true"],
    ];
    dl.innerHTML = rows
      .filter(([, v]) => v != null && v !== "")
      .map(([k, v]) => `<dt class="text-muted" style="margin-top:0.4rem;">${escape(k)}</dt><dd style="margin:0 0 0.2rem;">${escape(v)}</dd>`)
      .join("");
    const ver = detail.verifier || {};
    $("#verify-state").textContent = ver.state || "pending";
  }

  function renderAxesDetail(detail) {
    const host = $("#axes-detail");
    const agg = detail.aggregate || {};
    const items = AXES.map((a) => ({
      label: a,
      value: agg.axis_means && agg.axis_means[a],
      low: agg.axis_ci95 && agg.axis_ci95[a] && agg.axis_ci95[a][0],
      high: agg.axis_ci95 && agg.axis_ci95[a] && agg.axis_ci95[a][1],
      fill: a === "A6" && (agg.axis_means && agg.axis_means[a] < 60) ? "var(--bad)" : "var(--accent)",
    }));
    if (!window.CB_CHARTS) return;
    host.innerHTML = `
      <div class="card">${CB_CHARTS.axisBars(items, { width: 720, decimals: 1, aria: "per-axis means" })}</div>
      <p class="meta" style="margin-top:.6rem;">
        Whiskers are bootstrap 95% CI of the mean across arcs.
        A6 colour shifts to red below the 60 cap threshold.
      </p>
    `;
  }

  function renderFamiliesDetail(detail) {
    const host = $("#families-detail");
    const fm = detail.family_means || {};
    const items = Object.keys(fm).sort().map((f) => ({
      label: `${f} ${FAMILY_LABELS[f] || ""}`,
      value: fm[f].mean,
      low: fm[f].ci95 && fm[f].ci95[0],
      high: fm[f].ci95 && fm[f].ci95[1],
    }));
    if (!items.length) {
      host.innerHTML = `<p class="text-muted">No per-family data available.</p>`;
      return;
    }
    if (!window.CB_CHARTS) return;
    host.innerHTML = `
      <div class="card">${CB_CHARTS.axisBars(items, { width: 720, decimals: 1, aria: "per-family means" })}</div>
    `;
  }

  function renderCallbackList(arc) {
    const entries = arc.callback_ledger || [];
    if (!entries.length) {
      return `<p class="meta">No assistant callbacks detected in this arc.</p>`;
    }
    return `
      <ul class="callback-list">
        ${entries.map((e) => `
          <li class="${e.fabricated ? "fab" : ""}">
            <span class="claim">${e.fabricated ? "✗ fabricated:" : "✓ matched:"} "${escape(e.claim)}"</span>
            ${e.evidence_text
              ? `<span class="evidence">→ S${e.evidence_session} T${e.evidence_turn}: "${escape(e.evidence_text)}" (similarity ${fmt(e.similarity_score, 2)})</span>`
              : `<span class="evidence text-bad">no matching prior turn found</span>`}
          </li>
        `).join("")}
      </ul>
    `;
  }

  function renderSessions(arc) {
    const sessions = (arc.sessions || []).map((session) => {
      const turns = (session.turns || []).map((t) => `
        <div class="turn user">
          <div class="turn-label">
            <span>User · S${session.session_index} T${t.turn_index}</span>
            ${t.fsm_action ? `<span class="fsm-marker">${escape(t.fsm_action)}${t.fsm_payload ? ` &middot; ${escape(t.fsm_payload)}` : ""}</span>` : ""}
          </div>
          <div class="turn-text">${escape(t.user_text)}</div>
        </div>
        <div class="turn asst">
          <div class="turn-label"><span>Assistant</span></div>
          <div class="turn-text">${escape(t.assistant_text)}</div>
        </div>
      `).join("");
      return `
        <div class="session">
          <div class="session-header"><span class="toggle"></span>
            <span>Session ${session.session_index} · ${session.turns.length} turns
                  ${session.inter_session_gap_days ? "· +" + session.inter_session_gap_days + " day(s)" : ""}</span>
            <span class="meta">${escape(session.session_id)}</span>
          </div>
          <div class="session-body">${turns}</div>
        </div>
      `;
    }).join("");
    return sessions;
  }

  function renderHeatmap(arc) {
    const rubric = arc.per_turn_rubric;
    if (!rubric || !rubric.criteria || !rubric.turns || !rubric.turns.length) {
      return `<p class="meta">No per-turn rubric available for this arc.</p>`;
    }
    if (!window.CB_CHARTS) return "";
    return CB_CHARTS.rubricHeatmap(rubric.turns, rubric.criteria, { aria: "per-turn rubric" });
  }

  function renderJudgeNotes(arc) {
    const notes = arc.judge_notes || {};
    const entries = AXES.filter((a) => notes[a]).map((a) => `
      <div class="judge-box">
        <h4>${a} — judge notes</h4>
        <p>${escape(notes[a])}</p>
      </div>
    `);
    if (!entries.length) return "";
    return entries.join("");
  }

  function renderArcCard(arc) {
    const axisItems = AXES.map((a) => ({
      label: a,
      value: arc.axis_scores && arc.axis_scores[a],
      fill: a === "A6" && (arc.axis_scores && arc.axis_scores[a] < 60) ? "var(--bad)" : "var(--accent)",
    }));
    return `
      <article class="card" data-family="${escape(arc.family)}"
               data-cap="${arc.a6_cap_applied ? "yes" : "no"}"
               data-fab="${arc.fabrication_count > 0 ? "yes" : "no"}">
        <h3 class="mt-0">
          <span class="family-tag">${escape(arc.family)}</span>
          ${escape(arc.scenario_id)}
          ${arc.a6_cap_applied ? `<span class="cap-tag">A6 cap</span>` : ""}
          ${arc.fabrication_count > 0 ? `<span class="cap-tag">${arc.fabrication_count} fab</span>` : ""}
        </h3>
        <p class="meta"><code>${escape(arc.arc_id)}</code> · paraphrase seed ${arc.paraphrase_seed}</p>
        <div>${window.CB_CHARTS ? CB_CHARTS.axisBars(axisItems, { width: 540, decimals: 1, aria: "arc axis scores" }) : ""}</div>

        <details>
          <summary>Per-turn rubric heatmap</summary>
          <div style="margin-top:.5rem;">${renderHeatmap(arc)}</div>
        </details>

        <details>
          <summary>Callback ledger
            ${arc.fabrication_count > 0 ? `<span class="cap-tag">${arc.fabrication_count} fabricated</span>` : ""}
          </summary>
          <div style="margin-top:.5rem;">${renderCallbackList(arc)}</div>
        </details>

        <details>
          <summary>Multi-session transcript</summary>
          <div style="margin-top:.5rem;">${renderSessions(arc)}</div>
        </details>

        ${renderJudgeNotes(arc) ? `<details>
          <summary>Judge notes</summary>
          <div style="margin-top:.5rem;">${renderJudgeNotes(arc)}</div>
        </details>` : ""}
      </article>
    `;
  }

  function renderArcs(detail) {
    const host = $("#scenarios-detail");
    const arcs = (detail.arcs || []).slice().sort((a, b) => a.scenario_id.localeCompare(b.scenario_id));
    host.innerHTML = `<div class="card-grid">${arcs.map(renderArcCard).join("")}</div>`;
    const familyFilter = $("#scenario-family-filter");
    const issueFilter = $("#scenario-issue-filter");
    const apply = () => {
      const fam = familyFilter.value;
      const iss = issueFilter.value;
      $$("#scenarios-detail .card").forEach((card) => {
        const okFam = fam === "all" || card.dataset.family === fam;
        const okIss = iss === "all"
          || (iss === "cap" && card.dataset.cap === "yes")
          || (iss === "fab" && card.dataset.fab === "yes");
        card.style.display = (okFam && okIss) ? "" : "none";
      });
    };
    familyFilter.addEventListener("change", apply);
    issueFilter.addEventListener("change", apply);
    document.querySelectorAll("#scenarios-detail .session-header").forEach((h) =>
      h.addEventListener("click", () => h.parentElement.classList.toggle("open"))
    );
  }

  function renderCost(detail) {
    const host = $("#cost-detail");
    const c = detail.cost;
    if (!c) {
      host.innerHTML = `<p class="meta">No cost telemetry recorded for this submission.</p>`;
      return;
    }
    const totals = c.totals || {};
    const phaseRows = (c.by_phase || []).map((p) => `
      <tr>
        <td>${escape(p.phase)}</td>
        <td>${escape(p.model || "—")}</td>
        <td class="numeric">${(p.prompt_tokens || 0).toLocaleString()}</td>
        <td class="numeric">${(p.completion_tokens || 0).toLocaleString()}</td>
        <td class="numeric">${p.usd != null ? "$" + p.usd.toFixed(2) : "—"}</td>
      </tr>
    `).join("");
    const missing = (c.missing_models || []).length
      ? `<p class="meta">No price entry for: ${c.missing_models.map(escape).join(", ")}. These are reported as <em>—</em> rather than billed at $0.</p>`
      : "";
    host.innerHTML = `
      <p class="meta">
        Total: <strong>${totals.total_usd != null ? "$" + totals.total_usd.toFixed(2) : "—"}</strong>
        · SUT: ${totals.sut_usd != null ? "$" + totals.sut_usd.toFixed(2) : "—"}
        · Per-turn judge: ${totals.perturn_usd != null ? "$" + totals.perturn_usd.toFixed(2) : "—"}
        · Arc judge: ${totals.arc_usd != null ? "$" + totals.arc_usd.toFixed(2) : "—"}
      </p>
      <div class="table-wrap"><table>
        <thead><tr><th>Phase</th><th>Model</th><th class="numeric">Prompt tok</th><th class="numeric">Completion tok</th><th class="numeric">USD</th></tr></thead>
        <tbody>${phaseRows}</tbody>
      </table></div>
      ${missing}
    `;
  }

  function bindTabs() {
    const buttons = $$(".tabs button");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.tab;
        buttons.forEach((b) => b.classList.toggle("active", b === btn));
        $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${id}`));
      });
    });
  }

  async function init() {
    const id = getSubmissionId();
    const loading = $("#loading");
    const root = $("#detail-root");
    if (!id) {
      loading.innerHTML = `<p class="text-bad">No submission specified. Use the URL pattern
        <code>/results/?s=&lt;submission_id&gt;</code>, or pick one from the
        <a href="../leaderboard.html">leaderboard</a>.</p>`;
      return;
    }
    let detail;
    try {
      detail = await loadDetail(id);
    } catch (e) {
      loading.innerHTML = `<p class="text-bad">Could not load submission <code>${escape(id)}</code>.
        ${escape(String(e.message || e))}.</p>
        <p class="meta">Detail data is built by
          <code>scripts/companion_bench/build_site.py</code>.</p>`;
      return;
    }
    loading.hidden = true;
    root.hidden = false;
    bindTabs();
    renderHero(detail);
    renderManifest(detail);
    renderAxesDetail(detail);
    renderFamiliesDetail(detail);
    renderArcs(detail);
    renderCost(detail);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
