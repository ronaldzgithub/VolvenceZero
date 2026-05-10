// LSCB leaderboard — vanilla JS, no build step.

const DATA_URL = "data/aggregate_results.json";

const $ = (selector) => document.querySelector(selector);

function fmtAxis(v) {
  return Number.isFinite(v) ? v.toFixed(1) : "—";
}

function fmtLscb(v) {
  return Number.isFinite(v) ? v.toFixed(2) : "—";
}

function categoryLabel(cat) {
  return {
    "open-weight": "Open weight",
    "closed-api": "Closed API",
    "bespoke": "Bespoke",
  }[cat] || cat;
}

function applyFilters(systems) {
  const cat = $("#category-filter").value;
  const sortKey = $("#sort-by").value;
  let filtered = systems.slice();
  if (cat !== "all") filtered = filtered.filter((s) => s.leaderboard_category === cat);
  filtered.sort((a, b) => {
    const av = sortValue(a, sortKey);
    const bv = sortValue(b, sortKey);
    if (av === bv) return 0;
    return bv - av;
  });
  return filtered;
}

function sortValue(row, key) {
  switch (key) {
    case "lscb_final": return row.lscb_final;
    case "trueskill": return row.trueskill_conservative;
    case "A3":
    case "A4":
    case "A6":
      return row.axis_means?.[key] ?? 0;
    default:
      return 0;
  }
}

function renderTable(systems) {
  const tbody = $("#leaderboard-table tbody");
  tbody.innerHTML = "";
  systems.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>
        <span class="system-name">${escapeHtml(row.system_name)}</span>
        <span class="system-id">${escapeHtml(row.model_identifier || row.submission_id)}</span>
      </td>
      <td><span class="category-tag">${escapeHtml(categoryLabel(row.leaderboard_category))}</span></td>
      <td class="lscb-cell">${fmtLscb(row.lscb_final)}</td>
      <td>${fmtAxis(row.axis_means?.A1)}</td>
      <td>${fmtAxis(row.axis_means?.A2)}</td>
      <td>${fmtAxis(row.axis_means?.A3)}</td>
      <td>${fmtAxis(row.axis_means?.A4)}</td>
      <td>${fmtAxis(row.axis_means?.A5)}</td>
      <td>${fmtAxis(row.axis_means?.A6)}</td>
      <td>${fmtLscb(row.trueskill_conservative)}</td>
      <td>${fmtLscb(row.bradley_terry_score)}</td>
      <td>${row.a6_cap_applied ? `<span class="cap-tag">A6 cap</span>` : ""}</td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));
}

function renderMeta(payload) {
  const meta = [];
  if (payload.lscb_version) meta.push(`LSCB v${payload.lscb_version}`);
  if (payload.generated_at) meta.push(`generated ${payload.generated_at}`);
  if (Array.isArray(payload.systems)) meta.push(`${payload.systems.length} systems`);
  $("#aggregate-meta").textContent = meta.join(" · ");
  if (payload.lscb_version) $("#lscb-version").textContent = `v${payload.lscb_version}`;
  if (payload.demo) $("#demo-banner").hidden = false;
}

async function init() {
  let payload;
  try {
    const resp = await fetch(DATA_URL, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    payload = await resp.json();
  } catch (e) {
    const tbody = $("#leaderboard-table tbody");
    tbody.innerHTML = `<tr><td colspan="13">Could not load ${DATA_URL}: ${escapeHtml(String(e))}</td></tr>`;
    return;
  }
  renderMeta(payload);
  const systems = Array.isArray(payload.systems) ? payload.systems : [];
  const rerender = () => renderTable(applyFilters(systems));
  $("#category-filter").addEventListener("change", rerender);
  $("#sort-by").addEventListener("change", rerender);
  rerender();
}

init();
