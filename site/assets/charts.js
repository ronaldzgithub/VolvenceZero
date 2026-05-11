/* Companion Bench — minimal inline-SVG chart helpers.
 *
 * No external library. Each helper returns an SVG string, designed to be
 * inserted via .innerHTML in a host element. All charts inherit colour
 * from CSS variables via currentColor / fill="var(--...)" so the cozy
 * theme automatically swaps light and dark.
 */

(function () {
  "use strict";

  function escapeText(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
    }[c]));
  }

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function fmt(n, digits) {
    if (n == null || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits == null ? 1 : digits);
  }

  /** Horizontal bar charts for axis means.
   *  Items: [{label, value (0..100), low?, high?, weight?}]
   */
  function axisBars(items, opts) {
    opts = opts || {};
    const width = opts.width || 420;
    const rowH = 28;
    const labelW = 110;
    const valueW = 50;
    const barW = width - labelW - valueW - 10;
    const height = items.length * rowH + 10;
    const rows = items.map((item, i) => {
      const v = clamp(Number(item.value) || 0, 0, 100);
      const fillW = (v / 100) * barW;
      const ciHigh = item.high != null ? (clamp(item.high, 0, 100) / 100) * barW : null;
      const ciLow = item.low != null ? (clamp(item.low, 0, 100) / 100) * barW : null;
      const y = 5 + i * rowH;
      const cy = y + rowH / 2;
      const ciLine = (ciLow != null && ciHigh != null && ciHigh > ciLow)
        ? `<line x1="${labelW + ciLow}" x2="${labelW + ciHigh}" y1="${cy}" y2="${cy}"
                stroke="var(--text-muted)" stroke-width="1" />
           <line x1="${labelW + ciLow}" x2="${labelW + ciLow}" y1="${cy - 3}" y2="${cy + 3}" stroke="var(--text-muted)" stroke-width="1" />
           <line x1="${labelW + ciHigh}" x2="${labelW + ciHigh}" y1="${cy - 3}" y2="${cy + 3}" stroke="var(--text-muted)" stroke-width="1" />`
        : "";
      return `
        <g>
          <text x="0" y="${cy + 4}" font-size="12" fill="var(--subheader-color)">${escapeText(item.label)}</text>
          <rect x="${labelW}" y="${cy - 4}" width="${barW}" height="8" rx="3"
                fill="var(--bar-track)" />
          <rect x="${labelW}" y="${cy - 4}" width="${fillW}" height="8" rx="3"
                fill="${item.fill || "var(--accent)"}" />
          ${ciLine}
          <text x="${width - 5}" y="${cy + 4}" font-size="12" text-anchor="end"
                fill="var(--header-color)" font-weight="600">${fmt(v, opts.decimals)}</text>
        </g>
      `;
    }).join("");
    return `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}"
                  role="img" aria-label="${escapeText(opts.aria || "axis bars")}">
              ${rows}
            </svg>`;
  }

  /** Forest plot for TrueSkill confidence intervals across systems.
   *  Items: [{label, mu, sigma}]; renders mu as point, mu±3sigma as whiskers.
   */
  function forestPlot(items, opts) {
    opts = opts || {};
    const width = opts.width || 540;
    const rowH = 24;
    const labelW = 200;
    const valueW = 60;
    const barW = width - labelW - valueW - 10;
    const lo = opts.min != null ? opts.min : Math.min(...items.map((it) => (it.mu || 0) - 3 * (it.sigma || 0))) - 1;
    const hi = opts.max != null ? opts.max : Math.max(...items.map((it) => (it.mu || 0) + 3 * (it.sigma || 0))) + 1;
    const span = hi - lo || 1;
    const x = (v) => labelW + ((v - lo) / span) * barW;
    const height = items.length * rowH + 30;
    const rows = items.map((item, i) => {
      const mu = Number(item.mu) || 0;
      const sigma = Number(item.sigma) || 0;
      const cy = 25 + i * rowH;
      return `
        <line x1="${x(mu - 3 * sigma)}" x2="${x(mu + 3 * sigma)}"
              y1="${cy}" y2="${cy}" stroke="var(--text-muted)" stroke-width="1.5" />
        <line x1="${x(mu - 3 * sigma)}" x2="${x(mu - 3 * sigma)}" y1="${cy - 4}" y2="${cy + 4}" stroke="var(--text-muted)" />
        <line x1="${x(mu + 3 * sigma)}" x2="${x(mu + 3 * sigma)}" y1="${cy - 4}" y2="${cy + 4}" stroke="var(--text-muted)" />
        <circle cx="${x(mu)}" cy="${cy}" r="4.5" fill="var(--accent)" />
        <text x="0" y="${cy + 4}" font-size="12" fill="var(--subheader-color)">${escapeText(item.label)}</text>
        <text x="${width - 5}" y="${cy + 4}" font-size="12" text-anchor="end"
              fill="var(--header-color)" font-weight="600">${fmt(mu, 1)}</text>
      `;
    }).join("");
    const ticks = (function () {
      const step = (() => {
        const raw = span / 5;
        const pow = Math.pow(10, Math.floor(Math.log10(raw)));
        const m = raw / pow;
        let nice = 1;
        if (m >= 5) nice = 5;
        else if (m >= 2) nice = 2;
        return nice * pow;
      })();
      const out = [];
      const start = Math.ceil(lo / step) * step;
      for (let v = start; v <= hi; v += step) {
        out.push(`<line x1="${x(v)}" x2="${x(v)}" y1="20" y2="${height - 5}"
                          stroke="var(--border)" stroke-dasharray="2,3" />
                  <text x="${x(v)}" y="14" font-size="10" fill="var(--text-muted)" text-anchor="middle">${fmt(v, 0)}</text>`);
      }
      return out.join("");
    })();
    return `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}"
                  role="img" aria-label="${escapeText(opts.aria || "forest plot")}">
              ${ticks}${rows}
            </svg>`;
  }

  /** Per-turn rubric heatmap. Cells: [{session, turn, scores: {criterion: 0-5}}]
   *  Criteria: array of criterion ids in column order.
   */
  function rubricHeatmap(turns, criteria, opts) {
    opts = opts || {};
    const cellW = opts.cellW || 26;
    const cellH = opts.cellH || 22;
    const labelW = 130;
    const headerH = 42;
    const cols = turns.length;
    const rows = criteria.length;
    const width = labelW + cols * cellW + 10;
    const height = headerH + rows * cellH + 10;
    const colorFor = (v) => {
      if (v == null || Number.isNaN(v)) return "var(--bar-track)";
      const t = clamp(v / 5, 0, 1);
      const r = Math.round(255 * (1 - t) + 90 * t);
      const g = Math.round(220 * (1 - t) + 138 * t);
      const b = Math.round(180 * (1 - t) + 74 * t);
      return `rgb(${r}, ${g}, ${b})`;
    };
    let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}"
                     role="img" aria-label="${escapeText(opts.aria || "per-turn rubric heatmap")}"
                     class="heatmap-svg">`;
    // Column labels
    turns.forEach((t, ci) => {
      const x = labelW + ci * cellW + cellW / 2;
      svg += `<text x="${x}" y="14" font-size="9" fill="var(--text-muted)"
                    text-anchor="middle">S${t.session}</text>`;
      svg += `<text x="${x}" y="${headerH - 4}" font-size="9" fill="var(--text-muted)"
                    text-anchor="middle">T${t.turn}</text>`;
    });
    // Row labels + cells
    criteria.forEach((crit, ri) => {
      const y = headerH + ri * cellH;
      svg += `<text x="0" y="${y + cellH / 2 + 4}" font-size="11" fill="var(--subheader-color)">${escapeText(crit.label || crit.id)}</text>`;
      turns.forEach((t, ci) => {
        const v = t.scores ? t.scores[crit.id] : null;
        const cx = labelW + ci * cellW;
        const fill = colorFor(v);
        const tip = `S${t.session} T${t.turn} ${crit.id}: ${v == null ? "—" : v}`;
        svg += `<rect x="${cx + 1}" y="${y + 1}" width="${cellW - 2}" height="${cellH - 2}"
                       fill="${fill}" stroke="var(--bg)" stroke-width="1">
                  <title>${escapeText(tip)}</title>
                </rect>`;
        if (v != null && Number.isFinite(v)) {
          svg += `<text x="${cx + cellW / 2}" y="${y + cellH / 2 + 3}"
                        font-size="9" text-anchor="middle"
                        fill="rgba(0,0,0,0.65)">${v}</text>`;
        }
      });
    });
    svg += "</svg>";
    return svg;
  }

  /** Calibration scatter (judge_score on x, human_score on y).
   *  Items: [{x, y, label?}]; both axes 0..100.
   */
  function calibrationScatter(items, opts) {
    opts = opts || {};
    const width = opts.width || 360;
    const height = opts.height || 320;
    const pad = 38;
    const px = (v) => pad + (clamp(v, 0, 100) / 100) * (width - pad - 10);
    const py = (v) => (height - pad) - (clamp(v, 0, 100) / 100) * (height - pad - 10);
    const ticks = [0, 25, 50, 75, 100];
    let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}"
                     role="img" aria-label="${escapeText(opts.aria || "calibration scatter")}">`;
    // Axes
    svg += `<line x1="${pad}" y1="${pad - 5}" x2="${pad}" y2="${height - pad}" stroke="var(--border)" />`;
    svg += `<line x1="${pad}" y1="${height - pad}" x2="${width - 5}" y2="${height - pad}" stroke="var(--border)" />`;
    // Diagonal y=x
    svg += `<line x1="${px(0)}" y1="${py(0)}" x2="${px(100)}" y2="${py(100)}"
                  stroke="var(--text-muted)" stroke-dasharray="3,4" />`;
    ticks.forEach((t) => {
      svg += `<line x1="${px(t)}" y1="${height - pad}" x2="${px(t)}" y2="${height - pad + 4}" stroke="var(--text-muted)" />`;
      svg += `<text x="${px(t)}" y="${height - pad + 16}" font-size="9" text-anchor="middle" fill="var(--text-muted)">${t}</text>`;
      svg += `<line x1="${pad - 4}" y1="${py(t)}" x2="${pad}" y2="${py(t)}" stroke="var(--text-muted)" />`;
      svg += `<text x="${pad - 6}" y="${py(t) + 3}" font-size="9" text-anchor="end" fill="var(--text-muted)">${t}</text>`;
    });
    svg += `<text x="${(width + pad) / 2}" y="${height - 6}" font-size="10" text-anchor="middle" fill="var(--subheader-color)">Judge score</text>`;
    svg += `<text x="12" y="${(height - pad) / 2}" font-size="10" transform="rotate(-90 12 ${(height - pad) / 2})" text-anchor="middle" fill="var(--subheader-color)">Human score</text>`;
    items.forEach((p) => {
      svg += `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="3.5" fill="var(--accent)" fill-opacity="0.7">
                <title>${escapeText(p.label ? p.label + " " : "")}j=${fmt(p.x, 1)} h=${fmt(p.y, 1)}</title>
              </circle>`;
    });
    svg += "</svg>";
    return svg;
  }

  window.CB_CHARTS = { axisBars, forestPlot, rubricHeatmap, calibrationScatter, escapeText, fmt };
})();
