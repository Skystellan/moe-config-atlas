"use strict";

const TIERS = [
  { id: "S", label: "小（≤50B）" },
  { id: "M", label: "中（50–500B）" },
  { id: "L", label: "大（≥500B）" },
];
const FP_SORT_DEFAULT = { key: "vendor", dir: -1 };
const state = {
  q: "", tiers: new Set(), years: new Set(), vendor: "", group: "sparsity", sort: { key: "latest", dir: -1 },
  fpMode: "model", fpGroup: "all", fpSort: { ...FP_SORT_DEFAULT }, compare: [], adoptGroup: "all", techCat: "all", detSec: "gate",
  laySort: "vendor",
};
let DATA = [];
const charts = [];
// The deploy stamps index.html with app.js?v=<run id> (see .github/workflows/pages.yml) and the data files get the
// same stamp, so a fresh page never runs a script or reads data that a browser cached from an earlier deploy
// (GitHub Pages lets browsers reuse files for 10 minutes). Locally there is no stamp and nothing changes.
const VERSION = document.currentScript ? new URL(document.currentScript.src).searchParams.get("v") : null;
const dataUrl = (file) => `data/${file}${VERSION ? `?v=${encodeURIComponent(VERSION)}` : ""}`;

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function fmtParams(b) {
  if (b == null) return "–";
  if (b >= 1000) return `${+(b / 1000).toFixed(2)}T`;
  return `${+b.toFixed(1)}B`;
}
function fmtConfig(m) {
  if (!m.E) return "–";
  return `${m.per_layer ? "≈" : ""}${m.E}/${m.k}${m.s ? `+${m.s}` : ""}`;
}
function fmtCtx(n) {
  if (!n) return "–";
  if (n >= 1 << 20) return `${+(n / (1 << 20)).toFixed(1)}M`;
  return `${Math.round(n / 1024)}K`;
}

/* ---------- leaderboard scores (scripts/bench.py): Epoch's ECI, the Arena text leaderboard, the Artificial
   Analysis Intelligence Index, and "avg", their normalized average ---------- */

const BENCH_KEYS = ["avg", "eci", "arena", "aa"];
let BENCH = {}; // per key: label, name, url, date, n; avg has the boards it averages (sources) instead of a date
const benchFmt = (k, v) => (v == null ? "–" : k === "arena" ? String(Math.round(v)) : v.toFixed(1));
const boardList = (ks) => ks.map((s) => BENCH[s].label).join("、");
// the name key scripts/bench.py matches on: no org, no "(...)", no -Instruct / -it / precision / date tail
function nameKey(s) {
  const parts = s.split("/").pop().toLowerCase().replace(/\(.*?\)/g, " ").split(/[\s_-]+/).filter(Boolean);
  while (parts.length > 1 && /^(instruct|chat|it|bf16|fp8|nvfp4|base|pt|hf|\d{4,8})$/.test(parts[parts.length - 1])) parts.pop();
  return parts.join("");
}
// the version a score came from when it isn't the row's own model, e.g. GLM-5.3 on the GLM-5 row;
// for the average, the boards it is missing, e.g. "仅 Arena"
function benchFrom(m, k) {
  if (k === "avg") {
    const src = BENCH.avg?.sources ?? [], on = src.filter((s) => m[s] != null);
    if (m.avg == null || on.length === src.length) return null;
    return on.length === 1 ? `仅 ${BENCH[on[0]].label}` : `缺 ${src.filter((s) => m[s] == null).map((s) => BENCH[s].label).join("、")}`;
  }
  const as = m[`${k}_as`];
  if (!as || [m.name, m.repo].some((n) => nameKey(n) === nameKey(as))) return null;
  return as.replace(/\s*\(.*?\)/g, "").trim();
}
function benchTip(m, k) {
  const b = BENCH[k] || {};
  if (k === "avg") {
    if (m.avg == null) return `${m.name}：${boardList(b.sources)} 上都没有这个配置的任何版本`;
    const on = b.sources.filter((s) => m[s] != null), off = b.sources.filter((s) => m[s] == null);
    const parts = on.map((s) => `${BENCH[s].label} ${benchFmt(s, m[s])}`).join("、");
    return `${m.name} · 归一均分：${benchFmt(k, m.avg)}（站内最高 100，最低 0）\n` +
      (on.length > 1 ? `由 ${parts} 换成标准分后平均` : `只有 ${parts} 一个榜，按它的标准分折算`) + (off.length ? `；${boardList(off)} 没有分数` : "");
  }
  if (m[k] == null) return `${m.name}：${b.name ?? k} 上没有这个配置的任何版本`;
  const ci = m[`${k}_ci`];
  return `${m.name} · ${b.name}：${benchFmt(k, m[k])}${ci ? `（区间 ${benchFmt(k, ci[0])}–${benchFmt(k, ci[1])}）` : ""}\n` +
    `取自 ${m[`${k}_as`]}${k === "arena" && m.arena_votes ? `，${m.arena_votes.toLocaleString("en-US")} 票` : ""}${b.date ? ` · 榜单日期 ${b.date}` : ""}`;
}
// best first; configurations no leaderboard has scored go last, newest first
const benchOrder = (k, dir = -1) => (a, b) => {
  const x = a[k], y = b[k];
  if (x == null || y == null) return (x == null) - (y == null) || b.released.localeCompare(a.released);
  return (x - y) * dir || b.released.localeCompare(a.released);
};
const benchSortTitle = (k) => (!BENCH[k] ? "" : k === "avg"
  ? `归一均分：${boardList(BENCH.avg.sources)} 各自换成标准分后取平均，${BENCH.avg.n} 个配置至少有一个榜的分数，没有分数的排在最后`
  : `${BENCH[k].name}：${BENCH[k].n} 个配置有分数，没有分数的排在最后（榜单日期 ${BENCH[k].date}）`);
// the legend line under a view sorted by a leaderboard
const benchNote = (k) => (k === "avg"
  ? `归一均分：${boardList(BENCH.avg.sources)} 各自换成标准分后取平均，站内最高 100、最低 0；${BENCH.avg.n} 个配置有分数，小字是缺了哪个榜`
  : `${BENCH[k].name}：${BENCH[k].n} 个配置有分数，同一配置的多个版本取最高分，小字是取分的版本`);
function median(xs) {
  const v = xs.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return null;
  const h = Math.floor(v.length / 2);
  return v.length % 2 ? v[h] : (v[h - 1] + v[h]) / 2;
}
const num = (x, d = 1) => (x == null ? "–" : (+x.toFixed(d)).toString());
const tierColor = (t) => css(`--tier-${t}`);

function filtered() {
  const q = state.q.toLowerCase();
  return DATA.filter((m) =>
    (!q || `${m.name} ${m.vendor} ${m.repo} ${(m.variants || []).join(" ")}`.toLowerCase().includes(q)) &&
    (!state.tiers.size || state.tiers.has(m.tier)) &&
    (!state.years.size || state.years.has(m.year)) &&
    (!state.vendor || m.vendor === state.vendor));
}

/* ---------- controls ---------- */

function chip(label, pressed, onClick, dot) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.setAttribute("aria-pressed", pressed);
  b.innerHTML = (dot ? `<span class="dot" style="background:${dot}"></span>` : "") + esc(label);
  b.onclick = onClick;
  return b;
}

function renderControls() {
  const tc = $("#tier-chips");
  tc.innerHTML = "";
  for (const t of TIERS) {
    tc.append(chip(t.label, state.tiers.has(t.id), () => toggle(state.tiers, t.id), tierColor(t.id)));
  }
  const yc = $("#year-chips");
  yc.innerHTML = "";
  for (const y of [...new Set(DATA.map((m) => m.year))].sort()) {
    yc.append(chip(String(y), state.years.has(y), () => toggle(state.years, y)));
  }
}

function toggle(set, v) {
  set.has(v) ? set.delete(v) : set.add(v);
  update();
}

/* ---------- tiles ---------- */

function renderTiles(rows) {
  const groups = [{ id: null, label: "全部", rows }].concat(
    TIERS.map((t) => ({ id: t.id, label: t.label, rows: rows.filter((m) => m.tier === t.id) })));
  $("#tiles").innerHTML = groups.map((g) => `
    <div class="tile">
      <h3>${g.id ? `<span class="tier-dot" style="background:${tierColor(g.id)}"></span>` : ""}${esc(g.label)}</h3>
      <div class="big">${g.rows.length}<span class="muted" style="font-size:12px;font-weight:400"> 种配置</span></div>
      <div class="row"><span>E/k 中位数</span><b>${num(median(g.rows.map((m) => m.E_over_k)))}</b></div>
      <div class="row"><span>专家激活比 A</span><b>${num(median(g.rows.map((m) => m.expert_activation)))}%</b></div>
      <div class="row"><span>参数激活比</span><b>${num(median(g.rows.map((m) => m.param_activation)))}%</b></div>
      <div class="row"><span>粒度 G</span><b>${num(median(g.rows.map((m) => m.granularity)))}</b></div>
    </div>`).join("");
}

/* ---------- charts ---------- */

function baseAxis(extra) {
  return Object.assign({
    axisLine: { lineStyle: { color: css("--axis") } },
    axisTick: { show: false },
    axisLabel: { color: css("--muted"), fontSize: 11 },
    splitLine: { lineStyle: { color: css("--grid") } },
    nameTextStyle: { color: css("--muted"), fontSize: 11 },
  }, extra);
}

// Tooltips live on <body> so a scrolling or narrow chart container can't clip them,
// and are placed next to the pointer but kept inside the viewport.
const pointer = { x: null, y: null };
addEventListener("pointermove", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; }, true);
addEventListener("pointerdown", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; }, true);
function tipPosition(point, params, dom, rect, size) {
  const [w, h] = size.contentSize;
  if (pointer.x == null) return [point[0] + 14, point[1] + 14];
  const ox = pointer.x - point[0], oy = pointer.y - point[1]; // chart origin in the viewport
  let x = pointer.x + 14, y = pointer.y + 14;
  if (x + w > innerWidth - 8) x = pointer.x - 14 - w;
  x = Math.max(8, Math.min(x, innerWidth - 8 - w));
  if (y + h > innerHeight - 8) y = pointer.y - 14 - h;
  y = Math.max(8, y);
  return [x - ox, y - oy];
}
function tipStyle() {
  return {
    trigger: "item",
    appendTo: "body",
    position: tipPosition,
    backgroundColor: css("--surface"),
    borderColor: css("--ring"),
    textStyle: { color: css("--ink"), fontSize: 12 },
    extraCssText: "max-width: min(420px, calc(100vw - 32px)); white-space: normal;",
  };
}

function tooltip() {
  return {
    ...tipStyle(),
    formatter: (p) => {
      const m = p.data.m;
      if (!m) return "";
      return `<b>${esc(m.name)}</b> <span style="color:${css("--muted")}">${esc(m.vendor)} · ${m.released}</span><br>` +
        `总 / 激活：${fmtParams(m.total)} / ${fmtParams(m.active)}（${num(m.param_activation)}%）<br>` +
        (m.latest_name !== m.repo.split("/").pop() ? `最近版本：${esc(m.latest_name)}（${m.latest ?? "–"}）<br>` : "") +
        `配置：${fmtConfig(m)}，E/k = ${num(m.E_over_k)}，A = ${num(m.expert_activation)}%` +
        (m.granularity ? `，G = ${num(m.granularity)}` : "") +
        (m.aspect ? `<br>d/L = ${num(m.aspect)}，全局注意力 ${num(m.global_attn_frac)}%` +
          (m.kv_kib_per_token ? `，KV ${num(m.kv_kib_per_token)} KiB/token` : "") + `，上下文 ${fmtCtx(m.context)}` : "") +
        (m.heads && m.qk_dim ? `<br>注意力：${m.heads} 头 × ${m.qk_dim} 维 = ${num(m.attn_ratio, 2)}×d，` +
          `Q/KV ${m.mla ? "MLA" : num(m.gqa)}，RoPE ${ropeLabel(m)}` : "");
    },
  };
}

function chart(id, list = charts) {
  const el = document.getElementById(id);
  let c = echarts.getInstanceByDom(el);
  if (!c) { c = echarts.init(el, null, { renderer: "svg" }); list.push(c); }
  return c;
}

const releaseTime = (m) => Date.parse(m.released + "-01");
const timeAxis = (extra) => baseAxis(Object.assign({
  // a month of headroom so the last quarter's label isn't cut off at the right edge
  type: "time", min: Date.UTC(2025, 0, 1), max: (v) => v.max + 31 * 864e5,
  axisLabel: { color: css("--muted"), fontSize: 11, formatter: (t) => { const d = new Date(t); return `${d.getUTCFullYear()}.${d.getUTCMonth() + 1}`; } },
}, extra));

const grid = { left: 48, right: 16, top: 16, bottom: 40 };
const sizeAxis = (name) => baseAxis({
  type: "log", name, nameLocation: "middle", nameGap: 26, min: 5, max: 4000,
  axisLabel: { color: css("--muted"), fontSize: 11, formatter: (v) => fmtParams(v) },
});

const picked = (m) => state.compare.includes(m.name);
const halo = () => ({ textBorderColor: css("--surface"), textBorderWidth: 3 }); // keeps a name readable over points and lines

// The configs at the two ends of the y range are named on the chart as landmarks, and the configs picked for
// the side-by-side comparison are drawn bigger and named in bold (a bigger symbol also wins label collisions).
function scatterSeries(rows, xf, yf, c) {
  const ok = rows.filter((m) => xf(m) != null && yf(m) != null);
  const byY = ok.slice().sort((a, b) => yf(a) - yf(b));
  const n = byY.length;
  const ends = new Set(); // only an end no other config ties with, else the name would be an arbitrary pick
  if (n > 2 && yf(byY[0]) < yf(byY[1])) ends.add(byY[0].name);
  if (n > 2 && yf(byY[n - 1]) > yf(byY[n - 2])) ends.add(byY[n - 1].name);
  const W = c.getWidth();
  // names go right of the point, or left of it where they would run off the plot
  const labelLayout = (p) => (p.labelRect.x + p.labelRect.width > W - 4 ? { hideOverlap: true, x: p.rect.x - 3, align: "right" } : { hideOverlap: true });
  return TIERS.map((t) => ({
    name: t.id,
    type: "scatter",
    symbolSize: 9,
    z: 3,
    labelLayout,
    data: ok.filter((m) => m.tier === t.id).map((m) => {
      const pick = picked(m);
      const d = {
        value: [xf(m), yf(m)], m, symbolSize: pick ? 13 : 9,
        itemStyle: m.year >= 2026
          ? { color: tierColor(t.id), borderColor: css("--surface"), borderWidth: 1.5 }
          : { color: css("--surface"), borderColor: tierColor(t.id), borderWidth: pick ? 2.5 : 1.8 },
      };
      if (pick || ends.has(m.name)) {
        d.label = { show: true, formatter: m.name, position: "right", distance: 4, fontSize: 11,
          color: css(pick ? "--ink" : "--ink-2"), fontWeight: pick ? 600 : 400, ...halo() };
      }
      return d;
    }),
    emphasis: {
      focus: "self",
      scale: 1.4,
      label: { show: true, formatter: (p) => p.data.m.name, position: "top", color: css("--ink"), fontSize: 11, fontWeight: 600, ...halo() },
    },
    blur: { itemStyle: { opacity: 0.18 }, label: { show: false } },
  }));
}

// Hovering a config in one chart lights it up in all of them; clicking adds it to the comparison.
const chartIndex = new WeakMap(); // chart -> Map(model name -> [seriesIndex, dataIndex])
let linked = null;
function linkHover(name) {
  if (linked === name) return;
  linked = name;
  for (const c of charts) {
    c.dispatchAction({ type: "downplay" });
    const at = name && chartIndex.get(c)?.get(name);
    if (at) c.dispatchAction({ type: "highlight", seriesIndex: at[0], dataIndex: at[1] });
  }
}
function scatterChart(id) {
  const fresh = !echarts.getInstanceByDom(document.getElementById(id));
  const c = chart(id);
  if (fresh) {
    c.on("mouseover", { seriesType: "scatter" }, (p) => linkHover(p.data.m.name));
    c.on("mouseout", { seriesType: "scatter" }, () => linkHover(null));
    c.on("globalout", () => linkHover(null));
    c.on("click", { seriesType: "scatter" }, (p) => togglePick(p.data.m.name));
  }
  return c;
}
function plot(id, build) {
  const c = scatterChart(id);
  const opt = build(c);
  c.setOption(opt, true);
  const map = new Map();
  opt.series.forEach((s, si) => (s.data || []).forEach((d, di) => { if (d && d.m) map.set(d.m.name, [si, di]); }));
  chartIndex.set(c, map);
  linked = null;
}
function chartTake(id, html) {
  const cap = document.getElementById(id).closest("figure").querySelector("figcaption");
  let t = cap.querySelector(".take");
  if (!t) { t = document.createElement("span"); t.className = "take"; cap.append(t); }
  t.innerHTML = html;
}

// Trend lines. Against size: least squares of y (or log2 y) on log10 x, a straight line on these log axes.
function logFit(rows, xf, yf, ylog) {
  const pts = rows.map((m) => [xf(m), yf(m)]).filter(([x, y]) => x > 0 && y != null && (!ylog || y > 0));
  if (pts.length < 8) return null;
  const X = pts.map(([x]) => Math.log10(x)), Y = pts.map(([, y]) => (ylog ? Math.log2(y) : y));
  const n = X.length, mx = X.reduce((a, b) => a + b) / n, my = Y.reduce((a, b) => a + b) / n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxx += (X[i] - mx) ** 2; sxy += (X[i] - mx) * (Y[i] - my); syy += (Y[i] - my) ** 2; }
  if (!sxx || !syy) return null;
  const b = sxy / sxx, a = my - b * mx;
  const at = (lx) => (ylog ? 2 ** (a + b * lx) : a + b * lx);
  const lo = Math.min(...X), hi = Math.max(...X);
  return { b, r2: (sxy * sxy) / (sxx * syy), n, line: [[10 ** lo, at(lo)], [10 ** hi, at(hi)]] };
}
// Against time: a moving average over the seven months around each month (geometric on log axes). Medians
// jump between the clusters these charts have (100% or 25% global layers, 1x or 2x width), averages don't.
function movingMean(rows, yf, ylog) {
  const pts = rows.map((m) => [mon(m.released), yf(m)]).filter(([, y]) => y != null && (!ylog || y > 0));
  if (pts.length < 8) return [];
  const lo = Math.min(...pts.map(([t]) => t)), hi = Math.max(...pts.map(([t]) => t));
  const out = [];
  for (let t = lo; t <= hi; t++) {
    const ys = pts.filter(([u]) => Math.abs(u - t) <= 3).map(([, y]) => (ylog ? Math.log2(y) : y));
    if (ys.length < 6) continue;
    const mean = ys.reduce((a, b) => a + b) / ys.length;
    out.push([Date.UTC(Math.floor(t / 12), t % 12, 1), ylog ? 2 ** mean : mean]);
  }
  return out;
}
const trendLine = (data) => ({
  type: "line", silent: true, symbol: "none", z: 2, data, smooth: 0.3,
  lineStyle: { color: css("--ink-2"), width: 1.5, type: [6, 4], opacity: 0.8 },
});
const spaced = (s) => (/[A-Za-z0-9)]$/.test(s) ? `${s} ` : s); // a space between Latin and Chinese
// below this the size explains too little of the spread to draw or quote a line
const WEAK_FIT = 0.1;
const fitText = (fit, what, { ylog, unit = "", xname = "总参数" } = {}) => {
  if (!fit) return "";
  const stats = `<span class="muted">（对数回归，${fit.n} 个配置，R² = ${fit.r2.toFixed(2)}）</span>`;
  if (fit.r2 < WEAK_FIT) return `${spaced(what)}和${xname}基本无关${stats}`;
  return `${xname}每大 10 倍，${spaced(what)}平均${ylog ? `变成 <b>${num(2 ** fit.b, 2)}</b> 倍` : `${fit.b < 0 ? "低" : "高"} <b>${num(Math.abs(fit.b))}</b>${unit}`}${stats}`;
};
const fitLine = (fit) => (fit && fit.r2 >= WEAK_FIT ? [trendLine(fit.line)] : []);
function yearText(rows, yf, what, fmt) {
  const ys = [...new Set(rows.map((m) => m.year))].sort()
    .map((y) => [y, median(rows.filter((m) => m.year === y).map(yf))]).filter(([, v]) => v != null);
  return ys.length ? `${what}的中位数：${ys.map(([y, v]) => `${y} 年 <b>${fmt(v)}</b>`).join("，")}` : "";
}

function renderCharts(rows) {
  plot("c-param", (c) => {
    const fit = logFit(rows, (m) => m.total, (m) => m.param_activation);
    chartTake("c-param", fitText(fit, "激活比", { unit: " 个百分点" }));
    return {
      grid, tooltip: tooltip(),
      xAxis: sizeAxis("总参数"),
      yAxis: baseAxis({ type: "value", min: 0, axisLabel: { color: css("--muted"), fontSize: 11, formatter: "{value}%" } }),
      series: fitLine(fit).concat(scatterSeries(rows, (m) => m.total, (m) => m.param_activation, c)),
    };
  });

  plot("c-ek", (c) => {
    const fit = logFit(rows, (m) => m.total, (m) => m.E_over_k, true);
    chartTake("c-ek", fitText(fit, "E/k", { ylog: true }));
    return {
      grid, tooltip: tooltip(),
      xAxis: sizeAxis("总参数"),
      yAxis: baseAxis({ type: "log", logBase: 2, min: 4, max: 128 }),
      series: fitLine(fit).concat(scatterSeries(rows, (m) => m.total, (m) => m.E_over_k, c)),
    };
  });

  plot("c-ek2", (c) => {
    const ratios = [8, 16, 32, 64];
    const refLines = ratios.map((r) => ({
      type: "line", silent: true, symbol: "none", z: 1,
      lineStyle: { color: css("--axis"), width: 1, type: "dashed" },
      data: [[2, 2 * r], [Math.min(32, 1024 / r), Math.min(32, 1024 / r) * r]],
      endLabel: { show: true, formatter: `E/k=${r}`, color: css("--muted"), fontSize: 10 },
    }));
    const count = (f) => { const n = new Map(); for (const m of rows) if (f(m) != null) n.set(f(m), (n.get(f(m)) || 0) + 1); return [...n].sort((a, b) => b[1] - a[1])[0]; };
    const [k, nk] = count((m) => m.k) || [], [e, ne] = count((m) => m.E) || [];
    chartTake("c-ek2", k ? `最常见的是 k = <b>${k}</b>（${nk} 个配置）和 E = <b>${e}</b>（${ne} 个）` : "");
    return {
      grid: { ...grid, right: 56 }, tooltip: tooltip(),
      xAxis: baseAxis({ type: "log", logBase: 2, min: 2, max: 32, name: "每 token 选中的路由专家数 k", nameLocation: "middle", nameGap: 26 }),
      yAxis: baseAxis({ type: "log", logBase: 2, min: 16, max: 1024, name: "E", nameGap: 8 }),
      series: refLines.concat(scatterSeries(rows, (m) => m.k, (m) => m.E, c)),
    };
  });

  plot("c-gran", (c) => {
    const fit = logFit(rows, (m) => m.total, (m) => m.granularity);
    chartTake("c-gran", fitText(fit, "G"));
    return {
      grid, tooltip: tooltip(),
      xAxis: sizeAxis("总参数"),
      yAxis: baseAxis({ type: "value", min: 0 }),
      series: fitLine(fit).concat(scatterSeries(rows, (m) => m.total, (m) => m.granularity, c)),
    };
  });

  plot("c-aspect", (c) => {
    const fit = logFit(rows, (m) => m.active, (m) => m.aspect);
    chartTake("c-aspect", fitText(fit, "d/L", { xname: "激活参数" }));
    return {
      grid, tooltip: tooltip(),
      xAxis: baseAxis({ type: "log", name: "激活参数", nameLocation: "middle", nameGap: 26, min: 1, max: 128,
        axisLabel: { color: css("--muted"), fontSize: 11, formatter: (v) => fmtParams(v) } }),
      yAxis: baseAxis({ type: "value", min: 0 }),
      series: fitLine(fit).concat(scatterSeries(rows, (m) => m.active, (m) => m.aspect, c)),
    };
  });

  plot("c-global", (c) => {
    const med = movingMean(rows, (m) => m.global_attn_frac);
    chartTake("c-global", yearText(rows, (m) => m.global_attn_frac, "全局层占比", (v) => `${num(v)}%`));
    return {
      grid, tooltip: tooltip(),
      xAxis: timeAxis(),
      yAxis: baseAxis({ type: "value", min: 0, max: 100, axisLabel: { color: css("--muted"), fontSize: 11, formatter: "{value}%" } }),
      series: (med.length > 1 ? [trendLine(med)] : []).concat(scatterSeries(rows, releaseTime, (m) => m.global_attn_frac, c)),
    };
  });

  plot("c-attn", (c) => {
    const med = movingMean(rows, (m) => m.attn_ratio, true);
    chartTake("c-attn", yearText(rows, (m) => m.attn_ratio, "宽度比", (v) => `${num(v, 2)}×`));
    return {
      grid, tooltip: tooltip(),
      xAxis: timeAxis(),
      yAxis: baseAxis({ type: "log", logBase: 2, min: 0.5, max: 16, axisLabel: { color: css("--muted"), fontSize: 11, formatter: (v) => `${v}×` } }),
      series: (med.length > 1 ? [trendLine(med)] : []).concat(scatterSeries(rows, releaseTime, (m) => m.attn_ratio, c)),
    };
  });

  // Attention width (query heads x per-head QK dims) against d_model; dashed lines are equal ratios.
  plot("c-heads", (c) => {
    const widthRefs = [1, 2, 4, 8].map((r) => ({
      type: "line", silent: true, symbol: "none", z: 1,
      lineStyle: { color: css("--axis"), width: 1, type: "dashed" },
      data: [[1024, 1024 * r], [16384, 16384 * r]],
      endLabel: { show: true, formatter: `${r}×`, color: css("--muted"), fontSize: 10 },
    }));
    const kFmt = (v) => (v >= 1024 ? `${v / 1024}K` : v);
    const rs = rows.filter((m) => m.attn_ratio != null);
    chartTake("c-heads", rs.length ? `头数 × 头维超过 d_model 的（宽度比 > 1×）：<b>${rs.filter((m) => m.attn_ratio > 1.001).length}</b> / ${rs.length}` : "");
    return {
      grid: { ...grid, right: 40 }, tooltip: tooltip(),
      xAxis: baseAxis({ type: "log", logBase: 2, min: 1024, max: 16384, name: "d_model", nameLocation: "middle", nameGap: 26,
        axisLabel: { color: css("--muted"), fontSize: 11, formatter: kFmt } }),
      yAxis: baseAxis({ type: "log", logBase: 2, min: 1024, max: 131072, axisLabel: { color: css("--muted"), fontSize: 11, formatter: kFmt } }),
      series: widthRefs.concat(scatterSeries(rows, (m) => m.d_model, (m) => (m.heads && m.qk_dim ? m.heads * m.qk_dim : null), c)),
    };
  });

  plot("c-kv", (c) => {
    const fit = logFit(rows, (m) => m.total, (m) => m.kv_kib_per_token, true);
    chartTake("c-kv", fitText(fit, "每 token 的 KV", { ylog: true }));
    return {
      grid, tooltip: tooltip(),
      xAxis: sizeAxis("总参数"),
      yAxis: baseAxis({ type: "log", logBase: 2, min: 4, max: 512, axisLabel: { color: css("--muted"), fontSize: 11, formatter: "{value} KiB" } }),
      series: fitLine(fit).concat(scatterSeries(rows, (m) => m.total, (m) => m.kv_kib_per_token, c)),
    };
  });

  plot("c-ctx", (c) => {
    const med = movingMean(rows, (m) => m.context, true);
    chartTake("c-ctx", yearText(rows, (m) => m.context, "最大上下文", fmtCtx));
    return {
      grid, tooltip: tooltip(),
      xAxis: timeAxis(),
      yAxis: baseAxis({ type: "log", logBase: 4, min: 4096, max: 16 * (1 << 20), axisLabel: { color: css("--muted"), fontSize: 11, formatter: fmtCtx } }),
      series: (med.length > 1 ? [trendLine(med)] : []).concat(scatterSeries(rows, releaseTime, (m) => m.context, c)),
    };
  });

  const legend = TIERS.map((t) => `<span><i style="background:${tierColor(t.id)}"></i>${t.label}</span>`).join("") +
    `<span><i style="background:${css("--ink-2")}"></i>实心 2026 年首发</span><span><i class="hollow"></i>空心 2025 年首发</span>` +
    `<span><i class="trend"></i>虚线：趋势（对规模取对数做回归；横轴是时间的，取前后 3 个月的滑动平均）</span>` +
    `<span><i class="big"></i>大点、粗体名字：并排对比里的型号；悬停一个点，其他图里同一个配置也亮起来，点击加入或移出对比</span>`;
  document.querySelectorAll("#legend, .legend-copy").forEach((el) => { el.innerHTML = legend; });
}

// One lane per lab, one dot per config at its release month: size is total parameters (log), colour is the
// attention family, using the layer-strip colours and the same precedence (any linear layer, then sparse, then
// sliding window, else all global).
const FAMILIES = [["linear", "混线性层"], ["sparse", "稀疏注意力"], ["local", "混滑窗"], ["global", "纯全局注意力"]];
const FAMILY_LABEL = Object.fromEntries(FAMILIES);
function attnFamily(m) {
  const cats = new Set([...(m.layout || "")].map(layerCat));
  return FAMILIES.map(([id]) => id).find((c) => cats.has(c)) || null;
}
let vl = null;
function renderVendors(rows) {
  const box = $("#vendors");
  const labs = [...new Set(rows.map((m) => m.vendor))].map((v) => {
    const ms = rows.filter((m) => m.vendor === v).sort((a, b) => a.released.localeCompare(b.released) || b.total - a.total);
    return { v, ms };
  }).sort((a, b) => a.ms[0].released.localeCompare(b.ms[0].released) || b.ms.length - a.ms.length || a.v.localeCompare(b.v));
  if (!labs.length) { vl = null; box.innerHTML = `<p class="muted">当前筛选下没有配置。</p>`; $("#vendor-take").innerHTML = ""; return; }
  const narrow = box.clientWidth < 560;
  const W = box.clientWidth, nameW = narrow ? 76 : 128, padR = 12, LH = 22, top = 20;
  const t0 = mon(DATA.reduce((a, m) => (m.released < a ? m.released : a), "9999")) - 0.5;
  const t1 = mon(DATA.reduce((a, m) => (m.released > a ? m.released : a), "0")) + 1.5;
  const x = (t) => nameW + 8 + (W - padR - nameW - 8) * (t - t0) / (t1 - t0);
  const sizes = DATA.map((m) => Math.log10(m.total));
  const lo = Math.min(...sizes), hi = Math.max(...sizes);
  const rad = (total) => 3.5 + 6.5 * (Math.log10(total) - lo) / (hi - lo);
  const H = top + labs.length * LH + 4;
  let svg = "";
  for (let t = Math.ceil(t0); t < t1; t++) {
    if (t % 3) continue;
    const major = !(t % 12), px = x(t);
    svg += `<line class="vl-g${major ? " major" : ""}" x1="${px}" x2="${px}" y1="${top - 4}" y2="${H}"/>`;
    if (!narrow || !(t % 6)) svg += `<text class="vl-t" x="${px + 3}" y="12">${Math.floor(t / 12)}.${t % 12 + 1}</text>`;
  }
  const dots = [];
  labs.forEach((lab, i) => {
    const cy = top + i * LH + LH / 2;
    svg += `<line class="vl-lane" x1="0" x2="${W}" y1="${top + (i + 1) * LH}" y2="${top + (i + 1) * LH}"/>`;
    svg += `<text class="vl-name" x="${nameW}" y="${cy + 4}" text-anchor="end"><title>${esc(lab.v)}</title>${esc(clip(lab.v, nameW - 16, narrow ? 11 : 12))}<tspan class="vl-n" dx="4">${lab.ms.length}</tspan></text>`;
    const xs = [...new Set(lab.ms.map((m) => x(mon(m.released) + 0.5)))].sort((a, b) => a - b);
    if (xs.length > 1) svg += `<line class="vl-line" x1="${xs[0]}" x2="${xs[xs.length - 1]}" y1="${cy}" y2="${cy}"/>`;
    // bigger first, so a smaller config out the same month sits inside the bigger one
    for (const m of lab.ms.slice().sort((a, b) => b.total - a.total)) {
      const d = { m, cx: x(mon(m.released) + 0.5), cy, r: rad(m.total), lane: i };
      dots.push(d);
      const fam = attnFamily(m);
      svg += `<circle class="vl-dot ${fam ? `k-${fam}` : "unk"}${picked(m) ? " pick" : ""}" cx="${d.cx}" cy="${cy}" r="${d.r}" data-name="${esc(m.name)}"/>`;
    }
  });
  vl = { dots, top, LH };
  const legendSize = [10, 100, 1000].map((b) => `<svg width="${2 * rad(b) + 2}" height="22" aria-hidden="true"><circle class="vl-dot sz" cx="${rad(b) + 1}" cy="11" r="${rad(b)}"/></svg>${fmtParams(b)}`).join(" ");
  $("#vendor-legend").innerHTML = FAMILIES.map(([id, label]) => `<span><i class="vk k-${id}"></i>${label}</span>`).join("") +
    `<span class="vl-sizes">总参数 ${legendSize}</span><span><i class="vk pick"></i>带圈的：并排对比里的型号</span>`;
  box.innerHTML = `<svg class="vl-svg" width="${W}" height="${H}" role="img" aria-label="各厂家每个配置的首发时间、总参数和注意力类型">${svg}</svg>`;
  // takeaway: labs whose latest config uses a different attention family than their first one
  const fam1 = (ms) => attnFamily(ms.slice().sort((a, b) => b.total - a.total)[0]);
  const moved = labs.map((lab) => {
    const first = lab.ms.filter((m) => m.released === lab.ms[0].released);
    const last = lab.ms.filter((m) => m.released === lab.ms[lab.ms.length - 1].released);
    return { v: lab.v, n: lab.ms.length, a: fam1(first), b: fam1(last) };
  }).filter((l) => l.n > 1 && l.a && l.b && l.a !== l.b);
  const years = [...new Set(labs.map((l) => l.ms[0].released.slice(0, 4)))].sort();
  const joined = years.map((y) => `${y} 年 <b>${labs.filter((l) => l.ms[0].released.startsWith(y)).length}</b> 家`).join("，");
  const moves = new Map(); // "a>b" -> labs
  for (const l of moved) moves.set(`${l.a}>${l.b}`, (moves.get(`${l.a}>${l.b}`) || []).concat(l.v));
  const moveText = [...moves.entries()].sort((p, q) => q[1].length - p[1].length).map(([k, vs]) => {
    const [a, b] = k.split(">").map((id) => FAMILY_LABEL[id]);
    return vs.length > 2 ? `<b>${vs.length}</b> 家从${a}换成${b}（${vs.map(esc).join("、")}）` : `${vs.map(esc).join("、")} 从${a}换成${b}`;
  });
  $("#vendor-take").innerHTML = `第一次出现在这里的厂家：${joined}。` +
    (moved.length ? `有不止一个配置的厂家里，最新一个和第一个的注意力类型不同的有 <b>${moved.length}</b> 家：${moveText.join("；")}。` : "");
}

function clip(t, w, size) {
  if (textWidth(t, size) <= w) return t;
  while (t.length > 1 && textWidth(`${t}…`, size) > w) t = t.slice(0, -1);
  return `${t}…`;
}
function syncVendorPicks() {
  document.querySelectorAll("#vendors .vl-dot").forEach((c) => c.classList.toggle("pick", state.compare.includes(c.dataset.name)));
}

function initVendors() {
  const box = $("#vendors");
  const at = (e) => {
    const svg = box.querySelector("svg");
    if (!svg || !vl) return null;
    const r = svg.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const lane = Math.floor((py - vl.top) / vl.LH);
    const near = vl.dots.filter((d) => d.lane === lane).map((d) => ({ d, dist: Math.hypot(d.cx - px, d.cy - py) - d.r }))
      .filter((o) => o.dist <= 4).sort((p, q) => p.dist - q.dist || p.d.r - q.d.r)[0];
    return near ? near.d : null;
  };
  box.addEventListener("pointermove", (e) => {
    const d = at(e);
    box.style.cursor = d ? "pointer" : "";
    if (!d) return hideTip();
    const fam = attnFamily(d.m);
    showTip(tooltip().formatter({ data: { m: d.m } }) + `<br><span class="muted">注意力类型：${fam ? FAMILY_LABEL[fam] : "未知"} · 点击加入或移出并排对比</span>`, e);
  });
  box.addEventListener("pointerleave", hideTip);
  box.addEventListener("click", (e) => { const d = at(e); if (d) togglePick(d.m.name); });
}

/* ---------- layer strips: one row per config, one cell per token-mixer layer ---------- */

// Cell height says how the layer's cache grows with context: global and sparse layers keep every
// token, windowed layers keep a window, linear layers keep a fixed-size state.
const LAYER_CATS = [
  { id: "global", label: "全局注意力", h: 1 },
  { id: "sparse", label: "稀疏注意力", h: 1 },
  { id: "local", label: "滑窗 / 分块", h: 0.55 },
  { id: "linear", label: "线性注意力 / Mamba / 短卷积", h: 0.3 },
];
const LAYER_H = Object.fromEntries(LAYER_CATS.map((c) => [c.id, c.h]));
// the characters of models.json "layout", see LAYOUT_CODE in scripts/hfconfig.py
const LAYER_KIND = {
  A: ["global", "全注意力"], X: ["global", "MLA"], D: ["sparse", "DSA"], S: ["sparse", "稀疏注意力"],
  C: ["sparse", "CSA（压缩后再稀疏）"], H: ["sparse", "HCA（重度压缩）"], W: ["local", "滑窗注意力"], U: ["local", "分块注意力"],
  G: ["linear", "Gated DeltaNet"], K: ["linear", "KDA"], N: ["linear", "Lightning Attention"], B: ["linear", "Mamba-2"],
  V: ["linear", "门控短卷积"], L: ["linear", "线性注意力"],
};
const layerCat = (code) => (LAYER_KIND[code] || ["global"])[0];
function layerName(m, code) {
  const name = (LAYER_KIND[code] || [null, "?"])[1];
  if (code === "D") return `${m.mla ? "MLA + " : ""}${/LSA/.test(m.sparse_attn ?? "") ? "LSA" : "DSA"}（索引器给每个 query 选 top-k）`;
  if (code === "S") return typeof m.sparse_attn === "string" ? `${m.sparse_attn}（稀疏注意力）` : name;
  if (code === "W") return `${name}${m.window ? `，窗口 ${m.window}` : ""}`;
  if (code === "U") return `${name}${m.window ? `，块大小 ${m.window}` : ""}`;
  if (code === "C" || code === "H") return `${name}${m.window ? `，另带窗口 ${m.window} 的滑窗` : ""}`;
  return name;
}
const LAY_SORTS = [["vendor", "按厂家"], ["released", "按时间"], ["global", "按全局层占比"], ["avg", "按均分"], ["eci", "按 ECI"], ["arena", "按 Arena"], ["aa", "按 AA"]];
let layGeom = { cw: 8, gap: 1, H: 13 };

function layerOrder(rows) {
  if (BENCH_KEYS.includes(state.laySort)) return rows.slice().sort(benchOrder(state.laySort));
  if (state.laySort === "released") return rows.slice().sort((a, b) => b.released.localeCompare(a.released) || a.name.localeCompare(b.name));
  if (state.laySort === "global") {
    return rows.slice().sort((a, b) => (a.global_attn_frac ?? 101) - (b.global_attn_frac ?? 101) || b.released.localeCompare(a.released));
  }
  const count = {};
  for (const m of rows) count[m.vendor] = (count[m.vendor] || 0) + 1;
  return rows.slice().sort((a, b) => count[b.vendor] - count[a.vendor] || a.vendor.localeCompare(b.vendor) ||
    a.released.localeCompare(b.released) || a.total - b.total);
}

function renderLayers(rows) {
  const sorts = LAY_SORTS.filter(([id]) => !BENCH_KEYS.includes(id) || BENCH[id]);
  if (!sorts.some(([id]) => id === state.laySort)) state.laySort = "vendor";
  groupChips($("#lay-sort"), state.laySort, sorts.map(([id, label]) => ({ id, label })),
    (id) => { state.laySort = id; renderLayers(filtered()); });
  $("#lay-sort").querySelectorAll("button").forEach((b, i) => { if (BENCH[sorts[i][0]]) b.title = benchSortTitle(sorts[i][0]); });
  const bk = BENCH_KEYS.includes(state.laySort) ? state.laySort : null;
  $("#lay-legend").innerHTML = LAYER_CATS.map((c) => `<span><i class="lk k-${c.id}"></i>${c.label}</span>`).join("") +
    `<span><i class="lk csa"></i><i class="lk hca"></i>压缩注意力（CSA / HCA）</span><span><i class="lk dense"></i>格子下的短线：dense FFN 层</span><span><i class="lk mtp"></i>MTP 层</span>` +
    (bk ? `<span>左边的数是${bk === "avg" ? "归一均分，带 * 的缺了某个榜的分数" : ` ${esc(BENCH[bk].name)}，带 * 的取自同一配置的另一个版本`}（悬停看是哪个）；空出一行以下的没有分数</span>` : "");
  const ms = layerOrder(rows.filter((m) => m.layout));
  const box = $("#lay");
  const hybrid = (m) => [...m.layout].some((c) => ["local", "linear"].includes(layerCat(c)));
  const byYear = [...new Set(ms.map((m) => m.year))].sort().map((y) => {
    const ys = ms.filter((m) => m.year === y);
    return `${y} 年首发的 <b>${ys.filter(hybrid).length}/${ys.length}</b>`;
  });
  $("#lay-take").innerHTML = ms.length ? `混入了线性或滑窗层的配置：${byYear.join("，")}。` : "";
  const missing = rows.length - ms.length;
  $("#lay-note").textContent = missing ? `另有 ${missing} 个配置的 config 未公开，看不出逐层结构。` : "";
  if (!ms.length) { box.innerHTML = `<p class="muted">当前筛选下没有配置。</p>`; return; }

  const narrow = box.clientWidth < 560;
  const label = narrow ? (bk ? 44 + 92 + 12 : 104 + 6) : 76 + 164 + 16;
  const maxL = Math.max(...ms.map((m) => m.layout.length));
  const maxMtp = Math.max(0, ...ms.map((m) => m.mtp || 0));
  const avail = box.clientWidth - 22 - label - 26; // padding, labels, layer count
  const fit = avail / (maxL + (maxMtp ? maxMtp + 1 : 0));
  const cw = fit >= 5 ? Math.min(10, Math.floor(fit)) : Math.max(1.2, Math.floor(fit * 10) / 10); // phones get sub-pixel cells
  const gap = cw >= 5 ? 1 : 0;
  const H = 13;
  layGeom = { cw, gap, H };
  box.classList.toggle("bench", !!bk);
  let prev = null, prevScored = false;
  box.innerHTML = ms.map((m) => {
    const n = m.layout.length;
    let svg = "";
    for (let i = 0; i < n; i++) {
      const code = m.layout[i], cat = layerCat(code), h = Math.round(H * LAYER_H[cat]);
      svg += `<rect class="k-${cat}${code === "C" ? " csa" : code === "H" ? " hca" : ""}" x="${i * cw}" y="${H - h}" width="${cw - gap}" height="${h}"/>`;
    }
    for (const i of m.dense_layers || []) if (i < n) svg += `<rect class="dense" x="${i * cw}" y="${H + 1.5}" width="${cw - gap}" height="2"/>`;
    const x0 = (n + 1) * cw;
    for (let j = 0; j < (m.mtp || 0); j++) {
      svg += `<rect class="mtp" x="${x0 + j * cw + 0.5}" y="${H - 8.5}" width="${Math.max(1, cw - gap - 1)}" height="8"/>`;
    }
    const end = (m.mtp ? x0 + m.mtp * cw : n * cw) + 4;
    svg += `<text x="${end}" y="${H - 1}">${n}</text>`;
    const grouped = state.laySort === "vendor";
    const grp = grouped ? prev != null && prev !== m.vendor : bk ? prevScored && m[bk] == null : false;
    const showVendor = !grouped || prev !== m.vendor;
    prev = m.vendor;
    prevScored = bk && m[bk] != null;
    const on = state.compare.includes(m.name);
    const from = bk && benchFrom(m, bk);
    const left = bk ? `<span class="lv sc" title="${esc(benchTip(m, bk))}">${benchFmt(bk, m[bk])}${from ? "*" : ""}</span>` : `<span class="lv">${showVendor ? esc(m.vendor) : ""}</span>`;
    return `<div class="lay-row${grp ? " grp" : ""}${on ? " picked" : ""}" data-row="${esc(m.name)}">` + left +
      `<button type="button" class="lay-name" data-name="${esc(m.name)}" title="${esc(m.vendor)} · ${m.released} · 点击加入或移出并排对比">${esc(m.name)}</button>` +
      `<svg class="lay-strip" width="${end + 22}" height="${H + 4}" data-name="${esc(m.name)}" role="img" aria-label="${esc(m.name)}：${esc(m.attention ?? "")}">` +
      `${svg}<rect class="cur" hidden/></svg></div>`;
  }).join("");
}

function syncLayerPicks() {
  document.querySelectorAll("#lay .lay-row").forEach((r) => r.classList.toggle("picked", state.compare.includes(r.dataset.row)));
}

// One tooltip element for the hand-drawn views (the ECharts ones bring their own).
let vtip = null;
function showTip(html, e) {
  if (!vtip) { vtip = document.createElement("div"); vtip.className = "vtip"; document.body.append(vtip); }
  vtip.innerHTML = html;
  vtip.hidden = false;
  const r = vtip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - 14 - r.width;
  if (y + r.height > innerHeight - 8) y = e.clientY - 14 - r.height;
  vtip.style.left = `${Math.max(8, x)}px`;
  vtip.style.top = `${Math.max(8, y)}px`;
}
function hideTip() { if (vtip) vtip.hidden = true; }

function layerTip(m, i) {
  const n = m.layout.length;
  const head = `<b>${esc(m.name)}</b> <span class="muted">${esc(m.vendor)} · ${m.released}</span><br>`;
  const foot = `<br><span class="muted">全部 ${n} 层：${esc(m.attention ?? "–")} · 全局层 ${num(m.global_attn_frac)}%</span>`;
  if (i >= n) return `${head}第 ${i - n} 个 MTP 层（多 token 预测，推理时可做投机解码）${foot}`;
  const where = m.layers && n === 2 * m.layers ? `第 ${Math.floor(i / 2) + 1} 层的第 ${i % 2 + 1} 个注意力子层` : `第 ${i + 1} / ${n} 层`;
  const ffn = m.dense_layers ? ` · FFN：${m.dense_layers.includes(i) ? "dense" : "MoE"}` : m.layers > n ? " · MoE 是独立的块，不在条上" : "";
  return `${head}${where}：${esc(layerName(m, m.layout[i]))}${ffn}${foot}`;
}

function initLayers() {
  const box = $("#lay");
  const clear = (keep) => box.querySelectorAll(".lay-strip .cur:not([hidden])").forEach((c) => { if (c.parentNode !== keep) c.setAttribute("hidden", ""); });
  box.addEventListener("pointermove", (e) => {
    const svg = e.target.closest ? e.target.closest(".lay-strip") : null;
    clear(svg);
    const m = svg && DATA.find((d) => d.name === svg.dataset.name);
    if (!m) return hideTip();
    const { cw, gap, H } = layGeom;
    const n = m.layout.length;
    let i = Math.floor((e.clientX - svg.getBoundingClientRect().left) / cw);
    const mtpAt = i - (n + 1);
    const cur = svg.querySelector(".cur");
    if (i >= n && !(m.mtp && mtpAt >= 0 && mtpAt < m.mtp)) { cur.setAttribute("hidden", ""); return hideTip(); }
    const tipIndex = i >= n ? n + 1 + mtpAt : i;
    cur.removeAttribute("hidden");
    cur.setAttribute("x", i * cw - 1);
    cur.setAttribute("y", -1);
    cur.setAttribute("width", Math.max(2, cw - gap + 2));
    cur.setAttribute("height", H + 2);
    showTip(layerTip(m, i >= n ? n + mtpAt + 1 : tipIndex), e);
  });
  box.addEventListener("pointerleave", () => { hideTip(); clear(null); });
  box.addEventListener("click", (e) => {
    const t = e.target.closest ? e.target.closest("[data-name]") : null;
    if (t) togglePick(t.dataset.name);
  });
}

/* ---------- structural features ---------- */

// value(m) is what a cell shows; has(m) is true / false / null (unknown: config not public, or not determinable).
const known = (m, v) => (m.config_ok ? v : null);
const FEATURE_GROUPS = [
  { id: "attn", label: "注意力" },
  { id: "pos", label: "位置编码" },
  { id: "resid", label: "残差与归一化" },
  { id: "moe", label: "MoE 与 FFN" },
  { id: "io", label: "输出与记忆" },
];
const SHORT = { sqrtsoftplus: "√softplus" };
const FEATURES = [
  { key: "linear_attn", group: "attn", label: "线性注意力", th: "线性", title: "GDN、KDA、Lightning、Mamba-2 或门控短卷积层（状态大小固定）" },
  { key: "window", group: "attn", label: "滑窗", title: "滑动窗口注意力层（窗口大小）；Llama 4 是固定分块", value: (m) => windowLabel(m) },
  { key: "sparse_attn", group: "attn", label: "稀疏注意力", th: "稀疏", title: "DSA、LSA、QSA、CSA、块稀疏或压缩注意力（CSA / HCA）" },
  { key: "mla", group: "attn", label: "MLA", title: "多头潜注意力" },
  { key: "attn_gate", group: "attn", label: "输出门", title: "softmax 注意力输出上的门（sigmoid，Laguna 用 softplus；逐头或逐元素）" },
  { key: "attn_sink", group: "attn", label: "Sink", title: "可学习的 attention sink" },
  { key: "qk_norm", group: "attn", label: "QK-norm", title: "对每个头的 Q、K 做 RMSNorm", nullUnknown: true },
  { key: "k_eq_v", group: "attn", label: "K=V", title: "K 和 V 用同一个向量，只存一份" },
  { key: "kv_share", group: "attn", label: "跨层 KV", title: "后面的层直接读前面源层的 KV，自己不再存" },
  { key: "index_share", group: "attn", label: "索引复用", title: "稀疏注意力的 top-k 选择由几层共用（IndexShare、跨层索引）" },
  { key: "attn_conv", group: "attn", label: "QKV 短卷积", th: "QKV 卷积", title: "softmax 注意力的 Q/K/V（或子层输出）先过因果短卷积；值是加卷积的位置" },
  { key: "partial_rope", group: "pos", label: "部分 RoPE", title: "全局注意力层每个头只有这一比例的维度加 RoPE" },
  { key: "nope", group: "pos", label: "NoPE", title: "全局注意力层不加 RoPE，位置信息来自局部层或线性层（Inkling 改用学到的相对位置 logit）" },
  { key: "len_scale", group: "pos", label: "长度缩放", th: "SSMax", title: "按位置的 log 放大 query（Scalable-Softmax / Llama 4 的温度调节；Mistral Large 4 的系数按头学习）" },
  { key: "hyper_conn", group: "resid", label: "超连接", title: "多路残差流（HC / mHC / iHC）" },
  { key: "attn_res", group: "resid", label: "AttnRes", title: "Attention Residuals：用注意力聚合前面各层的输出（块大小）" },
  { key: "res_gate", group: "resid", label: "残差门", title: "子层输出和残差按逐通道的可学习权重相加（门或缩放）" },
  { key: "gated_norm", group: "resid", label: "GatedNorm", th: "门控 norm", title: "RMSNorm 后乘一个输入相关的低秩 sigmoid 门（秩）" },
  { key: "post_norm", group: "resid", label: "后置 norm", title: "子层输出再过一次 norm：三明治（前后都有）、Post-LN，或只在 FFN 后" },
  { key: "parallel", group: "resid", label: "并行块", th: "并行", title: "注意力和 FFN 读同一个输入，输出一起加回残差" },
  { key: "s", group: "moe", label: "共享专家", th: "共享", title: "共享专家数" },
  { key: "shared_gate", group: "moe", label: "共享专家门", th: "共享门", title: "共享专家的输出乘一个 sigmoid 门" },
  { key: "latent", group: "moe", label: "LatentMoE", th: "Latent", title: "专家在低维潜空间里计算（潜空间维度）" },
  { key: "zero_expert", group: "moe", label: "零计算专家", th: "零专家", title: "直接输出输入的空专家，让 token 动态少用算力（个数）" },
  { key: "router", group: "moe", label: "非 softmax 路由", th: "路由", title: "路由打分用 sigmoid 等，而不是 softmax",
    has: (m) => (m.router == null ? null : m.router !== "softmax") },
  { key: "expert_bias", group: "moe", label: "专家偏置", th: "偏置", title: "无辅助损失负载均衡的专家偏置；分位数 = 按分位数更新偏置" },
  { key: "group_routing", group: "moe", label: "分组路由", th: "分组", title: "先选专家组再选专家（组数/选中组数）" },
  { key: "hash_route", group: "moe", label: "哈希路由", th: "哈希", title: "前几层按 token id 哈希分配专家" },
  { key: "ffn_act", group: "moe", label: "非 SwiGLU 激活", th: "激活", title: "FFN 激活：GeGLU、ReLU²、SiTU、gpt-oss 式 SwiGLU 等",
    has: (m) => (m.ffn_act == null ? null : m.ffn_act !== "SwiGLU") },
  { key: "act_clamp", group: "moe", label: "激活限幅", th: "限幅", title: "SwiGLU 输出限幅（上限，括号里是只在部分层）；SiTU 用 tanh 软限幅" },
  { key: "mtp", group: "io", label: "MTP", title: "多 token 预测层数", nullUnknown: true },
  { key: "softcap", group: "io", label: "softcap", title: "输出 logit 的 tanh 软截断上限" },
  { key: "ngram_mem", group: "io", label: "N-gram 记忆", th: "N-gram", title: "哈希 n-gram 查表：Engram、层内注入或嵌入层" },
];
for (const f of FEATURES) {
  f.value ??= (m) => m[f.key];
  f.has ??= (m) => (f.nullUnknown && f.value(m) == null ? null : known(m, Boolean(f.value(m))));
}
const featureText = (f, m) => { const v = f.value(m); return v === true ? "✓" : String(SHORT[v] ?? v); };
function featureCell(f, m) {
  const h = f.has(m);
  if (h == null) return `<span class="muted">?</span>`;
  const v = f.value(m);
  if (!h) return v && v !== true ? `<span class="muted">${esc(featureText(f, m))}</span>` : `<span class="muted">·</span>`;
  return esc(featureText(f, m));
}
const ropeLabel = (m) => (m.rope_frac == null ? "–" : m.rope_frac === 0 ? `NoPE${typeof m.nope === "string" ? `（${m.nope}）` : ""}`
  : m.rope_frac >= 1 ? "全部" : m.partial_rope || num(m.rope_frac, 2));
const windowLabel = (m) => (m.window && /chunked/.test(m.attention ?? "") ? `分块 ${m.window}` : m.window);

// Shape numbers shown next to the features, colored by quintile over all configs.
const SHAPE_COLS = [
  { key: "E_over_k", label: "E/k", title: "路由稀疏度：路由专家数 / 每 token 选中数", fmt: (m) => num(m.E_over_k) },
  { key: "granularity", label: "G", title: "专家粒度 2d/d_expert（LatentMoE 不计）", fmt: (m) => num(m.granularity) },
  { key: "aspect", label: "d/L", title: "宽深比：隐藏维度 / token mixer 层数", fmt: (m) => num(m.aspect) },
  { key: "attn_ratio", label: "宽度比", title: "注意力宽度比：全局层 query 头数 × 每头 QK 维度 / d_model", fmt: (m) => num(m.attn_ratio, 2) },
  { key: "gqa", label: "Q/KV", title: "每个 KV 头对应几个 query 头（全局层）；MLA 不适用", fmt: (m) => (m.mla ? "MLA" : num(m.gqa)) },
  { key: "global_attn_frac", label: "全局层", title: "全局注意力层占 token mixer 层的比例", fmt: (m) => (m.global_attn_frac == null ? "–" : `${num(m.global_attn_frac)}%`) },
  { key: "kv_kib_per_token", label: "KV", title: "长上下文下每 token 的 bf16 KV 缓存（KiB）", fmt: (m) => num(m.kv_kib_per_token) },
];
const bins = {};
function computeBins() {
  for (const key of [...SHAPE_COLS.map((c) => c.key), ...BENCH_KEYS]) {
    const v = DATA.map((m) => m[key]).filter((x) => x != null).sort((a, b) => a - b);
    bins[key] = [0.2, 0.4, 0.6, 0.8].map((q) => v[Math.min(v.length - 1, Math.floor(q * v.length))]);
  }
}
const binOf = (key, x) => (x == null ? -1 : bins[key].filter((t) => x >= t).length);

const quarter = (m) => `${m.released.slice(0, 4)} Q${Math.ceil(+m.released.slice(5, 7) / 3)}`;
function quarters() {
  const last = DATA.map(quarter).sort().pop();
  const out = [];
  for (let y = 2025; ; y++) {
    for (let q = 1; q <= 4; q++) {
      const k = `${y} Q${q}`;
      out.push(k);
      if (k === last) return out;
    }
  }
}
const names = (ms) => ms.map((m) => esc(m.name)).join("、");
const shownFeatures = (g) => (g === "all" || g === "shape" ? FEATURES : FEATURES.filter((f) => f.group === g));
function groupChips(box, current, opts, onPick) {
  box.innerHTML = "";
  for (const g of opts) box.append(chip(g.label, current === g.id, () => onPick(g.id)));
}

// One row per feature, one column per quarter: the share of that quarter's new configs that use it.
// Rows are sorted by how much the share grew from the first year to the last.
let aqCells = [];
function renderAdoption(rows) {
  groupChips($("#adopt-groups"), state.adoptGroup, [{ id: "all", label: "全部" }, ...FEATURE_GROUPS],
    (id) => { state.adoptGroup = id; renderAdoption(filtered()); });
  const box = $("#adoption");
  const qs = quarters();
  const years = [...new Set(qs.map((q) => q.slice(0, 4)))];
  const share = (ms, f) => { const ks = ms.filter((m) => f.has(m) != null); return ks.length ? ks.filter((m) => f.has(m)).length / ks.length : null; };
  const lines = shownFeatures(state.adoptGroup).map((f) => {
    const cells = qs.map((q) => {
      const ms = rows.filter((m) => quarter(m) === q && f.has(m) != null);
      return { f, q, ms, yes: ms.filter((m) => f.has(m)), no: ms.filter((m) => !f.has(m)) };
    });
    const ys = years.map((y) => share(rows.filter((m) => m.released.startsWith(y)), f));
    const delta = ys.length > 1 && ys[0] != null && ys[ys.length - 1] != null ? ys[ys.length - 1] - ys[0] : null;
    return { f, cells, ys, delta };
  }).sort((a, b) => (b.delta ?? -2) - (a.delta ?? -2) || a.f.label.localeCompare(b.f.label));
  aqCells = [];
  const pctText = (x) => (x == null ? "–" : `${Math.round(100 * x)}%`);
  const body = lines.map(({ f, cells, ys, delta }) => {
    const tds = cells.map((c) => {
      const k = aqCells.push(c) - 1;
      if (!c.ms.length) return `<td class="e" data-k="${k}"></td>`;
      const sh = c.yes.length / c.ms.length;
      const b = sh ? Math.min(4, Math.floor(sh * 5 - 1e-9)) + 1 : 0;
      return b ? `<td class="q${b}" data-k="${k}">${c.yes.length}<span>/${c.ms.length}</span></td>` : `<td class="z" data-k="${k}"></td>`;
    }).join("");
    const d = delta == null ? `<td class="d muted">–</td>`
      : `<td class="d" title="${years[0]} 年 ${pctText(ys[0])} → ${years[years.length - 1]} 年 ${pctText(ys[ys.length - 1])}">` +
        `<i class="${delta < 0 ? "dn" : "up"}" style="width:${Math.round(Math.abs(delta) * 44)}px"></i>${delta > 0 ? "+" : delta < 0 ? "−" : ""}${Math.abs(Math.round(100 * delta))}</td>`;
    return `<tr><th class="f" title="${esc(f.title)}">${esc(f.label)}</th>${tds}${d}</tr>`;
  }).join("");
  const head = `<tr><th class="f"></th>${qs.map((q) => `<th>${q.slice(2, 4)}<br>${q.slice(5)}<small>${rows.filter((m) => quarter(m) === q).length}</small></th>`).join("")}<th class="d">${years.length > 1 ? `${years[0].slice(2)}→${years[years.length - 1].slice(2)}` : ""}</th></tr>`;
  box.innerHTML = `<table class="aq">${head}${body}</table>`;
  const ok = lines.filter((l) => l.delta != null);
  const up = ok.filter((l) => l.delta > 0).slice(0, 3), down = ok.filter((l) => l.delta < 0).slice(-2).reverse();
  const fmt = (l) => `${esc(l.f.label)}（${pctText(l.ys[0])} → ${pctText(l.ys[l.ys.length - 1])}）`;
  $("#adopt-take").innerHTML = years.length > 1 && ok.length
    ? `${years[0]} 到 ${years[years.length - 1]} 年涨得最多：${up.map(fmt).join("、") || "无"}${down.length ? `；跌得最多：${down.map(fmt).join("、")}` : ""}。`
    : "";
  $("#adopt-legend").innerHTML = `<span class="muted">当季用了的比例</span>` +
    ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"].map((t, i) => `<span><i class="sw q${i + 1}"></i>${t}</span>`).join("") +
    `<span><i class="sw off"></i>没有用的</span><span><i class="sw na"></i>当季没有新配置</span>` +
    `<span>格子里是“用了的 / 当季新配置”，表头是当季新配置数；最后一列是 ${years[0] ?? ""} 年到 ${years[years.length - 1] ?? ""} 年的占比变化（百分点）</span>`;
}

function initAdoption() {
  const box = $("#adoption");
  box.addEventListener("pointermove", (e) => {
    const td = e.target.closest ? e.target.closest("td[data-k]") : null;
    const c = td && aqCells[+td.dataset.k];
    if (!c) return hideTip();
    if (!c.ms.length) return showTip(`<b>${esc(c.f.label)}</b> · ${c.q}<br>这一季没有新配置`, e);
    showTip(`<b>${esc(c.f.label)}</b> · ${c.q}<br>${c.yes.length}/${c.ms.length}（${Math.round(100 * c.yes.length / c.ms.length)}%）` +
      (c.yes.length ? `<br>用了：${names(c.yes)}` : "") + (c.no.length ? `<br><span class="muted">没用：${names(c.no)}</span>` : ""), e);
  });
  box.addEventListener("pointerleave", hideTip);
}

/* ---------- fingerprint: one row per config, or aggregated by vendor / tier ---------- */

function renderFingerprint(rows) {
  const rerender = () => renderFingerprint(filtered());
  groupChips($("#fp-mode"), state.fpMode, [{ id: "model", label: "逐个模型" }, { id: "vendor", label: "按厂家汇总" }, { id: "tier", label: "按档位汇总" }],
    (id) => { state.fpMode = id; if (id !== "model" && state.fpGroup === "shape") state.fpGroup = "all"; rerender(); });
  const byModel = state.fpMode === "model";
  groupChips($("#fp-groups"), state.fpGroup,
    [{ id: "all", label: "全部" }, ...(byModel ? [{ id: "shape", label: "形状" }] : []), ...FEATURE_GROUPS],
    (id) => { state.fpGroup = id; rerender(); });
  $("#fp-table").hidden = !byModel;
  $("#fp-heat").hidden = byModel;
  $("#fp-sort").hidden = !byModel;
  if (byModel) renderMatrix(rows);
  else renderHeatmap(rows);
}

function sortMatrix(rows, col) {
  const { key, dir } = state.fpSort;
  if (col) {
    const val = col.f ? (m) => { const h = col.f.has(m); return h == null ? null : +h; } : (m) => (col.key === "gqa" && m.mla ? null : m[key]);
    return rows.slice().sort((a, b) => {
      const x = val(a), y = val(b);
      if (x == null) return y == null ? 0 : 1;
      if (y == null) return -1;
      return (x - y) * dir || b.released.localeCompare(a.released);
    });
  }
  if (BENCH_KEYS.includes(key)) return rows.slice().sort(benchOrder(key, dir));
  if (key === "released") return rows.slice().sort((a, b) => b.released.localeCompare(a.released) || b.total - a.total);
  // vendors with more configs first, each vendor oldest to newest
  const count = {};
  for (const m of rows) count[m.vendor] = (count[m.vendor] || 0) + 1;
  return rows.slice().sort((a, b) => count[b.vendor] - count[a.vendor] || a.vendor.localeCompare(b.vendor) ||
    a.released.localeCompare(b.released) || a.total - b.total);
}

const FP_SORTS = [["vendor", "按厂家"], ["released", "按时间"], ["avg", "按均分"], ["eci", "按 ECI"], ["arena", "按 Arena"], ["aa", "按 AA"]];

function renderMatrix(rows) {
  const g = state.fpGroup;
  const shape = g === "all" || g === "shape" ? SHAPE_COLS : [];
  const feats = g === "shape" ? [] : shownFeatures(g);
  // sorted by a leaderboard: its score becomes the first column
  const bench = BENCH_KEYS.includes(state.fpSort.key) && BENCH[state.fpSort.key] ? state.fpSort.key : null;
  const cols = (bench ? [{ key: bench, label: BENCH[bench].label, title: `${BENCH[bench].name}，越高越好`, bench: true, gs: true }] : [])
    .concat(shape.map((c, i) => ({ ...c, num: true, gs: i === 0 }))).concat(feats.map((f, i) =>
      ({ key: f.key, label: f.th ?? f.label, title: f.title, f, gs: i === 0 || feats[i - 1].group !== f.group })));
  const heads = (bench ? [{ label: "榜单", n: 1 }] : []).concat(shape.length ? [{ label: "形状", n: shape.length }] : []).concat(
    FEATURE_GROUPS.map((fg) => ({ label: fg.label, n: feats.filter((f) => f.group === fg.id).length })).filter((x) => x.n));

  // A column sort only lives while its column is on screen: switching to a group without it
  // drops the sort instead of leaving it to come back later.
  const col = cols.find((c) => !c.bench && c.key === state.fpSort.key);
  if (!col && !bench && state.fpSort.key !== "released") state.fpSort = { ...FP_SORT_DEFAULT };
  const sorted = sortMatrix(rows, col);
  const grouped = state.fpSort.key === "vendor";

  const sortBox = $("#fp-sort");
  sortBox.innerHTML = `<span class="muted">排序</span>`;
  for (const [id, label] of FP_SORTS) {
    if (BENCH_KEYS.includes(id) && !BENCH[id]) continue;
    const c = chip(label, state.fpSort.key === id, () => { state.fpSort = { key: id, dir: -1 }; renderMatrix(filtered()); });
    if (BENCH[id]) c.title = benchSortTitle(id);
    sortBox.append(c);
  }
  if (col) {
    const c = chip(`按「${col.label}」${state.fpSort.dir > 0 ? "↑" : "↓"}`, true, () => { state.fpSort = { ...FP_SORT_DEFAULT }; renderMatrix(filtered()); });
    c.classList.add("x");
    c.title = "取消按这一列排序，回到按厂家";
    c.insertAdjacentHTML("beforeend", `<span class="x-mark" aria-hidden="true">×</span>`);
    sortBox.append(c);
  }

  const cell = (c, m) => {
    const gs = c.gs ? " gs" : "";
    if (c.bench) {
      const b = binOf(c.key, m[c.key]), from = benchFrom(m, c.key);
      return `<td class="bench ${b >= 0 ? `q${b + 1}` : "na"}${gs}" title="${esc(benchTip(m, c.key))}">${benchFmt(c.key, m[c.key])}${from ? `<small>${esc(from)}</small>` : ""}</td>`;
    }
    if (c.num) {
      const b = c.key === "gqa" && m.mla ? -1 : binOf(c.key, m[c.key]);
      const text = c.fmt(m);
      return `<td class="${b >= 0 ? `q${b + 1}` : "na"}${gs}" title="${esc(m.name)} · ${esc(c.label)}：${esc(text)}">${esc(text)}</td>`;
    }
    const h = c.f.has(m);
    const tip = `${esc(m.name)} · ${esc(c.f.label)}`;
    if (h == null) return `<td class="na${gs}" title="${tip}：未知（config 未公开或看不出）">?</td>`;
    const v = c.f.value(m);
    if (!h) {
      const shown = v && v !== true ? esc(featureText(c.f, m)) : "·";
      return `<td class="off${gs}" title="${tip}：${v && v !== true ? shown : "没有"}">${shown}</td>`;
    }
    const text = esc(featureText(c.f, m));
    return `<td class="on${gs}" title="${tip}：${text}">${text}</td>`;
  };
  const sortAttr = (c) => (state.fpSort.key === c.key ? (state.fpSort.dir > 0 ? "ascending" : "descending") : "none");
  $("#fp-table").innerHTML = `
    <thead>
      <tr class="gh"><th class="l corner" rowspan="2">型号<span class="muted"> · 点 + 加入对比</span></th>${heads.map((h) => `<th class="gs" colspan="${h.n}">${esc(h.label)}</th>`).join("")}</tr>
      <tr class="ch">${cols.map((c) => `<th class="${c.gs ? "gs" : ""}" data-key="${c.key}" aria-sort="${sortAttr(c)}" title="${esc(c.title)}（点击排序：第一次降序，第二次升序，第三次取消）">${esc(c.label)}</th>`).join("")}</tr>
    </thead>
    <tbody>${sorted.map((m, i) => {
      const first = i > 0 && (grouped ? sorted[i - 1].vendor !== m.vendor : bench ? m[bench] == null && sorted[i - 1][bench] != null : false);
      const on = state.compare.includes(m.name);
      return `<tr class="${first ? "grp" : ""}${on ? " picked" : ""}">
        <th class="l" scope="row"><div class="rh"><button type="button" class="pick" aria-pressed="${on}" data-name="${esc(m.name)}" title="${on ? "移出对比" : "加入下面的并排对比"}">${on ? "✓" : "+"}</button><span><span class="nm"><span class="tier-dot" style="background:${tierColor(m.tier)}"></span>${esc(m.name)}</span><small>${esc(m.vendor)} · ${m.released}</small></span></div></th>
        ${cols.map((c) => cell(c, m)).join("")}</tr>`;
    }).join("")}</tbody>`;
  // descending → ascending → back to the default order
  $("#fp-table").querySelectorAll("th[data-key]").forEach((th) => th.onclick = () => {
    const k = th.dataset.key, cur = state.fpSort;
    state.fpSort = cur.key !== k ? { key: k, dir: -1 } : cur.dir < 0 ? { key: k, dir: 1 } : { ...FP_SORT_DEFAULT };
    renderMatrix(filtered());
  });
  $("#fp-table").querySelectorAll("button.pick").forEach((b) => b.onclick = () => togglePick(b.dataset.name));
  const colored = [bench ? "榜单分数" : "", shape.length ? "形状数值" : ""].filter(Boolean).join("和");
  $("#fp-legend").innerHTML = `<span><i class="sw on"></i>用了（格子里是取值）</span><span><i class="sw off"></i>没用</span><span><i class="sw na"></i>? 未知</span>` +
    (colored ? `<span>${colored}按全部配置的五分位上色：小<span class="ramp">${[1, 2, 3, 4, 5].map((i) => `<i class="sw q${i}"></i>`).join("")}</span>大</span>` : "") +
    (bench ? `<span>${esc(benchNote(bench))}；横线以下没有分数</span>` : "");
}

function renderHeatmap(rows) {
  const feats = shownFeatures(state.fpGroup);
  const groups = state.fpMode === "tier"
    ? TIERS.map((t) => ({ label: t.label, ms: rows.filter((m) => m.tier === t.id) })).filter((g) => g.ms.length)
    : [...new Set(rows.map((m) => m.vendor))].map((v) => ({ label: v, ms: rows.filter((m) => m.vendor === v) }))
      .sort((a, b) => b.ms.length - a.ms.length || a.label.localeCompare(b.label));
  const steps = [1, 2, 3, 4, 5].map((i) => css(`--seq-${i}`));
  const inkOn = [1, 2, 3, 4, 5].map((i) => css(`--seq-ink-${i}`));
  const data = [];
  groups.forEach((g, y) => feats.forEach((f, x) => {
    const ms = g.ms.filter((m) => f.has(m) != null);
    const yes = ms.filter((m) => f.has(m));
    const no = ms.filter((m) => !f.has(m));
    const share = ms.length ? yes.length / ms.length : null;
    const b = share ? Math.min(4, Math.floor(share * 5 - 1e-9)) : -1;
    data.push({
      value: [x, y, share ?? 0], f, g, yes, no, ms,
      itemStyle: { color: b >= 0 ? steps[b] : css("--page"), borderColor: css("--surface"), borderWidth: 2, borderRadius: 3 },
      label: { color: b >= 0 ? inkOn[b] : css("--muted") },
    });
  }));
  const el = $("#c-heat");
  el.style.height = `${groups.length * 28 + 56}px`;
  el.style.minWidth = `${Math.max(560, feats.length * 54 + 140)}px`;
  el.style.maxWidth = `${feats.length * 96 + 180}px`; // a few columns shouldn't stretch into wide bars
  const c = chart("c-heat");
  c.resize();
  c.setOption({
    grid: { left: 8, right: 8, top: 40, bottom: 4, containLabel: true },
    tooltip: {
      ...tipStyle(),
      formatter: (p) => {
        const { f, g, yes, no, ms } = p.data;
        if (!ms.length) return `<b>${esc(g.label)}</b> · ${esc(f.label)}<br>未知（config 未公开或看不出）`;
        return `<b>${esc(g.label)}</b> · ${esc(f.label)}<br>${yes.length}/${ms.length}` +
          (yes.length ? `<br>用了：${yes.map((m) => `${esc(m.name)}${f.value(m) === true ? "" : `（${esc(featureText(f, m))}）`}`).join("、")}` : "") +
          (no.length ? `<br><span style="color:${css("--muted")}">没用：${names(no)}</span>` : "");
      },
    },
    xAxis: { type: "category", position: "top", data: feats.map((f) => f.th ?? f.label), axisLine: { show: false }, axisTick: { show: false },
      splitLine: { show: false }, axisLabel: { color: css("--ink-2"), fontSize: 11, interval: 0, width: 50, overflow: "break", lineHeight: 13 } },
    yAxis: { type: "category", inverse: true, data: groups.map((g) => `${g.label} (${g.ms.length})`), axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: css("--ink-2"), fontSize: 11 } },
    series: [{
      type: "heatmap", data,
      label: { show: true, fontSize: 10, formatter: (p) => (!p.data.ms.length ? "?" : p.data.yes.length ? `${p.data.yes.length}/${p.data.ms.length}` : "") },
      emphasis: { itemStyle: { borderColor: css("--ink"), borderWidth: 1.5 } },
    }],
  }, true);
  $("#fp-legend").innerHTML = `<span class="muted">采用比例</span>` +
    ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"].map((t, i) => `<span><i class="sw" style="background:${steps[i]}"></i>${t}</span>`).join("") +
    `<span><i class="sw off"></i>没用</span><span><i class="sw na"></i>? 未知</span>`;
}

/* ---------- side-by-side comparison ---------- */

const MAX_PICK = 4;
function togglePick(name) {
  const i = state.compare.indexOf(name);
  if (i >= 0) state.compare.splice(i, 1);
  else {
    if (state.compare.length >= MAX_PICK) state.compare.shift();
    state.compare.push(name);
  }
  saveHash();
  if (state.fpMode === "model") renderMatrix(filtered());
  renderCompare();
  syncDetailPicks();
  syncLayerPicks();
  syncVendorPicks();
  renderCharts(filtered());
}
function saveHash() {
  const url = location.pathname + location.search + (state.compare.length ? `#cmp=${state.compare.map(encodeURIComponent).join(",")}` : "");
  try { history.replaceState(null, "", url); } catch (e) {}
}
function loadHash() {
  const m = location.hash.match(/^#cmp=(.+)$/);
  if (!m) return false;
  const want = m[1].split(",").map((s) => { try { return decodeURIComponent(s); } catch (e) { return ""; } });
  state.compare = [...new Set(want)].filter((n) => DATA.some((d) => d.name === n)).slice(0, MAX_PICK);
  return state.compare.length > 0;
}
function defaultPicks() { // the newest large config of each of the four most recent vendors
  const seen = new Set();
  const out = [];
  for (const m of DATA.filter((m) => m.tier === "L" && m.config_ok).sort((a, b) => b.released.localeCompare(a.released))) {
    if (seen.has(m.vendor)) continue;
    seen.add(m.vendor);
    out.push(m.name);
    if (out.length === MAX_PICK) break;
  }
  return out;
}

const pct = (x) => (x == null ? "–" : `${num(x)}%`);
// no bar: a leaderboard score has no zero to scale from
function benchCmp(m, k) {
  const from = benchFrom(m, k);
  return m[k] == null ? "–" : `<span title="${esc(benchTip(m, k))}">${benchFmt(k, m[k])}</span>${from ? ` <span class="muted">（${esc(from)}）</span>` : ""}`;
}
const CMP_SECTIONS = [
  { label: "规模", rows: [
    { label: "厂家 · 首发", f: (m) => `${esc(m.vendor)} · ${m.released}` },
    { label: "总参数", f: (m) => fmtParams(m.total), v: (m) => m.total },
    { label: "激活参数", f: (m) => fmtParams(m.active), v: (m) => m.active },
    { label: "参数激活比", f: (m) => pct(m.param_activation), v: (m) => m.param_activation },
  ] },
  { label: "榜单", rows: [
    { label: "归一均分", need: "avg", title: "下面几个榜各自换成标准分后的平均，站内最高 100、最低 0", f: (m) => benchCmp(m, "avg") },
    { label: "ECI", need: "eci", title: "Epoch Capabilities Index，同一配置的多个版本取最高", f: (m) => benchCmp(m, "eci") },
    { label: "Arena", need: "arena", title: "Arena 文本榜分数，同一配置的多个版本取最高", f: (m) => benchCmp(m, "arena") },
    { label: "AA", need: "aa", title: "Artificial Analysis 智能指数，同一配置的多个版本取最高", f: (m) => benchCmp(m, "aa") },
  ] },
  { label: "稀疏度", rows: [
    { label: "E/k+s", f: fmtConfig },
    { label: "E/k", f: (m) => num(m.E_over_k), v: (m) => m.E_over_k },
    { label: "专家激活比 A", f: (m) => pct(m.expert_activation), v: (m) => m.expert_activation },
    { label: "粒度 G", f: (m) => num(m.granularity), v: (m) => m.granularity },
  ] },
  { label: "形状", rows: [
    { label: "层数（MoE 层）", f: (m) => (m.layers ? `${m.layers}（${m.moe_layers ?? "–"}）` : "–"), v: (m) => m.layers },
    { label: "d_model", f: (m) => m.d_model ?? "–", v: (m) => m.d_model },
    { label: "宽深比 d/L", f: (m) => num(m.aspect), v: (m) => m.aspect },
    { label: "d_expert", f: (m) => m.d_expert ?? "–", v: (m) => m.d_expert },
    { label: "词表", f: (m) => (m.vocab ? `${Math.round(m.vocab / 1000)}K` : "–"), v: (m) => m.vocab },
    { label: "上下文", f: (m) => fmtCtx(m.context), v: (m) => m.context },
  ] },
  { label: "注意力形状", rows: [
    { label: "层构成", f: (m) => esc(m.attention ?? "–") },
    { label: "全局层占比", f: (m) => pct(m.global_attn_frac), v: (m) => m.global_attn_frac },
    { label: "KV / token", f: (m) => (m.kv_kib_per_token == null ? "–" : `${num(m.kv_kib_per_token)} KiB`), v: (m) => m.kv_kib_per_token },
    { label: "Q 头 × QK 维", f: (m) => (m.heads ? `${m.heads} × ${m.qk_dim}` : "–") },
    { label: "注意力宽度 / d", title: "全局层 query 头数 × 每头 QK 维度 / d_model", f: (m) => num(m.attn_ratio, 2), v: (m) => m.attn_ratio },
    { label: "Q/KV", f: (m) => (m.mla ? "MLA" : num(m.gqa)), v: (m) => (m.mla ? null : m.gqa) },
    { label: "RoPE 维度占比", f: ropeLabel },
  ] },
  ...FEATURE_GROUPS.map((g) => ({ label: g.label, rows: FEATURES.filter((f) => f.group === g.id)
    .map((f) => ({ label: f.label, title: f.title, f: (m) => featureCell(f, m) })) })),
];

function renderCompare() {
  const ms = state.compare.map((n) => DATA.find((m) => m.name === n)).filter(Boolean);
  const list = $("#model-list");
  if (!list.options.length) list.innerHTML = DATA.slice().sort((a, b) => a.name.localeCompare(b.name)).map((m) => `<option value="${esc(m.name)}">`).join("");
  const picked = $("#cmp-chips");
  picked.innerHTML = "";
  for (const m of ms) {
    const b = chip(`${m.name} ×`, true, () => togglePick(m.name), tierColor(m.tier));
    b.title = "移出对比";
    picked.append(b);
  }
  $("#fp-picked").innerHTML = ms.length
    ? `已选 ${ms.length}/${MAX_PICK} 个对比：${names(ms)} · <a href="#compare">看并排对比 ↓</a>` : "";
  $("#det-picked").innerHTML = ms.length
    ? `已选 ${ms.length}/${MAX_PICK} 个对比：${names(ms)} · <a href="#compare">看并排对比 ↑</a>` : "";
  const table = $("#cmp");
  if (!ms.length) {
    table.innerHTML = `<tbody><tr><td class="l muted">还没选型号。在上面的结构指纹里点 +，或在输入框里输入型号。</td></tr></tbody>`;
    return;
  }
  const strip = (h) => String(h).replace(/<[^>]*>/g, "");
  const onlyDiff = $("#cmp-diff").checked;
  let body = "";
  for (const sec of CMP_SECTIONS) {
    const rowsHtml = sec.rows.filter((r) => !r.need || BENCH[r.need]).map((r) => {
      const cells = ms.map((m) => r.f(m));
      const same = ms.length > 1 && new Set(cells.map(strip)).size === 1;
      if (onlyDiff && same) return "";
      const vs = r.v ? ms.map((m) => r.v(m)) : null;
      const max = vs ? Math.max(0, ...vs.filter((x) => x != null)) : 0;
      return `<tr class="${same ? "same" : ""}"><th class="l" scope="row"${r.title ? ` title="${esc(r.title)}"` : ""}>${esc(r.label)}</th>${cells.map((c, i) =>
        `<td class="l">${c}${vs && vs[i] != null && max > 0 ? `<span class="bar" style="width:${Math.max(2, 100 * vs[i] / max)}%"></span>` : ""}</td>`).join("")}</tr>`;
    }).join("");
    if (rowsHtml) body += `<tr class="sec"><th class="l" colspan="${ms.length + 1}">${esc(sec.label)}</th></tr>${rowsHtml}`;
  }
  table.innerHTML = `<colgroup><col class="lab">${ms.map(() => "<col>").join("")}</colgroup>` +
    `<thead><tr><th class="l"></th>${ms.map((m) => `<th class="l"><span class="tier-dot" style="background:${tierColor(m.tier)}"></span>${esc(m.name)}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
}

function initCompare() {
  const input = $("#cmp-add");
  const add = () => {
    const m = DATA.find((d) => d.name.toLowerCase() === input.value.trim().toLowerCase());
    if (!m) return;
    if (!state.compare.includes(m.name)) togglePick(m.name);
    input.value = "";
  };
  input.addEventListener("change", add);
  // picking from the datalist; typing a prefix of another name (GLM-4.5 → GLM-4.5-Air) must not add it early
  input.addEventListener("input", (e) => { if (!e.inputType || e.inputType === "insertReplacementText") add(); });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
  $("#cmp-diff").onchange = renderCompare;
  if (!loadHash()) state.compare = defaultPicks();
}

/* ---------- frontier techniques ---------- */

const TECH_CATS = [["all", "全部"], ["attn", "注意力"], ["pos", "位置编码"], ["mixer", "线性与稀疏注意力"],
  ["resid", "残差与归一化"], ["moe", "MoE 与 FFN"], ["io", "输出与记忆"], ["optim", "优化器"]];
const CAT_LABEL = Object.fromEntries(TECH_CATS);
let TECHS = [];

function adopters(t, rows) {
  if (t.models) return { list: t.models.map((x) => ({ m: rows.find((m) => m.name === x.name), cite: x.cite })).filter((x) => x.m), of: null };
  if (!t.feature) return null;
  const f = FEATURES.find((x) => x.key === t.feature) || { has: (m) => known(m, Boolean(m[t.feature])), value: (m) => m[t.feature], label: t.feature };
  const ks = rows.filter((m) => f.has(m) != null);
  const list = ks.filter((m) => f.has(m) && (!t.match || t.match.some((x) => String(f.value(m)).startsWith(x))))
    .sort((a, b) => b.released.localeCompare(a.released))
    // a single matched value (e.g. KDA) would only repeat the technique's name
    .map((m) => ({ m, v: f.value(m) === true || t.match?.length === 1 ? null : featureText(f, m) }));
  return { list, of: ks.length };
}

function renderTechniques(rows) {
  groupChips($("#tech-cats"), state.techCat,
    TECH_CATS.map(([id, label]) => ({ id, label: `${label} ${id === "all" ? TECHS.length : TECHS.filter((t) => t.cat === id).length}` })),
    (id) => { state.techCat = id; renderTechniques(filtered()); });
  const ts = TECHS.filter((t) => state.techCat === "all" || t.cat === state.techCat)
    .sort((a, b) => b.date.localeCompare(a.date) || a.name.localeCompare(b.name));
  renderTimeline(rows, ts);
  const link = (url, text) => `<a href="${esc(url)}" target="_blank" rel="noopener">${text}</a>`;
  $("#tech tbody").innerHTML = ts.map((t) => {
    const a = adopters(t, rows);
    let use;
    if (!a) use = `<span class="muted">config 看不出，没有核对过的采用记录</span>`;
    else if (!a.list.length) use = `<b class="zero">0</b>${a.of != null ? `<span class="muted"> / ${a.of}</span>` : ""}`;
    else {
      const shown = a.list.slice(0, 8).map((x) => `${esc(x.m.name)}${x.v ? `<span class="muted">（${esc(x.v)}）</span>` : ""}${x.cite ? ` ${link(x.cite, "↗")}` : ""}`).join("、");
      const more = a.list.length > 8 ? ` 等 ${a.list.length} 个` : "";
      use = `<b>${a.list.length}</b>${a.of != null ? `<span class="muted"> / ${a.of}</span>` : ""}<div class="who" title="${esc(a.list.map((x) => x.m.name).join("、"))}">${shown}${more}</div>`;
    }
    const ev = (t.evidence || []).map((e) => `<div><span class="by">${esc(e.by)}</span>${esc(e.text)}${e.url ? ` ${link(e.url, "↗")}` : ""}</div>`).join("") || `<span class="muted">–</span>`;
    return `<tr id="tech-${TECHS.indexOf(t)}">
      <td class="l"><b>${esc(t.name)}</b><div class="muted cat">${esc(CAT_LABEL[t.cat])}${t.note ? ` · ${esc(t.note)}` : ""}</div></td>
      <td class="l src"><span class="date">${esc(t.date)}</span>${link(t.source.url, esc(t.source.title))}</td>
      <td class="l ev">${ev}</td>
      <td class="l use">${use}</td></tr>`;
  }).join("");
}

/* ---------- technique timeline: from the paper to the configs that use it ---------- */

const mon = (s) => { const [y, m = 1] = String(s).split("-").map(Number); return y * 12 + m - 1; };
const monText = (i) => `${Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, "0")}`;
const lagText = (d) => (d > 0 ? `隔 ${d} 个月` : d === 0 ? "和论文同月" : `比论文早 ${-d} 个月`);
let tl = null; // geometry and rows of the last render, for the hover layer

function renderTimeline(rows, ts) {
  const box = $("#tech-tl");
  const all = ts.map((t) => ({ t, a: adopters(t, rows) }));
  const shown = all.filter((x) => x.a && x.a.list.length);
  const none = all.filter((x) => x.a && !x.a.list.length).sort((x, y) => y.t.date.localeCompare(x.t.date));
  const blind = all.filter((x) => !x.a).length;
  if (!shown.length) {
    tl = null;
    box.innerHTML = `<p class="muted">当前筛选下，这一类技术在收录的配置里都还没有出现。</p>`;
    return;
  }
  // Configs start in 2025 (DATA, not the filtered rows, so the axis holds still while filtering); the years
  // before are squeezed into a shaded band, since only paper dates fall there.
  const split = mon(`${DATA.reduce((a, m) => (m.released < a ? m.released : a), "9999").slice(0, 4)}-01`);
  const start = Math.min(split - 12, mon(`${TECHS.reduce((a, t) => (t.date < a ? t.date : a), "9999").slice(0, 4)}-01`));
  const lastYear = Math.max(...DATA.map((m) => +m.released.slice(0, 4)), ...TECHS.map((t) => +t.date.slice(0, 4)));
  const end = mon(`${lastYear + 1}-01`);
  const narrow = box.clientWidth < 560;
  const nameW = narrow ? 104 : 196, countW = narrow ? 0 : 30;
  const W = Math.max(160, box.clientWidth - 24 - nameW - countW - (narrow ? 6 : 16) - 2); // padding, columns, gaps
  const f = narrow ? 0.16 : 0.2;
  const x = (t) => (t <= split ? f * W * Math.max(0, t - start) / (split - start) : f * W + (1 - f) * W * (t - split) / (end - split));
  const perMonth = (1 - f) * W / (end - split);
  const step = perMonth >= 11 ? 1 : 3; // months per dot: a month on desktop, a quarter on phones
  const binW = perMonth * step;
  const H = 20, cy = H / 2;

  const items = shown.map(({ t, a }) => {
    const paper = mon(t.date);
    const bins = new Map();
    for (const { m } of a.list) {
      const b = Math.floor(mon(m.released) / step) * step;
      if (!bins.has(b)) bins.set(b, []);
      bins.get(b).push(m);
    }
    const list = [...bins.entries()].sort((p, q) => p[0] - q[0])
      .map(([b, ms]) => ({ b, ms: ms.sort((p, q) => p.name.localeCompare(q.name)), cx: x(b + step / 2), r: Math.min(binW / 2 - 0.75, 4 + 1.5 * (Math.sqrt(ms.length) - 1)) }));
    const first = Math.min(...a.list.map(({ m }) => mon(m.released)));
    return { t, a, paper, first, bins: list, firstMs: a.list.filter(({ m }) => mon(m.released) === first).map(({ m }) => m) };
  }).sort((p, q) => p.paper - q.paper || p.first - q.first || p.t.name.localeCompare(q.t.name));
  tl = { x, items, step, H, split };

  const grid = [];
  for (let t = split; t < end; t += 6) grid.push(`<line class="${t % 12 ? "tl-g half" : "tl-g"}" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H}"/>`);
  const back = `<rect class="tl-old" x="0" y="0" width="${x(split)}" height="${H}"/>${grid.join("")}<line class="tl-lead" x1="0" x2="${W}" y1="${cy}" y2="${cy}"/>`;
  const ticks = [];
  if (narrow) ticks.push([0, "更早"]);
  else for (let y = Math.floor(start / 12); y < Math.floor(split / 12); y++) ticks.push([x(y * 12), String(y)]);
  for (let t = split; t < end; t += 6) if (!(t % 12) || !narrow) ticks.push([x(t), t % 12 ? `${t % 12 + 1} 月` : String(t / 12)]);
  const axis = `<div class="tl-row tl-axis"><span></span><svg width="${W}" height="16" aria-hidden="true">` +
    ticks.map(([px, s]) => `<text x="${px + 3}" y="12">${s}</text>`).join("") + `</svg>${countW ? "<span></span>" : ""}</div>`;
  const short = (n) => n.replace(/（.*）$/, "");
  const fits = (s) => textWidth(s, 12) <= nameW - 10;
  const body = items.map((it, i) => {
    const { t, a, paper, first, bins } = it;
    let svg = back;
    const px = x(paper + 0.5);
    const fx = bins[0].cx, lx = bins[bins.length - 1].cx;
    if (first > paper) svg += `<line class="tl-wait" x1="${px}" x2="${fx}" y1="${cy}" y2="${cy}"/>`;
    if (lx > fx) svg += `<line class="tl-span" x1="${fx}" x2="${lx}" y1="${cy}" y2="${cy}"/>`;
    svg += bins.map((b) => `<circle class="tl-dot" cx="${b.cx}" cy="${cy}" r="${b.r}"/>`).join("");
    // where a config came out the same month (quarter on phones), the diamond becomes an outline around its dot
    const on = bins.find((b) => b.b === Math.floor(paper / tl.step) * tl.step);
    const dh = on ? Math.min(9.5, (on.r + 1.5) * Math.SQRT2) : 5.5;
    svg += `<path class="tl-paper${on ? " ring" : ""}" d="M${on ? on.cx : px} ${cy - dh}l${dh} ${dh}-${dh} ${dh}-${dh}-${dh}z"/><circle class="tl-cur" r="0" cy="${cy}" hidden/>`;
    const name = fits(t.name) || items.filter((o) => short(o.t.name) === short(t.name)).length > 1 ? t.name : short(t.name);
    return `<div class="tl-row" data-i="${i}"><a class="tl-name" href="#tech-${TECHS.indexOf(t)}" title="${esc(t.name)}">${esc(name)}</a>` +
      `<svg class="tl-track" width="${W}" height="${H}" role="img" aria-label="${esc(t.name)}：论文 ${t.date}，第一个配置 ${monText(first)}，共 ${a.list.length} 个">${svg}</svg>` +
      (countW ? `<span class="tl-n">${a.list.length}</span>` : "") + `</div>`;
  }).join("");

  // takeaway: among techniques published while the configs were being collected, how long until the first one
  const fresh = items.filter((it) => it.paper >= split);
  const lags = fresh.map((it) => it.first - it.paper);
  const waited = lags.filter((d) => d > 0);
  const same = lags.filter((d) => d === 0).length, before = lags.filter((d) => d < 0).length;
  const year = Math.floor(split / 12);
  const top = (fresh.length ? fresh : items).slice().sort((p, q) => q.a.list.length - p.a.list.length).slice(0, 3)
    .map((it) => `${esc(short(it.t.name))}（${it.a.list.length}）`);
  let take = "";
  if (fresh.length) {
    const parts = [];
    if (same) parts.push(`<b>${same}</b> 项的论文和第一个用它的配置同月发出`);
    if (before) parts.push(`<b>${before}</b> 项是配置先出、论文后发`);
    if (waited.length) parts.push(`其余 ${waited.length} 项从论文到第一个配置的中位数是 <b>${median(waited)}</b> 个月`);
    take = `${year} 年以后发表、已经有配置在用的 <b>${fresh.length}</b> 项里，${parts.join("，")}。` +
      `其中用得最多的是${top.join("、")}。`;
  } else take = `用得最多的是${top.join("、")}。`;
  const noneText = none.length ? `还没有收录的配置用上的 ${none.length} 项：${none.slice(0, 10).map((x) => esc(short(x.t.name))).join("、")}${none.length > 10 ? " 等" : ""}。` : "";
  box.innerHTML = `<h3>从论文到落地：每行一项技术</h3><span class="take">${take}</span>` +
    `<p class="legend-note tl-legend"><span><i class="tl-k paper"></i>论文或技术报告</span><span><i class="tl-k wait"></i>到第一个收录配置之前</span>` +
    `<span><i class="tl-k dot"></i><i class="tl-k dot big"></i>收录配置的首发，圆越大这${step === 1 ? "个月" : "个季度"}越多</span>` +
    `<span><svg class="tl-k ring" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="4"/><path d="M8 0.5l7.5 7.5-7.5 7.5-7.5-7.5z"/></svg>论文和配置同${step === 1 ? "月" : "季度"}</span>` +
    `<span><i class="tl-k old"></i>${monText(split).slice(0, 4)} 年以前：只有论文，压缩显示</span></p>` +
    `<div class="tl-wrap">${axis}${body}</div>` +
    `<p class="muted tl-note">${noneText}${blind ? `另有 ${blind} 项从 config 看不出、也没有核对过的采用记录，只在下表列出。` : ""}点技术名跳到下表那一行。</p>`;
}

function timelineTip(it, hit) {
  const { t, a, paper, first, firstMs } = it;
  const head = `<b>${esc(t.name)}</b> <span class="muted">${esc(CAT_LABEL[t.cat])}</span><br>`;
  if (hit && hit.b !== Math.floor(paper / tl.step) * tl.step) { // the paper's own bin gets the summary, which names its configs
    const when = tl.step === 1 ? monText(hit.b) : `${Math.floor(hit.b / 12)} Q${(hit.b % 12) / 3 + 1}`;
    const list = hit.ms.map((m) => esc(m.name));
    return `${head}${when} 首发的 ${hit.ms.length} 个配置：<br><span class="muted">${list.slice(0, 16).join("、")}${list.length > 16 ? ` 等 ${list.length} 个` : ""}</span>`;
  }
  const early = paper < tl.split ? `<br><span class="muted">收录的配置从 ${Math.floor(tl.split / 12)} 年开始，更早的采用不在这里。</span>` : "";
  return `${head}论文 ${t.date}：${esc(t.source.title)}<br>第一个收录的配置：${monText(first)}，${esc(firstMs.map((m) => m.name).slice(0, 6).join("、"))}${firstMs.length > 6 ? ` 等 ${firstMs.length} 个` : ""}（${lagText(first - paper)}）` +
    `<br>共 ${a.list.length} 个配置${a.of != null ? `<span class="muted">（config 能看出来的 ${a.of} 个里）</span>` : ""}${early}`;
}

function initTimeline() {
  const box = $("#tech-tl");
  const clear = () => box.querySelectorAll(".tl-cur:not([hidden])").forEach((c) => c.setAttribute("hidden", ""));
  box.addEventListener("pointermove", (e) => {
    const row = e.target.closest ? e.target.closest(".tl-row[data-i]") : null;
    clear();
    const it = row && tl && tl.items[+row.dataset.i];
    if (!it) return hideTip();
    const svg = row.querySelector(".tl-track");
    const px = e.clientX - svg.getBoundingClientRect().left;
    let hit = null;
    if (e.target.closest(".tl-track")) {
      const near = it.bins.map((b) => ({ b, d: Math.abs(b.cx - px) })).sort((p, q) => p.d - q.d)[0];
      const dPaper = Math.abs(tl.x(it.paper + 0.5) - px);
      if (near && near.d <= Math.max(near.b.r + 3, 7) && near.d < dPaper) {
        hit = near.b;
        const cur = svg.querySelector(".tl-cur");
        cur.setAttribute("cx", hit.cx);
        cur.setAttribute("r", hit.r + 2.5);
        cur.removeAttribute("hidden");
      }
    }
    showTip(timelineTip(it, hit), e);
  });
  box.addEventListener("pointerleave", () => { hideTip(); clear(); });
}

/* ---------- design details: one position, the variants each model uses ---------- */

const DSECS = [["gate", "门与 sigmoid"], ["attn", "注意力与位置"], ["linear", "线性注意力"], ["moe", "MoE 与 FFN"],
  ["resid", "残差与归一化"], ["io", "输出与发布"], ["all", "全部"]];
let DPOS = [];

// Same rules as detail_ok in scripts/build.py, which checks every model fits a variant.
const truthy = (v) => !(v == null || v === false || v === "" || v === 0 || (Array.isArray(v) && !v.length));
const wild = (w) => new RegExp(`^${w.split("*").map((x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join(".*")}$`);
function detailOk(v, want) {
  if (Array.isArray(want)) return want.some((w) => detailOk(v, w));
  if (want === null) return v == null;
  if (Array.isArray(v)) return want === false ? !v.length : v.some((x) => detailOk(x, want));
  if (want === "*") return truthy(v);
  if (want === false || want === true) return v === want;
  if (typeof want === "string" && want.includes("*")) return truthy(v) && wild(want).test(String(v));
  return typeof want === "number" ? v === want : String(v) === String(want);
}
const matchAll = (m, cond) => Object.entries(cond || {}).every(([k, w]) => detailOk(m[k], w));
const applies = (p, m) => matchAll(m, p.when);
function detailHits(p, m) {
  if (!applies(p, m)) return [];
  const hits = p.variants.filter((v) => !v.related && matchAll(m, v.match));
  return p.multi ? hits : hits.slice(0, 1);
}

const sci = (x) => { const e = Math.floor(Math.log10(x)); return `${+(x / 10 ** e).toFixed(1)}e${e}`; };
const DFMT = {
  rope_theta: (v) => (v ? `θ=${sci(v)}` : ""),
  rope_ext: (v) => (v ? String(v).replace(/^(YaRN|Llama 3 式|NTK) ?/, "") : ""),
  router_scale: (v) => (typeof v === "number" ? `λ=${v}` : v),
  act_clamp: (v) => `c=${v}`.replace(/（(.+)）$/, "，$1"),
  softcap: (v) => `c=${v}`,
  len_scale: (v) => (v === true ? "" : v),
};
function detailSuffix(p, v, m) {
  const show = [].concat(v.show ?? p.show ?? []);
  return show.map((k) => {
    const x = m[k];
    if (!truthy(x)) return "";
    return Array.isArray(x) ? x.join("、") : String(DFMT[k] ? DFMT[k](x) : x);
  }).filter(Boolean).join(" · ");
}
// What the compare table shows for one model at one position.
function detailText(p, m) {
  const hits = detailHits(p, m);
  if (!hits.length) return applies(p, m) ? "?" : "–";
  return hits.map((v) => { const sfx = detailSuffix(p, v, m); return sfx ? `${v.label}（${sfx}）` : v.label; }).join("；");
}

// Formulas are plain text with _x / _{..} subscripts and ^x / ^(..) superscripts.
const fmla = (s) => esc(s)
  .replace(/\^\(((?:[^()]|\([^()]*\))*)\)/g, "<sup>$1</sup>")
  .replace(/\^([A-Za-z0-9¼]+)/g, "<sup>$1</sup>")
  .replace(/_\{([^}]*)\}/g, "<sub>$1</sub>")
  .replace(/_([A-Za-z0-9]+)/g, "<sub>$1</sub>");

function modelChip(m, sfx) {
  const on = state.compare.includes(m.name);
  return `<button type="button" class="mchip" data-name="${esc(m.name)}" aria-pressed="${on}" title="${esc(m.vendor)} · ${m.released} · 点击${on ? "移出" : "加入"}并排对比">` +
    `<span class="tier-dot" style="background:${tierColor(m.tier)}"></span>${esc(m.name)}${sfx ? `<span class="sfx">${esc(sfx)}</span>` : ""}</button>`;
}
function syncDetailPicks() {
  document.querySelectorAll("#det .mchip").forEach((b) => {
    const on = state.compare.includes(b.dataset.name);
    b.setAttribute("aria-pressed", on);
    b.title = b.title.replace(/点击.*并排对比/, `点击${on ? "移出" : "加入"}并排对比`);
  });
}

// Which variant one model counts under in the share bars: its first match, or for a multi position its first
// match other than the default; -1 when the position doesn't apply to it.
function detailBucket(p, m) {
  if (!applies(p, m)) return -1;
  const hits = p.variants.map((v, i) => [v, i]).filter(([v]) => !v.related && matchAll(m, v.match));
  if (!hits.length) return -1;
  return (p.multi ? hits.find(([v]) => !v.default) || hits[0] : hits[0])[1];
}

let measureCtx = null;
function textWidth(t, size = 11) {
  if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
  measureCtx.font = `${size}px system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif`;
  return measureCtx.measureText(t).width;
}

const ddSegs = new Map();
function renderDetailDist(rows, ps) {
  const box = $("#det-dist");
  const years = [...new Set(rows.map((m) => m.year))].sort();
  ddSegs.clear();
  if (!ps.length || !rows.length) { box.hidden = true; return; }
  box.hidden = false;
  const narrow = box.clientWidth < 560;
  const trackW = Math.max(120, box.clientWidth - 26 - (narrow ? 0 : 162) - 34 - 26 - 12);
  const moves = [];
  const html = ps.map((p) => {
    const slot = new Map();
    p.variants.forEach((v, i) => { if (!v.default && !v.related) slot.set(i, Math.min(8, slot.size + 1)); });
    const order = [...p.variants.keys()].filter((i) => p.variants[i].default).concat([...slot.keys()]);
    const shares = {};
    const used = new Set();
    const bars = years.map((y) => {
      const ms = rows.filter((m) => m.year === y);
      const groups = new Map(order.map((i) => [i, []]).concat([[-1, []]]));
      for (const m of ms) groups.get(detailBucket(p, m))?.push(m);
      const segs = [...groups.entries()].filter(([, g]) => g.length);
      shares[y] = Object.fromEntries([...groups.entries()].map(([i, g]) => [i, ms.length ? g.length / ms.length : 0]));
      const inner = segs.map(([i, g]) => {
        const v = p.variants[i];
        const label = v ? v.label : p.when ? "没有这一项" : "不适用";
        const w = (trackW - 2 * (segs.length - 1)) * g.length / ms.length;
        const full = `${label} ${g.length}`;
        const text = textWidth(full) + 10 <= w ? full : textWidth(String(g.length)) + 10 <= w ? String(g.length) : "";
        const key = `${p.id}|${y}|${i}`;
        ddSegs.set(key, { p, y, label, ms: g, total: ms.length });
        if (i < 0) return `<i class="dd-seg none" style="flex-grow:${g.length}" data-k="${esc(key)}">${esc(text)}</i>`;
        if (v.default) return `<i class="dd-seg dflt" style="flex-grow:${g.length}" data-k="${esc(key)}">${esc(text)}</i>`;
        used.add(i);
        return `<i class="dd-seg${slot.get(i) === 4 ? " dark-ink" : ""}" style="flex-grow:${g.length};background:var(--cat-${slot.get(i)})" data-k="${esc(key)}">${esc(text)}</i>`;
      }).join("");
      return `<div class="dd-bar"><span class="yr">${y}</span><div class="dd-track">${inner}</div><span class="tot">${ms.length}</span></div>`;
    }).join("");
    if (years.length > 1) { // the variant whose share grew the most between the first and the last year
      const [a, b] = [shares[years[0]], shares[years[years.length - 1]]];
      const cands = Object.keys(b).map((i) => ({ label: p.variants[i]?.label ?? "没有这一项", from: a[i] ?? 0, to: b[i], bonus: 0 }));
      // where the position is optional, having it at all can be the bigger change than any one way of doing it
      if (p.when) cands.push({ label: "有这一项", from: 1 - (a[-1] ?? 0), to: 1 - (b[-1] ?? 0), bonus: -0.02 });
      const best = cands.reduce((x, c) => (x && x.to - x.from + x.bonus >= c.to - c.from + c.bonus ? x : c), null);
      if (best && best.to > best.from) {
        moves.push({ d: best.to - best.from, text: `${p.name}「${best.label}」${Math.round(100 * best.from)}% → ${Math.round(100 * best.to)}%` });
      }
    }
    const key = [...slot.entries()].filter(([i]) => used.has(i))
      .map(([i, k]) => `<span><i style="background:var(--cat-${k})"></i>${esc(p.variants[i].label)}</span>`).join("");
    const app = rows.filter((m) => applies(p, m)).length;
    return `<div class="dd-row"><div class="dd-name"><a href="#pos-${p.id}">${esc(p.name)}</a><small>${app} / ${rows.length} 个配置适用</small></div>` +
      `<div>${bars}${key ? `<div class="dd-key">${key}</div>` : ""}</div></div>`;
  }).join("");
  const top = moves.sort((x, y) => y.d - x.d).slice(0, 3).map((x) => esc(x.text));
  box.innerHTML = `<h3>各写法的占比：${years.join(" 年和 ")} 年首发的配置分开算</h3>` +
    (top.length ? `<span class="take">涨得最多：${top.join("；")}。</span>` : "") + html +
    `<p class="legend-note dd-legend"><span><i class="sw on"></i>灰色：多数做法</span><span><i class="sw off"></i>空白：没有这一项或不适用</span><span>彩色：其他写法，悬停看型号，点名称跳到下面的详细对照</span></p>`;
}

function initDetailDist() {
  const box = $("#det-dist");
  box.addEventListener("pointermove", (e) => {
    const seg = e.target.closest ? e.target.closest(".dd-seg") : null;
    const d = seg && ddSegs.get(seg.dataset.k);
    if (!d) return hideTip();
    const list = d.ms.map((m) => esc(m.name));
    showTip(`<b>${esc(d.p.name)}</b> <span class="muted">${d.y} 年首发</span><br>${esc(d.label)}：${d.ms.length}/${d.total}（${Math.round(100 * d.ms.length / d.total)}%）` +
      `<br><span class="muted">${list.slice(0, 24).join("、")}${list.length > 24 ? ` 等 ${list.length} 个` : ""}</span>`, e);
  });
  box.addEventListener("pointerleave", hideTip);
}

function renderDetails(rows) {
  const inTab = (p, id) => id === "all" || (id === "gate" ? p.gate : p.sec === id);
  groupChips($("#det-secs"), state.detSec,
    DSECS.map(([id, label]) => ({ id, label: `${label} ${DPOS.filter((p) => inTab(p, id)).length}` })),
    (id) => { state.detSec = id; renderDetails(filtered()); });
  const link = (src) => (src ? ` <a href="${esc(src.url)}" target="_blank" rel="noopener" title="${esc(src.title)}">出处 ↗</a>` : "");
  let hidden = 0;
  renderDetailDist(rows, DPOS.filter((p) => inTab(p, state.detSec) && rows.some((m) => applies(p, m))));
  const cards = DPOS.filter((p) => inTab(p, state.detSec)).map((p) => {
    const app = rows.filter((m) => applies(p, m));
    if (!app.length) { hidden++; return ""; }
    const by = new Map(p.variants.map((v) => [v, []]));
    for (const m of app) for (const v of detailHits(p, m)) by.get(v).push(m);
    const order = p.sort ? (a, b) => (a[p.sort] ?? 0) - (b[p.sort] ?? 0) || a.name.localeCompare(b.name)
      : (a, b) => b.released.localeCompare(a.released) || a.name.localeCompare(b.name);
    const vars = p.variants.map((v) => {
      const ms = by.get(v).slice().sort(order);
      if (!v.related && !ms.length) return "";
      const head = `<div class="dl"><b>${esc(v.label)}</b>${v.formula ? `<span class="f">${fmla(v.formula)}</span>` : ""}` +
        (v.note || v.source ? `<p class="note">${esc(v.note ?? "")}${link(v.source)}</p>` : "") + `</div>`;
      let who;
      if (v.related) who = `<span class="muted">相关工作，收录的模型里还没有</span>`;
      else {
        const chips = ms.map((m) => modelChip(m, detailSuffix(p, v, m))).join("");
        who = v.default
          ? `<details><summary><b>${ms.length}</b> 个配置（多数做法，展开看）</summary><div class="dw">${chips}</div></details>`
          : `<span class="n">${ms.length}</span>${chips}`;
      }
      return `<div class="dv${v.related ? " rel" : ""}">${head}<div class="dw">${who}</div></div>`;
    }).join("");
    return `<article class="dpos" id="pos-${esc(p.id)}"><h3>${esc(p.name)}<span class="muted">${app.length} 个配置适用</span></h3>` +
      `<p class="what">${esc(p.what ?? "")}${p.formula ? `<span class="f pf">${fmla(p.formula)}</span>` : ""}</p>${vars}</article>`;
  });
  $("#det").innerHTML = cards.join("") + (hidden ? `<p class="muted det-hidden">当前筛选下有 ${hidden} 个位置没有适用的配置，已隐藏。</p>` : "");
  $("#det").querySelectorAll(".mchip").forEach((b) => b.onclick = () => togglePick(b.dataset.name));
}

/* ---------- table ---------- */

const BASE_COLS = [
  { key: "vendor", label: "厂家", l: true, f: (m) => `<span class="tier-dot" style="background:${tierColor(m.tier)}"></span>${esc(m.vendor)}` },
  { key: "name", label: "型号", l: true, f: nameCell },
  { key: "released", label: "首发", title: "这个 MoE 配置第一次上传 HF 的年月" },
  { key: "latest", label: "最近上传", title: "同配置各版本里最近一次上传 HF 的日期", f: (m) => m.latest ?? "–" },
  { key: "total", label: "总参数", f: (m) => fmtParams(m.total) },
  { key: "active", label: "激活", f: (m) => fmtParams(m.active) },
];
const COL_GROUPS = {
  sparsity: { label: "稀疏度", cols: [
    { key: "E", label: "E/k+s", f: fmtConfig },
    { key: "E_over_k", label: "E/k", f: (m) => num(m.E_over_k) },
    { key: "expert_activation", label: "A", title: "专家激活比 (k+s)/(E+s)", f: (m) => `${num(m.expert_activation)}%` },
    { key: "param_activation", label: "激活比", title: "激活参数 / 总参数", f: (m) => `${num(m.param_activation)}%` },
    { key: "granularity", label: "G", title: "专家粒度 2d/d_expert", f: (m) => num(m.granularity) },
  ] },
  shape: { label: "形状", cols: [
    { key: "layers", label: "层（MoE）", f: (m) => m.layers ? `${m.layers}（${m.moe_layers ?? "–"}）` : "–" },
    { key: "d_model", label: "d", f: (m) => m.d_model ?? "–" },
    { key: "d_expert", label: "d_expert", f: (m) => m.d_expert ?? "–" },
    { key: "aspect", label: "d/L", title: "隐藏维度 / token mixer 层数", f: (m) => num(m.aspect) },
    { key: "vocab", label: "词表", f: (m) => m.vocab ? `${Math.round(m.vocab / 1000)}K` : "–" },
    { key: "mtp", label: "MTP", title: "多 token 预测层数", f: (m) => m.mtp ?? "–" },
  ] },
  attention: { label: "注意力", cols: [
    { key: "attention", label: "层构成", title: "按 config.json 字段推断的层类型", l: true, f: (m) => esc(m.attention ?? "–") },
    { key: "global_attn_frac", label: "全局层", title: "全局注意力层占 token mixer 层的比例", f: (m) => m.global_attn_frac == null ? "–" : `${num(m.global_attn_frac)}%` },
    { key: "kv_kib_per_token", label: "KV/token", title: "长上下文下每 token 的 bf16 KV 缓存", f: (m) => m.kv_kib_per_token == null ? "–" : `${num(m.kv_kib_per_token)} KiB` },
    { key: "window", label: "滑窗", f: (m) => windowLabel(m) ?? "–" },
    { key: "context", label: "上下文", f: (m) => fmtCtx(m.context) },
  ] },
  attnShape: { label: "注意力形状", cols: [
    { key: "heads", label: "Q 头", title: "全局注意力层的 query 头数", f: (m) => m.heads ?? "–" },
    { key: "kv_heads", label: "KV 头", title: "全局层的 KV 头数；MLA 缓存的是压缩潜向量，没有独立 KV 头",
      f: (m) => (m.mla ? "MLA" : m.kv_heads ?? (m.gqa ? Math.round(m.heads / m.gqa) : "–")), sort: (m) => (m.mla ? null : m.kv_heads ?? (m.gqa ? m.heads / m.gqa : null)) },
    { key: "qk_dim", label: "QK 维", title: "每个头的 Q·K 维度；MLA 是不加位置的部分 + RoPE 部分", f: (m) => m.qk_dim ?? "–" },
    { key: "attn_ratio", label: "宽度比", title: "注意力宽度比：Q 头 × QK 维 / d_model", f: (m) => num(m.attn_ratio, 2) },
    { key: "gqa", label: "Q/KV", title: "每个 KV 头对应几个 query 头；MLA 不适用", f: (m) => (m.mla ? "MLA" : num(m.gqa)), sort: (m) => (m.mla ? null : m.gqa) },
    { key: "rope_frac", label: "RoPE", title: "全局层每个头里加 RoPE 的维度占比", f: ropeLabel },
  ] },
};
COL_GROUPS.bench = { label: "榜单", cols: [
  { key: "avg", need: "avg", label: "归一均分", title: "各榜换成标准分后取平均，站内最高 100、最低 0", f: (m) => benchFmt("avg", m.avg) },
  { key: "eci", need: "eci", label: "ECI", title: "Epoch Capabilities Index：几十项 benchmark 拟合出的综合能力分，同一配置的多个版本取最高", f: (m) => benchFmt("eci", m.eci) },
  { key: "eci_as", need: "eci", label: "ECI 取自", l: true, f: (m) => esc(m.eci_as ?? "–") },
  { key: "arena", need: "arena", label: "Arena", title: "Arena 文本榜分数（默认的 style control 榜），同一配置的多个版本取最高", f: (m) => benchFmt("arena", m.arena) },
  { key: "arena_as", need: "arena", label: "Arena 取自", l: true, f: (m) => esc(m.arena_as ?? "–") },
  { key: "arena_votes", need: "arena", label: "票数", title: "Arena 上取分那个版本的投票数", f: (m) => (m.arena_votes == null ? "–" : m.arena_votes.toLocaleString("en-US")) },
  { key: "aa", need: "aa", label: "AA", title: "Artificial Analysis 智能指数：他们自己跑的一组评测合成的指数，同一配置的多个版本取最高", f: (m) => benchFmt("aa", m.aa) },
  { key: "aa_as", need: "aa", label: "AA 取自", l: true, f: (m) => esc(m.aa_as ?? "–") },
] };
// a leaderboard column shows once its board has scores (AA needs an API key at build time)
const cols = () => BASE_COLS.concat(COL_GROUPS[state.group].cols.filter((c) => !c.need || BENCH[c.need]));

// Model name plus the newest same-config versions, so a fresh post-training
// release (e.g. a -1017 checkpoint) is visible without expanding the row.
function nameCell(m) {
  const others = m.versions.filter((v) => v.repo !== m.repo);
  if (!others.length) return esc(m.name);
  const shown = others.slice(0, 3).map((v) => esc(v.name)).join("、");
  const more = others.length > 3 ? ` 等 ${others.length} 个` : "";
  return `${esc(m.name)}<div class="vers">同配置：${shown}${more}</div>`;
}

function renderColGroups() {
  const box = $("#col-groups");
  box.innerHTML = "";
  for (const [id, g] of Object.entries(COL_GROUPS)) {
    box.append(chip(g.label, state.group === id, () => { state.group = id; renderColGroups(); renderTable(filtered()); }));
  }
}

function renderTable(rows) {
  const COLS = cols();
  const head = $("#table thead tr");
  head.innerHTML = COLS.map((c) => {
    const sort = state.sort.key === c.key ? (state.sort.dir > 0 ? "ascending" : "descending") : "none";
    return `<th class="${c.l ? "l" : ""}" data-key="${c.key}" aria-sort="${sort}"${c.title ? ` title="${c.title}"` : ""}>${c.label}</th>`;
  }).join("");
  head.querySelectorAll("th").forEach((th) => th.onclick = () => {
    const k = th.dataset.key;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : (COLS.find((c) => c.key === k).l ? 1 : -1) };
    renderTable(filtered());
  });

  const { key, dir } = state.sort;
  const col = COLS.find((c) => c.key === key);
  const val = (m) => (col && col.sort ? col.sort(m) : m[key]);
  const sorted = rows.slice().sort((a, b) => {
    const x = val(a), y = val(b);
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * dir;
  });
  const body = $("#table tbody");
  body.innerHTML = sorted.map((m, i) => `
    <tr class="main" data-i="${i}">${COLS.map((c) => `<td class="${c.l ? "l" : ""}">${c.f ? c.f(m) : esc(m[c.key])}</td>`).join("")}</tr>
    <tr class="detail" hidden><td colspan="${COLS.length}">${detail(m)}</td></tr>`).join("");
  body.querySelectorAll("tr.main").forEach((tr) => tr.onclick = () => {
    const d = tr.nextElementSibling;
    d.hidden = !d.hidden;
  });
}

function repoLink(r) {
  return r.includes("/") ? `<a href="https://huggingface.co/${esc(r)}" target="_blank" rel="noopener">${esc(r)}</a>` : esc(r);
}

function detail(m) {
  const cfg = m.config_source
    ? `<a href="https://github.com/Skystellan/moe-config-atlas/blob/main/configs/${esc(m.repo.replace("/", "__"))}.json" target="_blank" rel="noopener">config.json（本仓库存档）</a>`
    : `<a href="https://huggingface.co/${esc(m.repo)}/blob/main/config.json" target="_blank" rel="noopener">config.json</a>`;
  const items = [
    ["仓库", `${repoLink(m.repo)} · ${cfg}`],
    m.config_source && ["config 来源", esc(m.config_source)],
    m.versions.length > 1 && ["各版本（新→旧）", m.versions.map((v) =>
      `${v.repo ? repoLink(v.repo) : esc(v.name)} <span class="muted">${v.created ?? "未上 HF"}${v.note ? `，${esc(v.note)}` : ""}</span>`).join("<br>")],
    ["model_type", esc(m.model_type ?? "–")],
    m.d_shared && ["共享专家宽度", m.d_shared],
    m.latent && ["LatentMoE 潜空间维度", m.latent],
    m.zero_experts && ["零计算专家", m.zero_experts],
    m.hf_params && ["HF 统计的权重参数", fmtParams(m.hf_params)],
    m.license && ["许可", esc(m.license)],
    !m.config_ok && ["说明", "config.json 无法公开读取，形状取自模型卡"],
    m.notes && ["备注", esc(m.notes)],
  ].filter(Boolean);
  return `<dl>${items.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

/* ---------- glue ---------- */

function update() {
  const rows = filtered();
  $("#count").textContent = `${rows.length} / ${DATA.length} 种配置`;
  renderControls();
  renderTiles(rows);
  renderCharts(rows);
  renderLayers(rows);
  renderFingerprint(rows);
  renderCompare();
  renderAdoption(rows);
  renderDetails(rows);
  renderTechniques(rows);
  renderVendors(rows);
  renderColGroups();
  renderTable(rows);
}

/* ---------- folding: every titled block collapses from its heading; the choice is kept per browser ---------- */

const FOLD_KEY = "folds";
const folds = [];

// A shut block keeps its width and only loses its height, so charts and hand-drawn views
// laid out while it is shut come back at the right size without a re-render.
function setFold(f, shut) {
  f.shut = shut;
  f.btn.setAttribute("aria-expanded", String(!shut));
  f.body.classList.toggle("shut", shut);
}
function saveFolds() {
  try { localStorage.setItem(FOLD_KEY, JSON.stringify(folds.filter((f) => f.shut).map((f) => f.key))); } catch (e) {}
  const all = $("#fold-all");
  if (all) all.textContent = folds.length && folds.every((f) => f.shut) ? "全部展开" : "全部折叠";
}
function initFolds() {
  let shut = new Set();
  try { shut = new Set(JSON.parse(localStorage.getItem(FOLD_KEY) || "[]")); } catch (e) {}
  const make = (head, nodes, card) => {
    const body = document.createElement("div");
    body.className = "fold-body";
    body.id = `fold-${folds.length}`;
    head.after(body);
    body.append(...nodes);
    const key = head.textContent.trim();
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "fold";
    btn.setAttribute("aria-controls", body.id);
    btn.innerHTML = `<span>${head.innerHTML}</span><svg class="fold-ico" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5"/></svg>`;
    head.replaceChildren(btn);
    const f = { key, head, body, btn, card };
    btn.onclick = () => { setFold(f, !f.shut); saveFolds(); };
    folds.push(f);
    setFold(f, shut.has(key));
  };
  for (const el of [...document.querySelectorAll("main > h2.sec, main > section.card")]) {
    if (el.matches("section.card")) {
      const h = el.querySelector(":scope > h2");
      if (h) make(h, [...el.children].filter((n) => n !== h), el);
      continue;
    }
    // a section heading folds what it titles: a grid of charts with its legend, or a card without a title of its own
    const nodes = [];
    for (let n = el.nextElementSibling; n && !n.matches("h2, section.card"); n = n.nextElementSibling) nodes.push(n);
    const next = el.nextElementSibling;
    if (!nodes.length && next && next.matches("section.card") && !next.querySelector(":scope > h2")) nodes.push(next);
    if (nodes.length) make(el, nodes, null);
  }
  if ($("#fold-all")) $("#fold-all").onclick = () => {
    const open = folds.every((f) => f.shut);
    folds.forEach((f) => setFold(f, !open));
    saveFolds();
  };
  // following a link into a shut block opens it first, so the browser scrolls to something visible
  const reveal = (hash) => {
    let id = hash;
    try { id = decodeURIComponent(hash); } catch (e) {}
    const t = id && document.getElementById(id);
    if (!t) return;
    for (const f of folds) if (f.shut && (f.head === t || f.body.contains(t) || (f.card && f.card === t))) { setFold(f, false); saveFolds(); }
  };
  document.addEventListener("click", (e) => {
    const a = e.target.closest ? e.target.closest('a[href^="#"]') : null;
    if (a) reveal(a.getAttribute("href").slice(1));
  });
  addEventListener("hashchange", () => reveal(location.hash.slice(1)));
  if (location.hash && !location.hash.startsWith("#cmp=")) reveal(location.hash.slice(1));
  saveFolds();
}

function initTheme() {
  $("#theme").onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
    update();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", update);
}

async function main() {
  const [res, tres, dres] = await Promise.all(["models.json", "techniques.json", "details.json"].map((f) => fetch(dataUrl(f))));
  const data = await res.json();
  DATA = data.models;
  BENCH = data.bench || {};
  $("#aa-credit").hidden = !BENCH.aa;
  TECHS = tres.ok ? await tres.json() : [];
  DPOS = dres.ok ? await dres.json() : [];
  CMP_SECTIONS.push({ label: "设计细节", rows: DPOS.map((p) => ({ label: p.name, title: p.what, f: (m) => esc(detailText(p, m)) })) });
  computeBins();
  $("#fp-n").textContent = FEATURES.length;
  initCompare();
  $("#meta").textContent = `${data.count} 种 MoE 配置 · ${new Set(DATA.map((m) => m.vendor)).size} 家厂家 · 数据更新于 ${data.generated}`;
  const sel = $("#vendor");
  for (const v of [...new Set(DATA.map((m) => m.vendor))].sort()) sel.add(new Option(v, v));
  sel.onchange = () => { state.vendor = sel.value; update(); };
  $("#q").oninput = (e) => { state.q = e.target.value; update(); };
  initTheme();
  initLayers();
  initDetailDist();
  initAdoption();
  initVendors();
  initTimeline();
  update();
  let lastWidth = innerWidth, resizeTimer = null;
  addEventListener("resize", () => {
    charts.forEach((c) => c.resize());
    if (innerWidth === lastWidth) return; // mobile browsers fire resize when the toolbar hides
    lastWidth = innerWidth;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { const rows = filtered(); renderLayers(rows); renderDetails(rows); renderTechniques(rows); renderVendors(rows); }, 150);
  });
}

initFolds(); // before the data arrives, so folded sections never flash open
main().catch((e) => { $("#meta").textContent = `数据加载失败：${e.message}`; });
