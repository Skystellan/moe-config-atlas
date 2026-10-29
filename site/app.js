"use strict";

const TIERS = [
  { id: "S", label: "小（≤50B）" },
  { id: "M", label: "中（50–500B）" },
  { id: "L", label: "大（≥500B）" },
];
const state = { q: "", tiers: new Set(), years: new Set(), vendor: "", sort: { key: "latest", dir: -1 } };
let DATA = [];
const charts = [];

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

function tooltip() {
  return {
    trigger: "item",
    backgroundColor: css("--surface"),
    borderColor: css("--ring"),
    textStyle: { color: css("--ink"), fontSize: 12 },
    formatter: (p) => {
      const m = p.data.m;
      if (!m) return "";
      return `<b>${esc(m.name)}</b> <span style="color:${css("--muted")}">${esc(m.vendor)} · ${m.released}</span><br>` +
        `总 / 激活：${fmtParams(m.total)} / ${fmtParams(m.active)}（${num(m.param_activation)}%）<br>` +
        (m.latest_name !== m.repo.split("/").pop() ? `最近版本：${esc(m.latest_name)}（${m.latest ?? "–"}）<br>` : "") +
        `配置：${fmtConfig(m)}，E/k = ${num(m.E_over_k)}，A = ${num(m.expert_activation)}%` +
        (m.granularity ? `，G = ${num(m.granularity)}` : "");
    },
  };
}

function scatterSeries(rows, xf, yf) {
  return TIERS.map((t) => ({
    name: t.id,
    type: "scatter",
    symbolSize: 9,
    data: rows.filter((m) => m.tier === t.id && xf(m) != null && yf(m) != null).map((m) => ({
      value: [xf(m), yf(m)],
      m,
      itemStyle: m.year >= 2026
        ? { color: tierColor(t.id), borderColor: css("--surface"), borderWidth: 1.5 }
        : { color: css("--surface"), borderColor: tierColor(t.id), borderWidth: 1.8 },
    })),
    emphasis: {
      scale: 1.4,
      label: { show: true, formatter: (p) => p.data.m.name, position: "top", color: css("--ink"), fontSize: 11 },
    },
  }));
}

function chart(id) {
  const el = document.getElementById(id);
  let c = echarts.getInstanceByDom(el);
  if (!c) { c = echarts.init(el, null, { renderer: "svg" }); charts.push(c); }
  return c;
}

const grid = { left: 48, right: 16, top: 16, bottom: 40 };
const sizeAxis = (name) => baseAxis({
  type: "log", name, nameLocation: "middle", nameGap: 26, min: 5, max: 4000,
  axisLabel: { color: css("--muted"), fontSize: 11, formatter: (v) => fmtParams(v) },
});

function renderCharts(rows) {
  chart("c-param").setOption({
    grid, tooltip: tooltip(),
    xAxis: sizeAxis("总参数"),
    yAxis: baseAxis({ type: "value", min: 0, axisLabel: { color: css("--muted"), fontSize: 11, formatter: "{value}%" } }),
    series: scatterSeries(rows, (m) => m.total, (m) => m.param_activation),
  }, true);

  chart("c-ek").setOption({
    grid, tooltip: tooltip(),
    xAxis: sizeAxis("总参数"),
    yAxis: baseAxis({ type: "log", logBase: 2, min: 4, max: 128 }),
    series: scatterSeries(rows, (m) => m.total, (m) => m.E_over_k),
  }, true);

  const ratios = [8, 16, 32, 64];
  const refLines = ratios.map((r) => ({
    type: "line", silent: true, symbol: "none", z: 1,
    lineStyle: { color: css("--axis"), width: 1, type: "dashed" },
    data: [[2, 2 * r], [Math.min(32, 1024 / r), Math.min(32, 1024 / r) * r]],
    endLabel: { show: true, formatter: `E/k=${r}`, color: css("--muted"), fontSize: 10 },
  }));
  chart("c-ek2").setOption({
    grid: { ...grid, right: 56 }, tooltip: tooltip(),
    xAxis: baseAxis({ type: "log", logBase: 2, min: 2, max: 32, name: "每 token 选中的路由专家数 k", nameLocation: "middle", nameGap: 26 }),
    yAxis: baseAxis({ type: "log", logBase: 2, min: 16, max: 1024, name: "E", nameGap: 8 }),
    series: refLines.concat(scatterSeries(rows, (m) => m.k, (m) => m.E)),
  }, true);

  chart("c-gran").setOption({
    grid, tooltip: tooltip(),
    xAxis: sizeAxis("总参数"),
    yAxis: baseAxis({ type: "value", min: 0 }),
    series: scatterSeries(rows, (m) => m.total, (m) => m.granularity),
  }, true);

  $("#legend").innerHTML = TIERS.map((t) => `<span><i style="background:${tierColor(t.id)}"></i>${t.label}</span>`).join("") +
    `<span><i style="background:${css("--ink-2")}"></i>实心 2026 年首发</span><span><i class="hollow"></i>空心 2025 年首发</span>`;
}

function renderMultiples(rows) {
  const box = $("#multiples");
  for (const c of charts.splice(4)) c.dispose();
  const vendors = [...new Set(rows.map((m) => m.vendor))].sort();
  box.innerHTML = vendors.map((v, i) => `<div class="mult"><h4>${esc(v)}</h4><div class="m" id="mult-${i}"></div></div>`).join("");
  const minT = Date.UTC(2025, 0, 1);
  const maxT = Math.max(...DATA.map((m) => Date.parse(m.released + "-01"))) + 40 * 864e5;
  vendors.forEach((v, i) => {
    const ms = rows.filter((m) => m.vendor === v && m.E_over_k).sort((a, b) => a.released.localeCompare(b.released));
    const c = chart(`mult-${i}`);
    c.setOption({
      grid: { left: 30, right: 10, top: 10, bottom: 22 },
      tooltip: tooltip(),
      xAxis: baseAxis({ type: "time", min: minT, max: maxT, splitNumber: 3,
        axisLabel: { color: css("--muted"), fontSize: 10, formatter: (t) => { const d = new Date(t); return `${String(d.getUTCFullYear()).slice(2)}.${d.getUTCMonth() + 1}`; } } }),
      yAxis: baseAxis({ type: "log", logBase: 2, min: 4, max: 128, axisLabel: { color: css("--muted"), fontSize: 10 } }),
      // One line per size tier, as a stand-in for a product line.
      series: TIERS.map((t) => ({
        type: "line", symbolSize: 7, symbol: "circle",
        lineStyle: { color: tierColor(t.id), width: 1.5, opacity: 0.6 },
        itemStyle: { color: tierColor(t.id), borderColor: css("--surface"), borderWidth: 1 },
        data: ms.filter((m) => m.tier === t.id).map((m) => ({ value: [Date.parse(m.released + "-01"), m.E_over_k], m })),
      })),
    }, true);
  });
}

/* ---------- table ---------- */

const COLS = [
  { key: "vendor", label: "厂家", l: true, f: (m) => `<span class="tier-dot" style="background:${tierColor(m.tier)}"></span>${esc(m.vendor)}` },
  { key: "name", label: "型号", l: true, f: nameCell },
  { key: "released", label: "首发", title: "这个 MoE 配置第一次上传 HF 的年月" },
  { key: "latest", label: "最近上传", title: "同配置各版本里最近一次上传 HF 的日期", f: (m) => m.latest ?? "–" },
  { key: "total", label: "总参数", f: (m) => fmtParams(m.total) },
  { key: "active", label: "激活", f: (m) => fmtParams(m.active) },
  { key: "E", label: "E/k+s", f: fmtConfig },
  { key: "E_over_k", label: "E/k", f: (m) => num(m.E_over_k) },
  { key: "expert_activation", label: "A", title: "专家激活比 (k+s)/(E+s)", f: (m) => `${num(m.expert_activation)}%` },
  { key: "param_activation", label: "激活比", title: "激活参数 / 总参数", f: (m) => `${num(m.param_activation)}%` },
  { key: "granularity", label: "G", title: "专家粒度 2d/d_expert", f: (m) => num(m.granularity) },
  { key: "layers", label: "层（MoE）", f: (m) => m.layers ? `${m.layers}（${m.moe_layers ?? "–"}）` : "–" },
  { key: "d_model", label: "d", f: (m) => m.d_model ?? "–" },
  { key: "d_expert", label: "d_expert", f: (m) => m.d_expert ?? "–" },
  { key: "attention", label: "注意力", title: "按 config.json 字段推断的层类型", l: true, f: (m) => esc(m.attention ?? "–") },
];

// Model name plus the newest same-config versions, so a fresh post-training
// release (e.g. a -1017 checkpoint) is visible without expanding the row.
function nameCell(m) {
  const others = m.versions.filter((v) => v.repo !== m.repo);
  if (!others.length) return esc(m.name);
  const shown = others.slice(0, 3).map((v) => esc(v.name)).join("、");
  const more = others.length > 3 ? ` 等 ${others.length} 个` : "";
  return `${esc(m.name)}<div class="vers">同配置：${shown}${more}</div>`;
}

function renderTable(rows) {
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
  const sorted = rows.slice().sort((a, b) => {
    const x = a[key], y = b[key];
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
  const items = [
    ["仓库", `${repoLink(m.repo)} · <a href="https://huggingface.co/${esc(m.repo)}/blob/main/config.json" target="_blank" rel="noopener">config.json</a>`],
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
  renderMultiples(rows);
  renderTable(rows);
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
  const res = await fetch("data/models.json");
  const data = await res.json();
  DATA = data.models;
  $("#meta").textContent = `${data.count} 种 MoE 配置 · ${new Set(DATA.map((m) => m.vendor)).size} 家厂家 · 数据更新于 ${data.generated}`;
  const sel = $("#vendor");
  for (const v of [...new Set(DATA.map((m) => m.vendor))].sort()) sel.add(new Option(v, v));
  sel.onchange = () => { state.vendor = sel.value; update(); };
  $("#q").oninput = (e) => { state.q = e.target.value; update(); };
  initTheme();
  update();
  addEventListener("resize", () => charts.forEach((c) => c.resize()));
}

main().catch((e) => { $("#meta").textContent = `数据加载失败：${e.message}`; });
