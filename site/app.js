"use strict";

const TIERS = [
  { id: "S", label: "小（≤50B）" },
  { id: "M", label: "中（50–500B）" },
  { id: "L", label: "大（≥500B）" },
];
const state = { q: "", tiers: new Set(), years: new Set(), vendor: "", group: "sparsity", heat: "vendor", sort: { key: "latest", dir: -1 } };
let DATA = [];
const charts = [];
const multCharts = [];
const adoptCharts = [];

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
          (m.kv_kib_per_token ? `，KV ${num(m.kv_kib_per_token)} KiB/token` : "") + `，上下文 ${fmtCtx(m.context)}` : "");
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

function chart(id, list = charts) {
  const el = document.getElementById(id);
  let c = echarts.getInstanceByDom(el);
  if (!c) { c = echarts.init(el, null, { renderer: "svg" }); list.push(c); }
  return c;
}

const releaseTime = (m) => Date.parse(m.released + "-01");
const timeAxis = (extra) => baseAxis(Object.assign({
  type: "time", min: Date.UTC(2025, 0, 1),
  axisLabel: { color: css("--muted"), fontSize: 11, formatter: (t) => { const d = new Date(t); return `${d.getUTCFullYear()}.${d.getUTCMonth() + 1}`; } },
}, extra));

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

  chart("c-aspect").setOption({
    grid, tooltip: tooltip(),
    xAxis: baseAxis({ type: "log", name: "激活参数", nameLocation: "middle", nameGap: 26, min: 1, max: 128,
      axisLabel: { color: css("--muted"), fontSize: 11, formatter: (v) => fmtParams(v) } }),
    yAxis: baseAxis({ type: "value", min: 0 }),
    series: scatterSeries(rows, (m) => m.active, (m) => m.aspect),
  }, true);

  chart("c-global").setOption({
    grid, tooltip: tooltip(),
    xAxis: timeAxis(),
    yAxis: baseAxis({ type: "value", min: 0, max: 100, axisLabel: { color: css("--muted"), fontSize: 11, formatter: "{value}%" } }),
    series: scatterSeries(rows, releaseTime, (m) => m.global_attn_frac),
  }, true);

  chart("c-kv").setOption({
    grid, tooltip: tooltip(),
    xAxis: sizeAxis("总参数"),
    yAxis: baseAxis({ type: "log", logBase: 2, min: 4, max: 512, axisLabel: { color: css("--muted"), fontSize: 11, formatter: "{value} KiB" } }),
    series: scatterSeries(rows, (m) => m.total, (m) => m.kv_kib_per_token),
  }, true);

  chart("c-ctx").setOption({
    grid, tooltip: tooltip(),
    xAxis: timeAxis(),
    yAxis: baseAxis({ type: "log", logBase: 4, min: 4096, max: 16 * (1 << 20), axisLabel: { color: css("--muted"), fontSize: 11, formatter: fmtCtx } }),
    series: scatterSeries(rows, releaseTime, (m) => m.context),
  }, true);

  const legend = TIERS.map((t) => `<span><i style="background:${tierColor(t.id)}"></i>${t.label}</span>`).join("") +
    `<span><i style="background:${css("--ink-2")}"></i>实心 2026 年首发</span><span><i class="hollow"></i>空心 2025 年首发</span>`;
  document.querySelectorAll("#legend, .legend-copy").forEach((el) => { el.innerHTML = legend; });
}

function renderMultiples(rows) {
  const box = $("#multiples");
  for (const c of multCharts.splice(0)) c.dispose();
  const vendors = [...new Set(rows.map((m) => m.vendor))].sort();
  box.innerHTML = vendors.map((v, i) => `<div class="mult"><h4>${esc(v)}</h4><div class="m" id="mult-${i}"></div></div>`).join("");
  const minT = Date.UTC(2025, 0, 1);
  const maxT = Math.max(...DATA.map((m) => Date.parse(m.released + "-01"))) + 40 * 864e5;
  vendors.forEach((v, i) => {
    const ms = rows.filter((m) => m.vendor === v && m.E_over_k).sort((a, b) => a.released.localeCompare(b.released));
    const c = chart(`mult-${i}`, multCharts);
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

/* ---------- structural features ---------- */

// value(m) is what the table shows; has(m) is true / false / null (config not public).
const known = (m, v) => (m.config_ok ? v : null);
const FEATURES = [
  { key: "linear_attn", label: "线性注意力", th: "线性", title: "GDN、KDA、Lightning 或 Mamba-2 层" },
  { key: "window", label: "滑窗", title: "滑动窗口注意力层（窗口大小）" },
  { key: "sparse_attn", label: "稀疏注意力", th: "稀疏", title: "DSA、块稀疏或压缩注意力" },
  { key: "mla", label: "MLA", title: "多头潜注意力" },
  { key: "attn_gate", label: "输出门", title: "softmax 注意力输出上的 sigmoid 门" },
  { key: "attn_sink", label: "Sink", title: "可学习的 attention sink" },
  { key: "qk_norm", label: "QK-norm", title: "对每个头的 Q、K 做 RMSNorm" },
  { key: "hyper_conn", label: "超连接", title: "多路残差流（HC / mHC）" },
  { key: "attn_res", label: "AttnRes", title: "Attention Residuals：用注意力聚合前面各层的输出（块大小）" },
  { key: "ngram_mem", label: "N-gram 记忆", th: "N-gram", heat: "N-gram\n记忆", title: "哈希 n-gram 查表：Engram、层内注入或嵌入层" },
  { key: "mtp", label: "MTP", title: "多 token 预测层数" },
  { key: "router", label: "非 softmax 路由", th: "路由", heat: "非 softmax\n路由", title: "路由打分用 sigmoid 等，而不是 softmax",
    has: (m) => (m.router == null ? null : m.router !== "softmax"), value: (m) => m.router },
  { key: "expert_bias", label: "专家偏置", th: "偏置", title: "无辅助损失负载均衡的专家偏置" },
  { key: "group_routing", label: "分组路由", th: "分组", title: "先选专家组再选专家（组数/选中组数）" },
  { key: "s", label: "共享专家", th: "共享", title: "共享专家数" },
  { key: "latent", label: "LatentMoE", th: "Latent", title: "专家在低维潜空间里计算（潜空间维度）" },
];
for (const f of FEATURES) {
  f.value ??= (m) => m[f.key];
  f.has ??= (m) => known(m, Boolean(f.value(m)));
}
function featureCell(f, m) {
  const h = f.has(m);
  if (h == null) return `<span class="muted">?</span>`;
  const v = f.value(m);
  if (f.key === "router") return h ? esc(v) : `<span class="muted">${esc(v)}</span>`;
  if (!h) return `<span class="muted">·</span>`;
  return v === true ? "✓" : esc(v);
}

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

function renderAdoption(rows) {
  const box = $("#adoption");
  for (const c of adoptCharts.splice(0)) c.dispose();
  const qs = quarters();
  box.innerHTML = FEATURES.map((f, i) => {
    const ks = rows.filter((m) => f.has(m) != null);
    const n = ks.filter((m) => f.has(m)).length;
    return `<div class="mult"><h4 title="${esc(f.title)}">${esc(f.label)} <span class="muted">${n}/${ks.length}</span></h4><div class="m" id="adopt-${i}"></div></div>`;
  }).join("");
  FEATURES.forEach((f, i) => {
    const data = qs.map((q) => {
      const ms = rows.filter((m) => quarter(m) === q && f.has(m) != null);
      const yes = ms.filter((m) => f.has(m));
      const no = ms.filter((m) => !f.has(m));
      return { value: ms.length ? Math.round(100 * yes.length / ms.length) : null, q, yes, no, ms };
    });
    chart(`adopt-${i}`, adoptCharts).setOption({
      grid: { left: 34, right: 6, top: 10, bottom: 22 },
      tooltip: {
        ...tipStyle(),
        formatter: (p) => {
          const d = p.data;
          if (!d.ms.length) return `${d.q}：没有新配置`;
          const no = d.no;
          return `<b>${esc(f.label)}</b> · ${d.q}<br>${d.yes.length}/${d.ms.length}（${d.value}%）` +
            (d.yes.length ? `<br>用了：${names(d.yes)}` : "") + (no.length ? `<br><span style="color:${css("--muted")}">没用：${names(no)}</span>` : "");
        },
      },
      xAxis: baseAxis({ type: "category", data: qs, axisLabel: { color: css("--muted"), fontSize: 10, interval: 3, formatter: (q) => q.replace(" Q1", "") } }),
      yAxis: baseAxis({ type: "value", min: 0, max: 100, interval: 50, axisLabel: { color: css("--muted"), fontSize: 10, formatter: "{value}%" } }),
      series: [{
        type: "bar", barCategoryGap: "18%",
        itemStyle: { color: css("--accent"), borderRadius: [3, 3, 0, 0] },
        // keep empty quarters hoverable so the tooltip can say there was nothing new
        data: data.map((d) => ({ ...d, value: d.value ?? 0, itemStyle: d.value == null ? { color: "transparent" } : undefined })),
      }],
    }, true);
  });
}

function renderHeatmap(rows) {
  const box = $("#heat-mode");
  box.innerHTML = "";
  for (const [id, label] of [["vendor", "按厂家"], ["tier", "按档位"]]) {
    box.append(chip(label, state.heat === id, () => { state.heat = id; renderHeatmap(filtered()); }));
  }
  const groups = state.heat === "tier"
    ? TIERS.map((t) => ({ label: t.label, ms: rows.filter((m) => m.tier === t.id) })).filter((g) => g.ms.length)
    : [...new Set(rows.map((m) => m.vendor))].map((v) => ({ label: v, ms: rows.filter((m) => m.vendor === v) }))
      .sort((a, b) => b.ms.length - a.ms.length || a.label.localeCompare(b.label));
  const steps = [1, 2, 3, 4, 5].map((i) => css(`--seq-${i}`));
  const inkOn = [1, 2, 3, 4, 5].map((i) => css(`--seq-ink-${i}`));
  const data = [];
  groups.forEach((g, y) => FEATURES.forEach((f, x) => {
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
  const c = chart("c-heat");
  c.resize();
  c.setOption({
    grid: { left: 8, right: 8, top: 40, bottom: 4, containLabel: true },
    tooltip: {
      ...tipStyle(),
      formatter: (p) => {
        const { f, g, yes, no, ms } = p.data;
        if (!ms.length) return `<b>${esc(g.label)}</b> · ${esc(f.label)}<br>config 未公开`;
        return `<b>${esc(g.label)}</b> · ${esc(f.label)}<br>${yes.length}/${ms.length}` +
          (yes.length ? `<br>用了：${yes.map((m) => `${esc(m.name)}${f.value(m) === true ? "" : `（${esc(f.value(m))}）`}`).join("、")}` : "") +
          (no.length ? `<br><span style="color:${css("--muted")}">没用：${names(no)}</span>` : "");
      },
    },
    xAxis: { type: "category", position: "top", data: FEATURES.map((f) => f.heat ?? f.label), axisLine: { show: false }, axisTick: { show: false },
      splitLine: { show: false }, axisLabel: { color: css("--ink-2"), fontSize: 11, interval: 0, width: 56, overflow: "break", lineHeight: 13 } },
    yAxis: { type: "category", inverse: true, data: groups.map((g) => `${g.label} (${g.ms.length})`), axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: css("--ink-2"), fontSize: 11 } },
    series: [{
      type: "heatmap", data,
      label: { show: true, fontSize: 10, formatter: (p) => (!p.data.ms.length ? "?" : p.data.yes.length ? `${p.data.yes.length}/${p.data.ms.length}` : "") },
      emphasis: { itemStyle: { borderColor: css("--ink"), borderWidth: 1.5 } },
    }],
  }, true);
  $("#heat-legend").innerHTML = `<span class="muted">采用比例</span>` +
    ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"].map((t, i) => `<span><i style="background:${steps[i]}"></i>${t}</span>`).join("") +
    `<span><i style="background:${css("--page")};border:1px solid ${css("--ring")}"></i>没用</span>`;
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
    { key: "kv_heads", label: "KV 头", title: "GQA 的 KV 头数；MLA 缓存的是压缩潜向量，没有独立 KV 头", f: (m) => m.mla ? "MLA" : (m.kv_heads ?? "–") },
    { key: "window", label: "滑窗", f: (m) => m.window ?? "–" },
    { key: "context", label: "上下文", f: (m) => fmtCtx(m.context) },
  ] },
  attnFeat: { label: "注意力结构", cols: FEATURES.slice(0, 7) },
  residFeat: { label: "残差与路由", cols: FEATURES.slice(7) },
};
for (const g of [COL_GROUPS.attnFeat, COL_GROUPS.residFeat]) {
  g.cols = g.cols.map((f) => ({ key: f.key, label: f.th ?? f.label, title: f.title, f: (m) => featureCell(f, m),
    sort: (m) => { const h = f.has(m); return h == null ? null : +h; } }));
}
const cols = () => BASE_COLS.concat(COL_GROUPS[state.group].cols);

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
  renderAdoption(rows);
  renderHeatmap(rows);
  renderMultiples(rows);
  renderColGroups();
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
  addEventListener("resize", () => charts.concat(multCharts, adoptCharts).forEach((c) => c.resize()));
}

main().catch((e) => { $("#meta").textContent = `数据加载失败：${e.message}`; });
