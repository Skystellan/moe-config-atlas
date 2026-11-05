"use strict";

const TIERS = [
  { id: "S", label: "小（≤50B）" },
  { id: "M", label: "中（50–500B）" },
  { id: "L", label: "大（≥500B）" },
];
const state = {
  q: "", tiers: new Set(), years: new Set(), vendor: "", group: "sparsity", sort: { key: "latest", dir: -1 },
  fpMode: "model", fpGroup: "all", fpSort: { key: "vendor", dir: -1 }, compare: [], adoptGroup: "all", techCat: "all",
};
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
          (m.kv_kib_per_token ? `，KV ${num(m.kv_kib_per_token)} KiB/token` : "") + `，上下文 ${fmtCtx(m.context)}` : "") +
        (m.heads && m.qk_dim ? `<br>注意力：${m.heads} 头 × ${m.qk_dim} 维 = ${num(m.attn_ratio, 2)}×d，` +
          `Q/KV ${m.mla ? "MLA" : num(m.gqa)}，RoPE ${ropeLabel(m)}` : "");
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
  // a month of headroom so the last quarter's label isn't cut off at the right edge
  type: "time", min: Date.UTC(2025, 0, 1), max: (v) => v.max + 31 * 864e5,
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

  chart("c-attn").setOption({
    grid, tooltip: tooltip(),
    xAxis: timeAxis(),
    yAxis: baseAxis({ type: "log", logBase: 2, min: 0.5, max: 16, axisLabel: { color: css("--muted"), fontSize: 11, formatter: (v) => `${v}×` } }),
    series: scatterSeries(rows, releaseTime, (m) => m.attn_ratio),
  }, true);

  // Attention width (query heads x per-head QK dims) against d_model; dashed lines are equal ratios.
  const widthRefs = [1, 2, 4, 8].map((r) => ({
    type: "line", silent: true, symbol: "none", z: 1,
    lineStyle: { color: css("--axis"), width: 1, type: "dashed" },
    data: [[1024, 1024 * r], [16384, 16384 * r]],
    endLabel: { show: true, formatter: `${r}×`, color: css("--muted"), fontSize: 10 },
  }));
  const kFmt = (v) => (v >= 1024 ? `${v / 1024}K` : v);
  chart("c-heads").setOption({
    grid: { ...grid, right: 40 }, tooltip: tooltip(),
    xAxis: baseAxis({ type: "log", logBase: 2, min: 1024, max: 16384, name: "d_model", nameLocation: "middle", nameGap: 26,
      axisLabel: { color: css("--muted"), fontSize: 11, formatter: kFmt } }),
    yAxis: baseAxis({ type: "log", logBase: 2, min: 1024, max: 131072, axisLabel: { color: css("--muted"), fontSize: 11, formatter: kFmt } }),
    series: widthRefs.concat(scatterSeries(rows, (m) => m.d_model, (m) => (m.heads && m.qk_dim ? m.heads * m.qk_dim : null))),
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
  for (const c of SHAPE_COLS) {
    const v = DATA.map((m) => m[c.key]).filter((x) => x != null).sort((a, b) => a - b);
    bins[c.key] = [0.2, 0.4, 0.6, 0.8].map((q) => v[Math.min(v.length - 1, Math.floor(q * v.length))]);
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

function renderAdoption(rows) {
  groupChips($("#adopt-groups"), state.adoptGroup, [{ id: "all", label: "全部" }, ...FEATURE_GROUPS],
    (id) => { state.adoptGroup = id; renderAdoption(filtered()); });
  const box = $("#adoption");
  for (const c of adoptCharts.splice(0)) c.dispose();
  const qs = quarters();
  const feats = shownFeatures(state.adoptGroup);
  box.innerHTML = feats.map((f, i) => {
    const ks = rows.filter((m) => f.has(m) != null);
    const n = ks.filter((m) => f.has(m)).length;
    return `<div class="mult"><h4 title="${esc(f.title)}">${esc(f.label)} <span class="muted">${n}/${ks.length}</span></h4><div class="m" id="adopt-${i}"></div></div>`;
  }).join("");
  feats.forEach((f, i) => {
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
          return `<b>${esc(f.label)}</b> · ${d.q}<br>${d.yes.length}/${d.ms.length}（${d.value}%）` +
            (d.yes.length ? `<br>用了：${names(d.yes)}` : "") + (d.no.length ? `<br><span style="color:${css("--muted")}">没用：${names(d.no)}</span>` : "");
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
  if (key === "released") return rows.slice().sort((a, b) => b.released.localeCompare(a.released) || b.total - a.total);
  // vendors with more configs first, each vendor oldest to newest
  const count = {};
  for (const m of rows) count[m.vendor] = (count[m.vendor] || 0) + 1;
  return rows.slice().sort((a, b) => count[b.vendor] - count[a.vendor] || a.vendor.localeCompare(b.vendor) ||
    a.released.localeCompare(b.released) || a.total - b.total);
}

function renderMatrix(rows) {
  const g = state.fpGroup;
  const shape = g === "all" || g === "shape" ? SHAPE_COLS : [];
  const feats = g === "shape" ? [] : shownFeatures(g);
  const cols = shape.map((c, i) => ({ ...c, num: true, gs: i === 0 })).concat(feats.map((f, i) =>
    ({ key: f.key, label: f.th ?? f.label, title: f.title, f, gs: i === 0 || feats[i - 1].group !== f.group })));
  const heads = (shape.length ? [{ label: "形状", n: shape.length }] : []).concat(
    FEATURE_GROUPS.map((fg) => ({ label: fg.label, n: feats.filter((f) => f.group === fg.id).length })).filter((x) => x.n));

  const col = cols.find((c) => c.key === state.fpSort.key);
  const sorted = sortMatrix(rows, col);
  const grouped = !col && state.fpSort.key !== "released";

  const sortBox = $("#fp-sort");
  sortBox.innerHTML = `<span class="muted">排序</span>`;
  for (const [id, label] of [["vendor", "按厂家"], ["released", "按时间"]]) {
    const on = id === "vendor" ? grouped : !col && state.fpSort.key === id;
    sortBox.append(chip(label, on, () => { state.fpSort = { key: id, dir: -1 }; renderMatrix(filtered()); }));
  }
  if (col) sortBox.append(chip(`按「${col.label}」${state.fpSort.dir > 0 ? "↑" : "↓"}`, true, () => { state.fpSort.dir *= -1; renderMatrix(filtered()); }));

  const cell = (c, m) => {
    const gs = c.gs ? " gs" : "";
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
      <tr class="ch">${cols.map((c) => `<th class="${c.gs ? "gs" : ""}" data-key="${c.key}" aria-sort="${sortAttr(c)}" title="${esc(c.title)}（点击排序）">${esc(c.label)}</th>`).join("")}</tr>
    </thead>
    <tbody>${sorted.map((m, i) => {
      const first = grouped && i > 0 && sorted[i - 1].vendor !== m.vendor;
      const on = state.compare.includes(m.name);
      return `<tr class="${first ? "grp" : ""}${on ? " picked" : ""}">
        <th class="l" scope="row"><div class="rh"><button type="button" class="pick" aria-pressed="${on}" data-name="${esc(m.name)}" title="${on ? "移出对比" : "加入下面的并排对比"}">${on ? "✓" : "+"}</button><span><span class="nm"><span class="tier-dot" style="background:${tierColor(m.tier)}"></span>${esc(m.name)}</span><small>${esc(m.vendor)} · ${m.released}</small></span></div></th>
        ${cols.map((c) => cell(c, m)).join("")}</tr>`;
    }).join("")}</tbody>`;
  $("#fp-table").querySelectorAll("th[data-key]").forEach((th) => th.onclick = () => {
    const k = th.dataset.key;
    state.fpSort = { key: k, dir: state.fpSort.key === k ? -state.fpSort.dir : -1 };
    renderMatrix(filtered());
  });
  $("#fp-table").querySelectorAll("button.pick").forEach((b) => b.onclick = () => togglePick(b.dataset.name));
  $("#fp-legend").innerHTML = `<span><i class="sw on"></i>用了（格子里是取值）</span><span><i class="sw off"></i>没用</span><span><i class="sw na"></i>? 未知</span>` +
    (shape.length ? `<span>形状数值按全部配置的五分位上色：小<span class="ramp">${[1, 2, 3, 4, 5].map((i) => `<i class="sw q${i}"></i>`).join("")}</span>大</span>` : "");
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
const CMP_SECTIONS = [
  { label: "规模", rows: [
    { label: "厂家 · 首发", f: (m) => `${esc(m.vendor)} · ${m.released}` },
    { label: "总参数", f: (m) => fmtParams(m.total), v: (m) => m.total },
    { label: "激活参数", f: (m) => fmtParams(m.active), v: (m) => m.active },
    { label: "参数激活比", f: (m) => pct(m.param_activation), v: (m) => m.param_activation },
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
  const table = $("#cmp");
  if (!ms.length) {
    table.innerHTML = `<tbody><tr><td class="l muted">还没选型号。在上面的结构指纹里点 +，或在输入框里输入型号。</td></tr></tbody>`;
    return;
  }
  const strip = (h) => String(h).replace(/<[^>]*>/g, "");
  const onlyDiff = $("#cmp-diff").checked;
  let body = "";
  for (const sec of CMP_SECTIONS) {
    const rowsHtml = sec.rows.map((r) => {
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
    return `<tr>
      <td class="l"><b>${esc(t.name)}</b><div class="muted cat">${esc(CAT_LABEL[t.cat])}${t.note ? ` · ${esc(t.note)}` : ""}</div></td>
      <td class="l src"><span class="date">${esc(t.date)}</span>${link(t.source.url, esc(t.source.title))}</td>
      <td class="l ev">${ev}</td>
      <td class="l use">${use}</td></tr>`;
  }).join("");
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
  renderFingerprint(rows);
  renderCompare();
  renderAdoption(rows);
  renderTechniques(rows);
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
  const [res, tres] = await Promise.all([fetch("data/models.json"), fetch("data/techniques.json")]);
  const data = await res.json();
  DATA = data.models;
  TECHS = tres.ok ? await tres.json() : [];
  computeBins();
  initCompare();
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
