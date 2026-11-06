"""Fetch config.json for every model in models.yaml and write the site data.

    python scripts/build.py            # uses .cache/ when present
    python scripts/build.py --refresh  # re-download everything

Set HF_TOKEN to read gated repos. The leaderboard scores (scripts/bench.py) need no key.
Writes site/data/models.json and models.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402
import hfconfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
LOCAL_CONFIGS = ROOT / "configs"  # configs of gated repos, with their source in "_source" (and "_mirror")
OUT = ROOT / "site" / "data"
HF = "https://huggingface.co"


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "moe-config-atlas"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some vendors ship config.json with trailing commas or NaN.
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def cached(name, fetch, refresh):
    path = CACHE / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    try:
        data = fetch()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  ! {name}: {e}", file=sys.stderr)
        return json.loads(path.read_text()) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))
    return data


def fetch_config(repo, refresh):
    """Return (config, source): source is "hf" or the "_source" note of a local copy."""
    key = repo.replace("/", "__")
    local = LOCAL_CONFIGS / f"{key}.json"
    if local.exists():
        cfg = json.loads(local.read_text())
        return cfg, cfg.get("_source", "local")
    cfg = cached(f"{key}.config.json", lambda: http_json(f"{HF}/{repo}/resolve/main/config.json"), refresh)
    if cfg is None:  # Mistral ships params.json instead
        cfg = cached(f"{key}.params.json", lambda: http_json(f"{HF}/{repo}/resolve/main/params.json"), refresh)
    return cfg, "hf" if cfg is not None else None


def fetch_info(repo, refresh):
    key = repo.replace("/", "__")
    return cached(f"{key}.info.json", lambda: http_json(f"{HF}/api/models/{repo}?expand[]=createdAt&expand[]=safetensors&expand[]=cardData"), refresh) or {}


def fetch_index_mtp(repo, refresh):
    """MTP layers found in the weight index, for configs that don't declare them (MiMo, Qwen3-Next)."""
    key = repo.replace("/", "__")

    def count():
        try:
            names = http_json(f"{HF}/{repo}/resolve/main/model.safetensors.index.json").get("weight_map", {})
        except urllib.error.HTTPError as e:
            if e.code == 404:  # single-file checkpoint
                return {"mtp": 0}
            raise
        return {"mtp": len({int(n) for w in names for n in re.findall(r"(?:^|\.)mtp\.layers\.(\d+)\.", w)})
                or (1 if any(w.startswith("mtp.") or ".mtp." in w for w in names) else 0)}

    return (cached(f"{key}.mtp.json", count, refresh) or {}).get("mtp")


def precision(raw, info):
    """Weight format of the main repo as (format, detail): FP8 with its block size and scale format, FP4 or
    MXFP4 experts, else the stored dtype."""
    if not raw:
        return None, None
    c = hfconfig.text_config(raw)
    q = raw.get("quantization_config") or c.get("quantization_config") or {}
    groups = [g.get("weights") or {} for g in (q.get("config_groups") or {}).values()]
    block = q.get("weight_block_size") or next((g.get("block_structure") for g in groups if g.get("block_structure")), None)
    fp8 = ("块 " + "×".join(map(str, block)) if block else "逐张量") + ("，ue8m0 缩放" if q.get("scale_fmt") == "ue8m0" else "")
    if "fp4" in (q.get("expert_dtype"), c.get("expert_dtype")):  # DeepSeek V4: FP4 experts, FP8 elsewhere
        return "FP4 专家 + FP8", fp8
    if str(q.get("moe_quant_algo") or "").upper() == "NVFP4":  # MiniMax-M3.1-Flash: NVFP4 experts, MXFP8 elsewhere
        return "NVFP4 专家 + " + ("MXFP8" if q.get("quant_method") == "mxfp8" else "FP8"), fp8
    if q.get("quant_method") == "mxfp8":  # Raon-1 mini: only the routed experts, attention and the rest stay BF16
        experts = any("self_attn" in str(x) for x in q.get("ignored_layers") or q.get("ignore") or [])
        return ("MXFP8 专家" if experts else "MXFP8"), "每 32 个数一个缩放"
    int4 = [g for g in groups if g.get("num_bits") == 4 and g.get("type") == "int"]
    if q.get("quant_method") == "compressed-tensors" and int4:  # Kimi K2 Thinking: INT4 routed experts (QAT)
        size = int4[0].get("group_size")
        experts = any("self_attn" in str(x) for x in q.get("ignore") or [])  # attention and shared experts stay BF16
        return "INT4 专家" if experts else "INT4", f"每 {size} 个数一个缩放" if size else None
    fmt = str(q.get("format") or "")
    if q.get("quant_method") == "mxfp4" or "mxfp4" in fmt:  # only the routed experts are quantized
        return "MXFP4 专家", None
    if q.get("quant_method") == "fp8" or (fmt == "float-quantized" and any(g.get("num_bits") == 8 for g in groups)):
        return "FP8", fp8
    params = (info.get("safetensors") or {}).get("parameters") or {}
    if params:
        top = max(params, key=params.get)
        return {"BF16": "BF16", "F32": "FP32", "F16": "FP16"}.get(top, top), None
    # HF indexes no safetensors metadata for some layouts (Grok 2's tensor-parallel shards): use the config's dtype
    dtype = str(raw.get("torch_dtype") or raw.get("dtype") or c.get("torch_dtype") or "")
    return {"bfloat16": "BF16", "float32": "FP32", "float16": "FP16"}.get(dtype), None


def tier(total):
    return "S" if total <= 50 else ("M" if total < 500 else "L")


def r(x, n=1):
    return None if x is None else round(x, n)


# Problems with how models.yaml groups versions; main() lists them all and stops before writing anything.
PROBLEMS = []
ARCHS = {}  # (vendor, architecture) -> row name: one vendor's versions of one architecture share a row
NEWER_OK = dt.timedelta(days=31)  # a sibling (Instruct / Thinking) may follow the row's repo by up to a month


def check_versions(m, parsed, versions, refresh):
    """A row holds one architecture under the name of its latest mainline version: every version on HF must
    parse to the repo's architecture signature, and no earlier mainline version may be much newer than the repo."""
    want = hfconfig.arch_signature(parsed)
    for v in versions[1:]:
        if not v["repo"]:
            continue
        raw, _ = fetch_config(v["repo"], refresh)
        if raw is None:
            print(f"  · {v['repo']}: config.json unreadable, architecture not checked", file=sys.stderr)
            continue
        got = hfconfig.parse(raw)
        got.update(m.get("override") or {})
        got = hfconfig.arch_signature(got)
        diff = [f"{k}: {want.get(k)!r} -> {got.get(k)!r}" for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)]
        if diff:
            PROBLEMS.append(f"{m['name']}: {v['repo']} has another architecture ({'; '.join(diff[:6])}); give it its own row")
    top = versions[0]["created"]
    for v in versions[1:]:
        if v["repo"] and not v.get("derived") and top and v["created"] and \
                dt.date.fromisoformat(v["created"]) - dt.date.fromisoformat(top) > NEWER_OK:
            PROBLEMS.append(f"{m['name']}: variant {v['repo']} ({v['created']}) is newer than the repo {m['repo']} "
                            f"({top}); make the latest version the repo and the name, or list it under derived")


def build_entry(m, refresh):
    repo = m["repo"]
    raw, config_source = fetch_config(repo, refresh)
    parsed = hfconfig.parse(raw) if raw else {f: None for f in hfconfig.ALIASES}
    parsed.update(m.get("override") or {})
    info = fetch_info(repo, refresh)

    # Every upload of this architecture: the main repo, the earlier mainline versions (variants), and the models
    # made from it for one use or copied into another format (derived).
    versions = [{"name": repo.split("/")[-1], "repo": repo, "created": (info.get("createdAt") or "")[:10] or None}]
    for v in m.get("variants") or []:
        if isinstance(v, dict):  # a version whose weights are not on HF yet, e.g. {name, released, note}
            versions.append({"name": v["name"], "repo": None, "created": str(v.get("released") or "") or None,
                             "note": v.get("note")})
            continue
        created = (fetch_info(v, refresh).get("createdAt") or "")[:10] or None if "/" in v else None
        versions.append({"name": v.split("/")[-1], "repo": v if "/" in v else None, "created": created})
    for v in m.get("derived") or []:
        created = (fetch_info(v, refresh).get("createdAt") or "")[:10] or None
        versions.append({"name": v.split("/")[-1], "repo": v, "created": created, "derived": True})
    if raw:
        arch = (m["vendor"], json.dumps(hfconfig.arch_signature(parsed), sort_keys=True, default=str))
        if arch in ARCHS:
            PROBLEMS.append(f"{ARCHS[arch]} and {m['name']} have the same architecture; make them one row")
        ARCHS[arch] = m["name"]
        check_versions(m, parsed, versions, refresh)
    dated = sorted((x for x in versions if x["created"]), key=lambda x: x["created"], reverse=True)
    latest = dated[0] if dated else versions[0]
    # the mainline version this architecture first shipped as (list order breaks ties, e.g. R1 before R1-Zero)
    mainline = [x for x in versions if not x.get("derived")]
    first = min((x for x in mainline if x["created"]), key=lambda x: x["created"], default=versions[0])

    E, k, s = parsed.get("E"), parsed.get("k"), parsed.get("s") or 0
    d, de = parsed.get("d_model"), parsed.get("d_expert")
    total, active = m["total"], m["active"]
    latent = parsed.get("latent")
    entry = {
        "vendor": m["vendor"],
        "name": m["name"],
        "repo": repo,
        "variants": [v["name"] if isinstance(v, dict) else v for v in m.get("variants") or []],
        "derived": list(m.get("derived") or []),
        "released": str(m["released"]),
        "year": int(str(m["released"])[:4]),
        "total": total,
        "active": active,
        "tier": tier(total),
        "E": E, "k": k, "s": s,
        "E_over_k": r(E / k, 1) if E and k else None,
        "expert_activation": r(100 * (k + s) / (E + s), 1) if E and k else None,
        "param_activation": r(100 * active / total, 1),
        "granularity": r(2 * d / de, 1) if d and de and not latent else None,
        "d_model": d,
        "d_expert": de,
        "d_shared": parsed.get("d_shared") or None,
        "latent": latent,
        "zero_experts": parsed.get("zero_experts"),
        "layers": parsed.get("layers"),
        "mixer_layers": parsed.get("mixer_layers"),
        "aspect": parsed.get("aspect"),
        "global_attn_frac": parsed.get("global_attn_frac"),
        "kv_kib_per_token": parsed.get("kv_kib_per_token"),
        "window": parsed.get("window"),
        "window_sink": parsed.get("window_sink"),
        "mla": parsed.get("mla", False),
        "kv_heads": parsed.get("kv_heads"),
        "head_dim": parsed.get("head_dim"),
        "heads": parsed.get("heads"),
        "qk_dim": parsed.get("qk_dim"),
        "attn_ratio": parsed.get("attn_ratio"),
        "gqa": parsed.get("gqa"),
        "rope_frac": r(parsed.get("rope_frac"), 3),
        "vocab": parsed.get("vocab"),
        "context": parsed.get("context"),
        # a gated repo's weight index can be read from the public copy its local config names in "_mirror"
        "mtp": (parsed.get("mtp") or fetch_index_mtp(raw.get("_mirror") or repo, refresh)) if raw else None,
        "moe_layers": parsed.get("moe_layers"),
        "attention": parsed.get("attention") if raw else None,
        "layout": parsed.get("layout") if raw else None,
        "dense_layers": parsed.get("dense_layers") if raw else None,
        **{f: parsed.get(f) if raw else None for f in hfconfig.FEATURES},
        **{f: parsed.get(f) if raw else None for f in hfconfig.DETAILS},
        **dict(zip(("precision", "quant"), precision(raw, info))),
        "model_type": parsed.get("model_type"),
        "per_layer": parsed.get("per_layer", False),
        "config_ok": raw is not None,
        "config_source": None if config_source == "hf" else config_source,
        "hf_created": (info.get("createdAt") or "")[:10] or None,
        "latest": latest["created"],
        "latest_name": latest["name"],
        "first_name": first["name"],
        "versions": dated + [x for x in versions if not x["created"]],
        "hf_params": r((info.get("safetensors") or {}).get("total", 0) / 1e9, 1) or None,
        "license": (info.get("cardData") or {}).get("license"),
        "notes": m.get("notes"),
    }
    return entry


TECH_CATS = {"attn", "pos", "mixer", "resid", "moe", "io", "optim"}
TECH_FIELDS = set(hfconfig.FEATURES) | set(hfconfig.DETAILS) | {"mla", "mtp", "latent", "window", "s"}


def write_techniques(rows):
    """Check techniques.yaml against the model data and copy it to the site."""
    techs = yaml.safe_load((ROOT / "techniques.yaml").read_text())
    names = {x["name"] for x in rows}
    for t in techs:
        where = f"techniques.yaml: {t.get('name')}"
        assert t.get("cat") in TECH_CATS, f"{where}: unknown cat {t.get('cat')}"
        assert re.fullmatch(r"\d{4}-\d{2}", str(t.get("date"))), f"{where}: date must be YYYY-MM"
        assert len(t.get("source") or []) == 2, f"{where}: source is [title, url]"
        assert not t.get("feature") or t["feature"] in TECH_FIELDS, f"{where}: unknown feature {t['feature']}"
        for m in t.get("models") or []:
            assert m["name"] in names and m.get("cite"), f"{where}: adopter {m['name']} needs a tracked model and a cite"
        t["date"] = str(t["date"])
        t["source"] = {"title": t["source"][0], "url": t["source"][1]}
    (OUT / "techniques.json").write_text(json.dumps(techs, ensure_ascii=False, indent=1))


DETAIL_SECS = {"attn", "linear", "moe", "resid", "io"}


def _truthy(v):
    return v not in (None, False, "", 0, [])


def _wild(want, v):
    return re.fullmatch(".*".join(map(re.escape, want.split("*"))), str(v)) is not None


def detail_ok(v, want):
    """One field of a details.yaml match; site/app.js detailOk mirrors this."""
    if isinstance(want, list):
        return any(detail_ok(v, w) for w in want)
    if want is None:
        return v is None
    if isinstance(v, list):
        return not v if want is False else any(detail_ok(x, want) for x in v)
    if want == "*":
        return _truthy(v)
    if want is False or want is True:
        return v is want
    if isinstance(want, str) and "*" in want:
        return _truthy(v) and _wild(want, v)
    return v == want if isinstance(want, (int, float)) else str(v) == str(want)


def write_details(rows):
    """Check that every model sits under a variant of each details.yaml position, then copy it to the site."""
    positions = yaml.safe_load((ROOT / "details.yaml").read_text())
    match = lambda m, cond: all(detail_ok(m.get(k), w) for k, w in (cond or {}).items())
    ids = set()
    for p in positions:
        where = f"details.yaml: {p.get('id')}"
        assert p.get("id") and p["id"] not in ids, f"{where}: missing or repeated id"
        ids.add(p["id"])
        assert p.get("sec") in DETAIL_SECS and p.get("name") and p.get("variants"), f"{where}: sec, name and variants"
        for v in p["variants"]:
            assert v.get("label") and (v.get("related") or v.get("match")), f"{where}: {v.get('label')} needs a match"
            for k in list((v.get("match") or {})) + list(p.get("when") or {}):
                assert k in rows[0], f"{where}: unknown field {k}"
            if v.get("source"):
                assert len(v["source"]) == 2, f"{where}: source is [title, url]"
                v["source"] = {"title": v["source"][0], "url": v["source"][1]}
        used = {i: [] for i in range(len(p["variants"]))}
        for m in rows:
            if not match(m, p.get("when")):
                continue
            hits = [i for i, v in enumerate(p["variants"]) if not v.get("related") and match(m, v["match"])]
            assert hits, f"{where}: {m['name']} fits no variant"
            for i in hits if p.get("multi") else hits[:1]:
                used[i].append(m["name"])
        for i, v in enumerate(p["variants"]):
            assert v.get("related") or used[i], f"{where}: no model uses {v['label']}"
    (OUT / "details.json").write_text(json.dumps(positions, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    models = yaml.safe_load((ROOT / "models.yaml").read_text())
    rows = []
    for m in models:
        print(f"- {m['repo']}")
        if not m.get("total") or not m.get("active"):
            print("  ! skipped: total / active parameters missing in models.yaml", file=sys.stderr)
            continue
        rows.append(build_entry(m, args.refresh))
    if PROBLEMS:
        sys.exit("models.yaml groups versions wrongly:\n- " + "\n- ".join(PROBLEMS))
    rows.sort(key=lambda x: (x["vendor"], x["released"], x["total"]))

    OUT.mkdir(parents=True, exist_ok=True)
    prev = OUT / "models.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}

    print("- leaderboards")
    aliases = yaml.safe_load((ROOT / "bench.yaml").read_text()) or {}
    unknown = set(aliases) - {x["name"] for x in rows}
    if unknown:
        sys.exit(f"bench.yaml names models that are not in models.yaml: {', '.join(sorted(unknown))}")
    boards = bench.load(cached, args.refresh)
    scores = bench.attach(rows, boards, aliases, {x["name"]: x for x in old.get("models", [])}, old.get("bench"))
    for src, s in scores.items():
        print(f"  {s['label']}: {s['n']} of {len(rows)} configurations scored (leaderboard of {s['date']})")
    scores = bench.composite(rows, scores)
    if "avg" in scores:
        a = scores["avg"]
        base = f"the {a['common']} configurations every board scored" if a["common"] else "each board's own configurations"
        print(f"  normalized average of {', '.join(scores[s]['label'] for s in a['sources'])}: "
              f"{a['n']} configurations, standardized on {base}")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if old.get("models") == rows:  # keep the old timestamp when nothing changed, so the data commit stays empty
        now = old.get("generated", now)
    payload = {
        "generated": now,
        "count": len(rows),
        "bench": scores,
        "models": rows,
    }
    (OUT / "models.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    cols = [c for c in rows[0] if c not in ("variants", "derived", "versions")] + ["variants", "derived"]
    with open(OUT / "models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: "; ".join(map(str, row[c])) if isinstance(row[c], list) else row[c] for c in cols})

    write_techniques(rows)
    write_details(rows)

    missing = [x["name"] for x in rows if not x["config_ok"]]
    print(f"\n{len(rows)} models written to {OUT}")
    if missing:
        print("config.json unavailable (override used):", ", ".join(missing))


if __name__ == "__main__":
    main()
