"""Fetch config.json for every model in models.yaml and write the site data.

    python scripts/build.py            # uses .cache/ when present
    python scripts/build.py --refresh  # re-download everything

Set HF_TOKEN to read gated repos. Writes site/data/models.json and models.csv.
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
import hfconfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
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
    key = repo.replace("/", "__")
    cfg = cached(f"{key}.config.json", lambda: http_json(f"{HF}/{repo}/resolve/main/config.json"), refresh)
    if cfg is None:  # Mistral ships params.json instead
        cfg = cached(f"{key}.params.json", lambda: http_json(f"{HF}/{repo}/resolve/main/params.json"), refresh)
    return cfg


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


def tier(total):
    return "S" if total <= 50 else ("M" if total < 500 else "L")


def r(x, n=1):
    return None if x is None else round(x, n)


def build_entry(m, refresh):
    repo = m["repo"]
    raw = fetch_config(repo, refresh)
    parsed = hfconfig.parse(raw) if raw else {f: None for f in hfconfig.ALIASES}
    parsed.update(m.get("override") or {})
    info = fetch_info(repo, refresh)

    # Every upload of this configuration (main repo + variants), newest first.
    versions = [{"name": repo.split("/")[-1], "repo": repo, "created": (info.get("createdAt") or "")[:10] or None}]
    for v in m.get("variants") or []:
        if isinstance(v, dict):  # a version whose weights are not on HF yet, e.g. {name, released, note}
            versions.append({"name": v["name"], "repo": None, "created": str(v.get("released") or "") or None,
                             "note": v.get("note")})
            continue
        created = (fetch_info(v, refresh).get("createdAt") or "")[:10] or None if "/" in v else None
        versions.append({"name": v.split("/")[-1], "repo": v if "/" in v else None, "created": created})
    dated = sorted((x for x in versions if x["created"]), key=lambda x: x["created"], reverse=True)
    latest = dated[0] if dated else versions[0]

    E, k, s = parsed.get("E"), parsed.get("k"), parsed.get("s") or 0
    d, de = parsed.get("d_model"), parsed.get("d_expert")
    total, active = m["total"], m["active"]
    latent = parsed.get("latent")
    entry = {
        "vendor": m["vendor"],
        "name": m["name"],
        "repo": repo,
        "variants": [v["name"] if isinstance(v, dict) else v for v in m.get("variants") or []],
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
        "mla": parsed.get("mla", False),
        "kv_heads": parsed.get("kv_heads"),
        "head_dim": parsed.get("head_dim"),
        "vocab": parsed.get("vocab"),
        "context": parsed.get("context"),
        "mtp": (parsed.get("mtp") or fetch_index_mtp(repo, refresh)) if raw else None,
        "moe_layers": parsed.get("moe_layers"),
        "attention": parsed.get("attention") if raw else None,
        **{f: parsed.get(f) if raw else None for f in hfconfig.FEATURES},
        "model_type": parsed.get("model_type"),
        "per_layer": parsed.get("per_layer", False),
        "config_ok": raw is not None,
        "hf_created": (info.get("createdAt") or "")[:10] or None,
        "latest": latest["created"],
        "latest_name": latest["name"],
        "versions": dated + [x for x in versions if not x["created"]],
        "hf_params": r((info.get("safetensors") or {}).get("total", 0) / 1e9, 1) or None,
        "license": (info.get("cardData") or {}).get("license"),
        "notes": m.get("notes"),
    }
    return entry


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
    rows.sort(key=lambda x: (x["vendor"], x["released"], x["total"]))

    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prev = OUT / "models.json"
    if prev.exists():  # keep the old timestamp when nothing changed, so the data commit stays empty
        old = json.loads(prev.read_text())
        if old.get("models") == rows:
            now = old.get("generated", now)
    payload = {
        "generated": now,
        "count": len(rows),
        "models": rows,
    }
    (OUT / "models.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    cols = [c for c in rows[0] if c not in ("variants", "versions")]
    with open(OUT / "models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["variants"])
        w.writeheader()
        for row in rows:
            w.writerow({**{c: row[c] for c in cols}, "variants": "; ".join(row["variants"])})

    missing = [x["name"] for x in rows if not x["config_ok"]]
    print(f"\n{len(rows)} models written to {OUT}")
    if missing:
        print("config.json unavailable (override used):", ", ".join(missing))


if __name__ == "__main__":
    main()
