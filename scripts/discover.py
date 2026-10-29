"""Scan vendor orgs on Hugging Face for MoE checkpoints not yet in models.yaml.

New repos whose MoE shape matches a tracked entry are appended to that entry's
`variants`; repos with a new shape are appended as new entries marked
a guessed size. The weekly workflow opens a pull request with the result, so
nothing is published until it is merged.

Repos already looked at (dense, MoE, or proposed) are remembered in
discover_state.json, so deleting a proposed entry from the PR before merging
keeps it from coming back.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
from pathlib import Path

from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).parent))
import hfconfig  # noqa: E402
from build import HF, http_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "discover_state.json"
FIELDS = ("E", "k", "s", "d_model", "layers", "d_expert")


def fingerprint(p):
    return tuple(p.get(f) for f in FIELDS)


def list_org(org, since):
    url = (f"{HF}/api/models?author={org}&sort=createdAt&direction=-1&limit=500"
           "&expand[]=createdAt&expand[]=safetensors&expand[]=tags")
    return [m for m in http_json(url) if (m.get("createdAt") or "") >= since]


def base_models(info):
    """Repos named in base_model:* tags (finetune, adapter, quantized, merge)."""
    return {t.split(":")[-1] for t in info.get("tags") or [] if t.startswith("base_model:")}


def compile_patterns(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


def size_from_name(name, pattern):
    m = re.search(pattern, name, re.I)
    if not m:
        return None
    val = float(m.group(1))
    return val * 1000 if m.group(2).upper() == "T" else val


def get_config(repo):
    for fname in ("config.json", "params.json"):
        try:
            return http_json(f"{HF}/{repo}/resolve/main/{fname}")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return "gated"
        except (urllib.error.URLError, ValueError, TimeoutError):
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-unreachable", action="store_true",
                    help="re-try repos whose config.json could not be read before (e.g. after setting HF_TOKEN)")
    args = ap.parse_args()
    seen = ("dense", "moe", "proposed", "skipped") + (() if args.retry_unreachable else ("unreachable",))

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    models_path = ROOT / "models.yaml"
    models = yaml.load(models_path.read_text())
    conf = yaml.load((ROOT / "discover.yaml").read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    skip = compile_patterns(conf["skip_patterns"] + conf["non_llm_patterns"])
    variant_skip = compile_patterns(conf["variant_skip_patterns"])

    known = {}  # repo -> entry it belongs to
    shapes = {}
    for m in models:
        known[m["repo"]] = m
        known.update({v: m for v in m.get("variants") or [] if isinstance(v, str) and "/" in v})
        raw = None
        cache = ROOT / ".cache" / (m["repo"].replace("/", "__") + ".config.json")
        if cache.exists():
            raw = json.loads(cache.read_text())
        else:
            got = get_config(m["repo"])
            raw = got if isinstance(got, dict) else None
        if raw:
            shapes[(m["vendor"], fingerprint(hfconfig.parse(raw)))] = m

    added, new = [], []
    for org, vendor in conf["orgs"].items():
        try:
            repos = list_org(org, str(conf["since"]))
        except urllib.error.URLError as e:
            print(f"! {org}: {e}", file=sys.stderr)
            continue
        for info in reversed(repos):  # oldest first, so the first upload becomes the main repo
            repo = info["id"]
            if repo in known or state.get(repo) in seen:
                continue
            if skip.search(repo):
                state[repo] = "skipped"
                continue
            bases = base_models(info) - {repo}
            parent = next((known[b] for b in bases if b in known), None)
            if parent is not None and parent["vendor"] != vendor:
                state[repo] = "skipped"  # another vendor's fine-tune of a tracked model
                continue
            if bases and parent is None and not any(b.split("/")[0] == org for b in bases):
                state[repo] = "skipped"  # fine-tune of another vendor's model
                continue
            raw = get_config(repo)
            if raw == "gated" or raw is None:
                state[repo] = "unreachable"
                continue
            if not hfconfig.is_moe(raw):
                state[repo] = "dense"
                continue
            p = hfconfig.parse(raw)
            state[repo] = "proposed"
            match = parent or shapes.get((vendor, fingerprint(p)))
            if match is not None:
                known[repo] = match
                if variant_skip.search(repo.split("/", 1)[1]):
                    state[repo] = "skipped"
                    continue
                match.setdefault("variants", [])
                match["variants"].append(repo)
                added.append(f"{repo} → variant of {match['name']}")
                continue
            name = repo.split("/", 1)[1]
            total = ((info.get("safetensors") or {}).get("total") or 0) / 1e9 or None
            entry = {
                "vendor": vendor,
                "name": name,
                "repo": repo,
                "released": info["createdAt"][:7],
                "total": round(total, 1) if total else size_from_name(name, r"(\d+(?:\.\d+)?)([BT])(?![a-z])"),
                "active": size_from_name(name, r"A(\d+(?:\.\d+)?)(B)"),
            }
            models.append(entry)
            shapes[(vendor, fingerprint(p))] = entry
            known[repo] = entry
            new.append(f"{repo}: {p['E']}/{p['k']}+{p['s']}, total≈{entry['total']}B, active={entry['active']}B")

    if new or added:
        with open(models_path, "w") as f:
            yaml.dump(models, f)
        STATE.write_text(json.dumps(dict(sorted(state.items())), indent=1))

    report = ROOT / ".cache" / "discover-report.md"
    report.parent.mkdir(exist_ok=True)
    lines = [
        "每周自动扫描 `discover.yaml` 里的厂家组织，下面是 `models.yaml` 里还没有的 MoE 检查点。",
        "",
        "审核方法：新配置的 `total` / `active` 是从权重统计和仓库名猜的，请对照模型卡改准，`released` 是首次上传的年月；"
        "不想收录的条目直接在这个 PR 里删掉，`discover_state.json` 已经记下它们，以后不会再提。合并后网站自动更新。",
        "",
    ]
    if new:
        lines += ["### 新的 MoE 配置", ""] + [f"- {x}" for x in new] + [""]
    if added:
        lines += ["### 与已收录模型配置相同（加为同配置版本）", ""] + [f"- {x}" for x in added] + [""]
    report.write_text("\n".join(lines))
    print(report.read_text())

    found = bool(new or added)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"found={'true' if found else 'false'}\n")


if __name__ == "__main__":
    main()
