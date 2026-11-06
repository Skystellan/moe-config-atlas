"""Leaderboard scores for each configuration, used to sort the site by how well the models do.

Three public sources:

- Epoch AI's Capabilities Index (ECI): one number fitted from dozens of benchmarks
  (GPQA Diamond, SWE-bench Verified, HLE, FrontierMath, ...). CC BY 4.0, no key. https://epoch.ai/benchmarks
- Arena's text leaderboard (arena.ai, formerly LMArena), the default style-controlled
  rating from blind pairwise votes. CC BY 4.0, no key. https://arena.ai/leaderboard
- The Artificial Analysis Intelligence Index: their own runs of a set of evaluations, combined
  into one index. The free API needs a key in AA_API_KEY, and the site has to credit them.
  https://artificialanalysis.ai/

A leaderboard entry belongs to a row when its name matches the row's name, its repo or any
version uploaded with the same configuration (bench.yaml lists the names that don't match on
their own). A row takes the best score among its versions and records which one it came from,
since a later post-training release on the same architecture is still that architecture.

The three use different scales, so `composite` also gives every scored row one normalized average.

    python scripts/bench.py   # after a build: what each row matched, and likely misses
"""
import csv
import datetime as dt
import io
import json
import os
import re
import statistics
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

EPOCH_ZIP = "https://epoch.ai/data/benchmark_data.zip"
ARENA_PARQUET = ("https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/"
                 "text_style_control/latest-00000-of-00001.parquet")
AA_MODELS = "https://artificialanalysis.ai/api/v2/language/models/free"
SOURCES = {
    "eci": {"label": "ECI", "name": "Epoch Capabilities Index", "url": "https://epoch.ai/benchmarks"},
    "arena": {"label": "Arena", "name": "Arena 文本榜", "url": "https://arena.ai/leaderboard/text"},
    # the API has no publication date, so this one is dated by when its scores last changed
    "aa": {"label": "AA", "name": "Artificial Analysis 智能指数", "url": "https://artificialanalysis.ai/", "fetched": True},
}

# trailing name parts that don't change the weights' architecture: chat/instruct tags, precisions, dates
_TAIL = re.compile(r"^(instruct|chat|it|bf16|fp8|nvfp4|base|pt|hf|\d{4,8})$")


def key(name):
    s = re.sub(r"\(.*?\)", " ", name.split("/")[-1].lower())
    parts = [p for p in re.split(r"[\s_\-]+", s) if p]
    while len(parts) > 1 and _TAIL.match(parts[-1]):
        parts.pop()
    return "".join(parts)


def _get(url, timeout=120, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "moe-config-atlas", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _num(x):
    return float(x) if x not in (None, "") else None


def fetch_eci():
    with zipfile.ZipFile(io.BytesIO(_get(EPOCH_ZIP))) as z:
        name = next(n for n in z.namelist() if n.endswith("eci_scores.csv"))
        date = "%04d-%02d-%02d" % z.getinfo(name).date_time[:3]
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))
    entries = [{"name": r["Model"], "score": float(r["eci"]), "lo": _num(r.get("eci_ci_low")), "hi": _num(r.get("eci_ci_high"))}
               for r in rows if r.get("eci")]
    return {"date": date, "entries": entries}


def fetch_arena():
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(_get(ARENA_PARQUET)), filters=[("category", "==", "overall")])
    rows = table.to_pylist()
    entries = [{"name": r["model_name"], "score": r["rating"], "lo": r["rating_lower"], "hi": r["rating_upper"],
                "votes": int(r["vote_count"] or 0)} for r in rows if r["rating"] is not None]
    return {"date": max(r["leaderboard_publish_date"] for r in rows), "entries": entries}


def fetch_aa():
    key = os.environ.get("AA_API_KEY")
    if not key:
        raise urllib.error.URLError("AA_API_KEY is not set (a free key from artificialanalysis.ai/data-api/docs)")
    entries, version = [], None
    for page in range(1, 51):
        body = json.loads(_get(f"{AA_MODELS}?page={page}", timeout=60, headers={"x-api-key": key}))
        version = body.get("intelligence_index_version") or version
        for m in body["data"]:
            score = (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
            if score is not None:
                entries.append({"name": m["name"], "score": score, "lo": None, "hi": None})
        if not body.get("pagination", {}).get("has_more"):
            break
    return {"date": dt.date.today().isoformat(), "version": version, "entries": entries}


FETCH = {"eci": fetch_eci, "arena": fetch_arena, "aa": fetch_aa}


def load(cached, refresh):
    """{source: {"date", "entries"}} for the sources that could be read, fresh or from .cache/."""
    out = {}
    for src, fetch in FETCH.items():
        def safe(fetch=fetch, src=src):
            try:
                return fetch()
            except urllib.error.URLError:
                raise
            except Exception as e:  # a changed file layout falls back like a network error
                raise urllib.error.URLError(f"{src}: {e!r}") from e
        data = cached(f"bench/{src}.json", safe, refresh)
        if data and data.get("entries"):
            out[src] = data
        else:
            print(f"  ! {src}: no leaderboard data, keeping the previous scores", file=sys.stderr)
    return out


FIELDS = {"eci": ("eci", "eci_ci", "eci_as"), "arena": ("arena", "arena_ci", "arena_as", "arena_votes"),
          "aa": ("aa", "aa_as")}
DECIMALS = {"eci": 1, "arena": 0, "aa": 1}


def _round(src, x):
    return round(x, DECIMALS[src]) if DECIMALS[src] else round(x)


def _matcher(board, aliases, src):
    """row -> the board's entries for that row: by name key, plus the names bench.yaml lists."""
    by_key = {}
    for e in board["entries"]:
        by_key.setdefault(key(e["name"]), []).append(e)
    names = {e["name"]: e for e in board["entries"]}

    def hits(row):
        own = {key(n) for n in [row["name"], row["repo"], *row["variants"]]}
        return [e for k in own for e in by_key.get(k, [])] + [
            names[n] for n in (aliases.get(row["name"]) or {}).get(src, []) if n in names]
    return hits


def attach(rows, boards, aliases, prev_rows, prev_meta):
    """Add the score fields of every source to every row; return metadata for the sources that score any row.

    A source that could not be read keeps the values of the previous build, so a leaderboard
    being down for a day doesn't wipe the scores from the site.
    """
    meta = {}
    prev_meta = prev_meta or {}
    for src in FETCH:
        board = boards.get(src)
        if board is None:
            for row in rows:
                old = prev_rows.get(row["name"], {})
                row.update({f: old.get(f) for f in FIELDS[src]})
            meta[src] = {**SOURCES[src], **prev_meta.get(src, {})}
            continue
        hits = _matcher(board, aliases, src)
        for row in rows:
            best = max(hits(row), key=lambda e: e["score"], default=None)
            vals = {src: _round(src, best["score"]) if best else None,
                    f"{src}_ci": [_round(src, best["lo"]), _round(src, best["hi"])] if best and best.get("lo") is not None else None,
                    f"{src}_as": best["name"] if best else None,
                    f"{src}_votes": best.get("votes") if best else None}
            row.update({f: vals[f] for f in FIELDS[src]})
        date = board["date"]
        old = prev_meta.get(src, {})
        if SOURCES[src].get("fetched") and old.get("date") and all(
                prev_rows.get(r["name"], {}).get(f) == r[f] for r in rows for f in FIELDS[src]):
            date = old["date"]  # same scores as last time: keep the date so the data commit stays empty
        meta[src] = dict(SOURCES[src], date=date)
        if board.get("version"):
            meta[src]["name"] += f" v{board['version']:g}"
    for src in meta:
        meta[src].pop("fetched", None)
        meta[src]["n"] = sum(r[src] is not None for r in rows)
    return {src: m for src, m in meta.items() if m["n"]}


MIN_COMMON = 8  # fewer configs on every board than this, and each board is standardized on its own rows


def composite(rows, meta):
    """Give every scored row `avg`, the normalized average of its boards (0-100 across the site), and `avg_n`.

    ECI, Arena and AA use different scales, and they cover different configs. Each board becomes a
    standard score, (score - mean) / spread, with the mean and spread taken over the configs that every
    board has scored, so a config one board leaves out is still measured against the same group. A row
    averages the boards it is on; the averages are then scaled so the best on the site is 100 and the
    worst 0. Adds the "avg" entry to `meta`.
    """
    srcs = [s for s in FETCH if s in meta]
    if not srcs:
        for r in rows:
            r["avg"] = r["avg_n"] = None
        return meta
    common = [r for r in rows if all(r[s] is not None for s in srcs)]
    base = {s: [r[s] for r in common] if len(common) >= MIN_COMMON else [r[s] for r in rows if r[s] is not None]
            for s in srcs}
    stats = {s: (statistics.fmean(xs), statistics.pstdev(xs)) for s, xs in base.items()}
    z = {}
    for r in rows:
        zs = [(r[s] - stats[s][0]) / stats[s][1] for s in srcs if r[s] is not None and stats[s][1]]
        z[r["name"]] = statistics.fmean(zs) if zs else None
        r["avg_n"] = len(zs) or None
    vals = [v for v in z.values() if v is not None]
    lo, hi = min(vals), max(vals)
    for r in rows:
        r["avg"] = round(100 * (z[r["name"]] - lo) / (hi - lo), 1) if z[r["name"]] is not None and hi > lo else None
    meta["avg"] = {"label": "均分", "name": "归一均分", "sources": srcs,
                   "common": len(common) if len(common) >= MIN_COMMON else 0, "n": sum(r["avg"] is not None for r in rows)}
    return meta


def report(rows, boards, aliases):
    """Print the entry each row took, and the entries from model families on the site that match no row."""
    family = lambda k: re.match(r"[a-z]*", k).group(0) or k  # noqa: E731  "qwen", "glm", "gptoss", "deepseekv"
    families = {family(key(n)) for r in rows for n in [r["name"], r["repo"], *r["variants"]]}
    for src, board in boards.items():
        hits = _matcher(board, aliases, src)
        matched = {e["name"] for r in rows for e in hits(r)}
        print(f"\n{SOURCES[src]['label']} ({board['date']}): {sum(r[src] is not None for r in rows)} rows scored")
        for r in rows:
            if r[src] is not None:
                print(f"  {r['name']:<28} <- {r[f'{src}_as']}")
        miss = sorted({e["name"] for e in board["entries"] if e["name"] not in matched and family(key(e["name"])) in families})
        print(f"  no row, same family ({len(miss)}): " + "; ".join(miss))


if __name__ == "__main__":
    import yaml

    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "site" / "data" / "models.json").read_text())
    boards = {src: json.loads(p.read_text()) for src in FETCH if (p := root / ".cache" / "bench" / f"{src}.json").exists()}
    report(data["models"], boards, yaml.safe_load((root / "bench.yaml").read_text()) or {})
