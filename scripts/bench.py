"""Leaderboard scores for each configuration, used to sort the site by how well the models do.

Two public sources, both CC BY 4.0 and both readable without a key:

- Epoch AI's Capabilities Index (ECI): one number fitted from dozens of benchmarks
  (GPQA Diamond, SWE-bench Verified, HLE, FrontierMath, ...). https://epoch.ai/benchmarks
- Arena's text leaderboard (arena.ai, formerly LMArena), the default style-controlled
  rating from blind pairwise votes. https://arena.ai/leaderboard

A leaderboard entry belongs to a row when its name matches the row's name, its repo or any
version uploaded with the same configuration (bench.yaml lists the names that don't match on
their own). A row takes the best score among its versions and records which one it came from,
since a later post-training release on the same architecture is still that architecture.
"""
import csv
import io
import re
import sys
import urllib.error
import urllib.request
import zipfile

EPOCH_ZIP = "https://epoch.ai/data/benchmark_data.zip"
ARENA_PARQUET = ("https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/"
                 "text_style_control/latest-00000-of-00001.parquet")
SOURCES = {
    "eci": {"label": "ECI", "name": "Epoch Capabilities Index", "url": "https://epoch.ai/benchmarks"},
    "arena": {"label": "Arena", "name": "Arena 文本榜", "url": "https://arena.ai/leaderboard/text"},
}

# trailing name parts that don't change the weights' architecture: chat/instruct tags, precisions, dates
_TAIL = re.compile(r"^(instruct|chat|it|bf16|fp8|nvfp4|base|pt|hf|\d{4,8})$")


def key(name):
    s = re.sub(r"\(.*?\)", " ", name.split("/")[-1].lower())
    parts = [p for p in re.split(r"[\s_\-]+", s) if p]
    while len(parts) > 1 and _TAIL.match(parts[-1]):
        parts.pop()
    return "".join(parts)


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "moe-config-atlas"})
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


FETCH = {"eci": fetch_eci, "arena": fetch_arena}


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


FIELDS = {"eci": ("eci", "eci_ci", "eci_as"), "arena": ("arena", "arena_ci", "arena_as", "arena_votes")}


def attach(rows, boards, aliases, prev_rows, prev_meta):
    """Add the score fields of every source to every row; return per-source metadata.

    A source that could not be read keeps the values of the previous build, so a leaderboard
    being down for a day doesn't wipe the scores from the site.
    """
    meta = {}
    for src in FETCH:
        board = boards.get(src)
        if board is None:
            for row in rows:
                old = prev_rows.get(row["name"], {})
                row.update({f: old.get(f) for f in FIELDS[src]})
            meta[src] = dict(SOURCES[src], date=(prev_meta or {}).get(src, {}).get("date"))
            continue
        by_key = {}
        for e in board["entries"]:
            by_key.setdefault(key(e["name"]), []).append(e)
        names = {e["name"]: e for e in board["entries"]}
        for row in rows:
            own = {key(n) for n in [row["name"], row["repo"], *row["variants"]]}
            hits = [e for k in own for e in by_key.get(k, [])]
            hits += [names[n] for n in (aliases.get(row["name"]) or {}).get(src, []) if n in names]
            best = max(hits, key=lambda e: e["score"], default=None)
            rnd = (lambda x: round(x, 1)) if src == "eci" else round
            row[src] = rnd(best["score"]) if best else None
            row[f"{src}_ci"] = [rnd(best["lo"]), rnd(best["hi"])] if best and best["lo"] is not None else None
            row[f"{src}_as"] = best["name"] if best else None
            if src == "arena":
                row["arena_votes"] = best["votes"] if best else None
        meta[src] = dict(SOURCES[src], date=board["date"])
    for src in meta:
        meta[src]["n"] = sum(r[src] is not None for r in rows)
    return meta
