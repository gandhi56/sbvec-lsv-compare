"""CSV / JSON / Markdown reporting."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict

from .blockers import DERIVED_TAGS, DESCRIPTIONS
from .runner import CaseResult


def write_json(results: list[CaseResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=1)


PER_TEST_FIELDS = [
    "test", "run_line", "status", "triple", "detail",
    "input_vec_lanes", "lsv_vec_lanes", "sbvec_vec_lanes",
    "lsv_gain_lanes", "sbvec_gain_lanes",
    "lsv_vec_mem_ops", "sbvec_vec_mem_ops",
    "lsv_scalar_removed", "sbvec_scalar_removed",
    "lsv_overhead", "sbvec_overhead",
    "cmd_sbvec",
]


def write_per_test_csv(results: list[CaseResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_TEST_FIELDS)
        w.writeheader()
        for r in results:
            i, l, s = r.input_total, r.lsv_total, r.sbvec_total
            w.writerow({
                "test": r.test,
                "run_line": r.run_line,
                "status": r.status,
                "triple": r.triple or "",
                "detail": r.detail,
                "input_vec_lanes": i.get("vec_lanes", ""),
                "lsv_vec_lanes": l.get("vec_lanes", ""),
                "sbvec_vec_lanes": s.get("vec_lanes", ""),
                "lsv_gain_lanes": _d(l, i, "vec_lanes"),
                "sbvec_gain_lanes": _d(s, i, "vec_lanes"),
                "lsv_vec_mem_ops": l.get("vec_mem_ops", ""),
                "sbvec_vec_mem_ops": s.get("vec_mem_ops", ""),
                "lsv_scalar_removed": _d(i, l, "scalar_mem_ops"),
                "sbvec_scalar_removed": _d(i, s, "scalar_mem_ops"),
                "lsv_overhead": _d(l, i, "overhead"),
                "sbvec_overhead": _d(s, i, "overhead"),
                "cmd_sbvec": r.cmd_sbvec,
            })


PER_FUNC_FIELDS = [
    "test", "run_line", "function", "category",
    "lsv_gain_lanes", "sbvec_gain_lanes",
    "lsv_vec_mem_ops", "sbvec_vec_mem_ops",
    "lsv_scalar_removed", "sbvec_scalar_removed",
    "recovered_by", "blockers", "ablation",
]


def write_per_function_csv(results: list[CaseResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_FUNC_FIELDS)
        w.writeheader()
        for r in results:
            for fn in r.functions:
                w.writerow({
                    "test": r.test,
                    "run_line": r.run_line,
                    "function": fn.name,
                    "category": fn.category,
                    "lsv_gain_lanes": fn.lsv_gain["vec_lanes"],
                    "sbvec_gain_lanes": fn.sbvec_gain["vec_lanes"],
                    "lsv_vec_mem_ops": fn.lsv["vec_mem_ops"],
                    "sbvec_vec_mem_ops": fn.sbvec["vec_mem_ops"],
                    "lsv_scalar_removed": fn.lsv_gain["scalar_mem_ops_removed"],
                    "sbvec_scalar_removed": fn.sbvec_gain["scalar_mem_ops_removed"],
                    "recovered_by": fn.recovered_by or "",
                    "blockers": ";".join(fn.blockers),
                    "ablation": ";".join(f"{k}={v}" for k, v in fn.ablation.items()),
                })


def _d(a: dict, b: dict, key: str):
    if key not in a or key not in b:
        return ""
    return a[key] - b[key]


def write_summary(results: list[CaseResult], path: str, meta: dict) -> str:
    statuses = Counter(r.status for r in results)
    ok = [r for r in results if r.status not in ("skipped",)]
    compared = [r for r in results if r.lsv_total and r.sbvec_total]

    tot_lsv = sum(_d(r.lsv_total, r.input_total, "vec_lanes") or 0 for r in compared)
    tot_sb = sum(_d(r.sbvec_total, r.input_total, "vec_lanes") or 0 for r in compared)
    tot_lsv_ops = sum(_d(r.lsv_total, r.input_total, "vec_mem_ops") or 0 for r in compared)
    tot_sb_ops = sum(_d(r.sbvec_total, r.input_total, "vec_mem_ops") or 0 for r in compared)
    tot_lsv_scalar = sum(_d(r.input_total, r.lsv_total, "scalar_mem_ops") or 0 for r in compared)
    tot_sb_scalar = sum(_d(r.input_total, r.sbvec_total, "scalar_mem_ops") or 0 for r in compared)

    cats = Counter(fn.category for r in results for fn in r.functions)
    recovered = Counter(
        fn.recovered_by or "not-recovered"
        for r in results for fn in r.functions
        if fn.category in ("full-miss", "partial-miss")
    )
    blocker_counts = Counter(
        b for r in results for fn in r.functions
        if fn.category in ("full-miss", "partial-miss") for b in fn.blockers
    )

    L = []
    L.append("# LoadStoreVectorizer vs SandboxVectorizer LoadStoreVec\n")
    L.append("| | |")
    L.append("|---|---|")
    for k, v in meta.items():
        L.append(f"| {k} | `{v}` |")
    L.append("")

    L.append("## Run status\n")
    L.append("| status | RUN lines |")
    L.append("|---|---:|")
    for k, v in statuses.most_common():
        L.append(f"| {k} | {v} |")
    L.append(f"| **total** | **{len(results)}** |")
    L.append("")

    L.append("## Vectorization totals (compared RUN lines only)\n")
    L.append("| metric | LSV | SandboxVectorizer | sbvec/LSV |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| vector lanes created | {tot_lsv} | {tot_sb} | {_pct(tot_sb, tot_lsv)} |")
    L.append(f"| vector memory ops created | {tot_lsv_ops} | {tot_sb_ops} | {_pct(tot_sb_ops, tot_lsv_ops)} |")
    L.append(f"| scalar memory ops removed | {tot_lsv_scalar} | {tot_sb_scalar} | {_pct(tot_sb_scalar, tot_lsv_scalar)} |")
    L.append(f"| RUN lines compared | {len(compared)} | | |")
    L.append("")

    L.append("## Per-function outcome\n")
    L.append("| category | functions |")
    L.append("|---|---:|")
    for k in ("parity", "sbvec-better", "partial-miss", "full-miss", "no-opportunity"):
        L.append(f"| {k} | {cats.get(k, 0)} |")
    L.append("")

    if recovered:
        L.append("## What recovers the misses (empirical ablation)\n")
        L.append("| knob | functions recovered |")
        L.append("|---|---:|")
        for k, v in recovered.most_common():
            L.append(f"| {k} | {v} |")
        L.append("")

    if blocker_counts:
        L.append("## Static blocker hints on missed functions\n")
        L.append("| blocker | functions | meaning |")
        L.append("|---|---:|---|")
        for k, v in blocker_counts.most_common():
            L.append(f"| `{k}` | {v} | {DESCRIPTIONS.get(k, '')} |")
        L.append("")

    crashes = [r for r in results if r.status in ("sbvec-crash", "sbvec-timeout", "sbvec-error", "sbvec-verify-failed")]
    if crashes:
        L.append("## SandboxVectorizer failures\n")
        for r in crashes:
            L.append(f"- `{r.test}:{r.run_line}` — **{r.status}** — {r.detail}")
            L.append(f"  - `{r.cmd_sbvec}`")
        L.append("")

    abl_fail = Counter(
        (label, err) for r in results for label, err in r.ablation_failures
    )
    if abl_fail:
        L.append("## Failures under relaxed knobs\n")
        L.append("These configurations are not used by the default pipeline, but "
                 "they crash or hang the vectorizer.\n")
        L.append("| knob | occurrences | first error |")
        L.append("|---|---:|---|")
        for (label, err), n in abl_fail.most_common():
            L.append(f"| `{label}` | {n} | `{err[:120]}` |")
        L.append("")

    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def write_missed(results: list[CaseResult], path: str) -> None:
    by_cause: dict[str, list[tuple[CaseResult, object]]] = defaultdict(list)
    seen: set[tuple] = set()
    for r in results:
        for fn in r.functions:
            if fn.category not in ("full-miss", "partial-miss"):
                continue
            # The same function is measured once per RUN line; collapse
            # duplicates that produced identical numbers.
            key = (r.test, fn.name, fn.lsv_gain["vec_lanes"], fn.sbvec_gain["vec_lanes"])
            if key in seen:
                continue
            seen.add(key)
            by_cause[_cause(fn)].append((r, fn))

    L = ["# Missed vectorization opportunities\n"]
    L.append(
        "Grouped by attributed cause: the relaxed-knob configuration that "
        "recovers the vectorization if one does, otherwise the strongest "
        "static blocker hint.\n"
    )
    order = sorted(by_cause.items(), key=lambda kv: -len(kv[1]))
    L.append("| cause | functions |")
    L.append("|---|---:|")
    for cause, items in order:
        L.append(f"| `{cause}` | {len(items)} |")
    L.append("")

    for cause, items in order:
        L.append(f"## `{cause}` ({len(items)} functions)\n")
        if cause in DESCRIPTIONS:
            L.append(f"> {DESCRIPTIONS[cause]}\n")
        items.sort(key=lambda it: -(it[1].lsv_gain["vec_lanes"] - it[1].sbvec_gain["vec_lanes"]))
        for r, fn in items:
            delta = fn.lsv_gain["vec_lanes"] - fn.sbvec_gain["vec_lanes"]
            L.append(
                f"- `{r.test}:{r.run_line}` `@{fn.name}` — "
                f"LSV +{fn.lsv_gain['vec_lanes']} lanes, sbvec +{fn.sbvec_gain['vec_lanes']} "
                f"(**-{delta}**), category {fn.category}"
            )
            if fn.blockers:
                L.append(f"  - hints: {', '.join('`' + b + '`' for b in fn.blockers)}")
            if fn.ablation:
                L.append(
                    "  - ablation lanes: "
                    + ", ".join(f"`{k}`={v}" for k, v in fn.ablation.items())
                )
            L.append(f"  - repro: `{r.cmd_sbvec}`")
        L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _cause(fn) -> str:
    """Best single explanation: an empirically recovering knob beats a static
    'why' hint, which beats a derived 'what was missed' hint."""
    if fn.recovered_by:
        return fn.recovered_by
    static = [b for b in fn.blockers if b not in DERIVED_TAGS]
    if static:
        return static[0]
    return fn.blockers[0] if fn.blockers else "unknown"


def _pct(a, b) -> str:
    if not b:
        return "n/a"
    return f"{100.0 * a / b:.1f}%"
