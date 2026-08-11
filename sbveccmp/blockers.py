"""Static heuristics for *why* the SandboxVectorizer's LoadStoreVec bailed.

These mirror the bail-outs in
llvm/lib/Transforms/Vectorize/SandboxVectorizer/Passes/LoadStoreVec.cpp
(runOnRegion) and the seed collection in Passes/SeedCollection.cpp.  They are
hints, not proof -- the empirical ablation in runner.py is the stronger signal.
"""

from __future__ import annotations

import re

LOAD_DEF_RE = re.compile(
    r"^\s*(%[\w.$\-]+)\s*=\s*load\s+(?:atomic\s+|volatile\s+)*([^,]+),\s*ptr"
)
STORE_RE = re.compile(
    r"^\s*store\s+(?:atomic\s+|volatile\s+)*([^,]+?)\s+([^,]+),\s*ptr"
)
LABEL_RE = re.compile(r"^([\w.$\-]+):")
VALUE_USE_RE = re.compile(r"(%[\w.$\-]+)")

DESCRIPTIONS = {
    "loads-only": "vectorizable load chain with no store chain; seeds default to stores only",
    "store-operand-not-load": "stored value is neither a load nor a constant (LoadStoreVec only handles those)",
    "load-multi-use": "a candidate load has >= 2 uses; LoadStoreVec skips those",
    "multi-block": "memory operations span multiple basic blocks; LoadStoreVec does not cross BBs",
    "volatile-or-atomic": "volatile/atomic accesses in the function",
    "vector-elements": "chain elements are already vector-typed (needs concatenation)",
    "mixed-types": "the store chain mixes element types (needs enable-diff-types)",
    "non-pow2-chain": "chain length is not a power of two",
    "multi-addrspace": "several address spaces in play",
    "phi-or-call-operands": "stored values come from phis/calls",
    "no-stores": "function contains no stores at all",
    # Derived from the measured IR (see runner._derived_blockers).
    "load-chain-missed": "LSV vectorized a load chain that LoadStoreVec left scalar",
    "load-chain-shorter": "LoadStoreVec vectorized fewer load lanes than LSV",
    "store-chain-missed": "LSV vectorized a store chain that LoadStoreVec left scalar",
    "store-chain-shorter": "LoadStoreVec vectorized fewer store lanes than LSV",
}

# Tags computed from the measured IR rather than from source-level heuristics.
# They describe *what* was missed, so they are weaker grouping keys than the
# static tags, which describe *why*.
DERIVED_TAGS = frozenset(
    {
        "load-chain-missed",
        "load-chain-shorter",
        "store-chain-missed",
        "store-chain-shorter",
    }
)


def _function_bodies(ir_text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    name, body = None, []
    for line in ir_text.splitlines():
        if name is None:
            if line.startswith("define") and line.rstrip().endswith("{"):
                m = re.search(r"@\"?([A-Za-z0-9_$.\\\-]+)\"?\s*\(", line)
                name = m.group(1) if m else f"<anon{len(out)}>"
                body = []
            continue
        if line.startswith("}"):
            out[name] = body
            name = None
            continue
        body.append(line)
    return out


def analyse(ir_text: str) -> dict[str, list[str]]:
    """Return {function name: [blocker tags]} for the *input* IR."""
    result: dict[str, list[str]] = {}
    for fn, body in _function_bodies(ir_text).items():
        result[fn] = _analyse_body(body)
    return result


def _analyse_body(body: list[str]) -> list[str]:
    tags: list[str] = []
    loads: dict[str, str] = {}  # ssa name -> loaded type
    stored_values: list[tuple[str, str]] = []  # (type, value)
    use_counts: dict[str, int] = {}
    n_blocks = 1
    n_stores = 0
    has_volatile_atomic = False
    addrspaces = set()

    for line in body:
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        if LABEL_RE.match(s):
            n_blocks += 1
            continue
        if " volatile " in f" {s} " or re.search(r"\b(load|store)\s+atomic\b", s):
            has_volatile_atomic = True
        for m in re.finditer(r"addrspace\((\d+)\)", s):
            addrspaces.add(int(m.group(1)))

        m = LOAD_DEF_RE.match(line)
        if m:
            loads[m.group(1)] = m.group(2).strip()
        m = STORE_RE.match(line)
        if m:
            n_stores += 1
            stored_values.append((m.group(1).strip(), m.group(2).strip()))
        # Count uses on the RHS only (skip the defined value).
        rhs = s.split("=", 1)[1] if "=" in s.split(",")[0] else s
        for v in VALUE_USE_RE.findall(rhs):
            use_counts[v] = use_counts.get(v, 0) + 1

    if n_stores == 0:
        tags.append("no-stores")
        if len(loads) >= 2:
            tags.append("loads-only")
    else:
        non_load_operands = [
            v for ty, v in stored_values if v not in loads and not _is_constant(v)
        ]
        if non_load_operands:
            tags.append("store-operand-not-load")
            joined = "\n".join(body)
            if re.search(r"=\s*(phi|call|tail call)\b", joined):
                tags.append("phi-or-call-operands")
        store_load_ops = [v for _, v in stored_values if v in loads]
        if any(use_counts.get(v, 0) >= 2 for v in store_load_ops):
            tags.append("load-multi-use")
        store_types = {ty for ty, _ in stored_values}
        if len(store_types) > 1:
            tags.append("mixed-types")
        if any(ty.startswith("<") for ty in store_types):
            tags.append("vector-elements")
        if n_stores >= 2 and (n_stores & (n_stores - 1)) != 0:
            tags.append("non-pow2-chain")

    if n_blocks > 1:
        tags.append("multi-block")
    if has_volatile_atomic:
        tags.append("volatile-or-atomic")
    if len(addrspaces) > 1:
        tags.append("multi-addrspace")
    return tags


def _is_constant(v: str) -> bool:
    if v.startswith("%"):
        return False
    return bool(
        re.fullmatch(r"-?\d+|true|false|null|zeroinitializer|undef|poison|@.*", v)
        or v.startswith("0x")
        or v.startswith("<")
    )
