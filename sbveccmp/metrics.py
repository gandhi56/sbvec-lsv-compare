"""Metrics extracted from textual LLVM IR.

Both pipelines are measured with the exact same parser so the numbers are
directly comparable.  The SandboxVectorizer has no STATISTIC counters (LSV has
NumVectorInstructions / NumScalarsVectorized), so counting the resulting IR is
the only apples-to-apples option available today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

DEFINE_RE = re.compile(r"^define\b.*\{\s*$")
FUNC_NAME_RE = re.compile(r"@\"?([A-Za-z0-9_$.\\\-]+)\"?\s*\(")

# `%v = load atomic volatile <4 x i8>, ptr %p` / `%v = load i32, ptr %p`
LOAD_RE = re.compile(
    r"^\s*%[^=]+=\s*load\s+(?:atomic\s+|volatile\s+)*"
    r"(<\s*\d+\s+x\s+[^>]+>|[^,]+?)\s*,"
)
# `store <4 x i8> %v, ptr %p` / `store i32 %v, ptr %p`
STORE_RE = re.compile(
    r"^\s*store\s+(?:atomic\s+|volatile\s+)*(<\s*\d+\s+x\s+[^>]+>|\S+)\s"
)
VECTOR_TY_RE = re.compile(r"^<\s*(\d+)\s+x\s")

MASKED_LOAD_RE = re.compile(r"@llvm\.masked\.load")
MASKED_STORE_RE = re.compile(r"@llvm\.masked\.store")


@dataclass
class Counts:
    vec_loads: int = 0
    vec_stores: int = 0
    vec_load_lanes: int = 0
    vec_store_lanes: int = 0
    scalar_loads: int = 0
    scalar_stores: int = 0
    extractelement: int = 0
    insertelement: int = 0
    shufflevector: int = 0
    bitcast: int = 0
    instructions: int = 0
    sandboxvec_md: int = 0

    @property
    def vec_mem_ops(self) -> int:
        return self.vec_loads + self.vec_stores

    @property
    def vec_lanes(self) -> int:
        return self.vec_load_lanes + self.vec_store_lanes

    @property
    def scalar_mem_ops(self) -> int:
        return self.scalar_loads + self.scalar_stores

    @property
    def overhead(self) -> int:
        """Lane-shuffling instructions the vectorizer had to introduce."""
        return self.extractelement + self.insertelement + self.shufflevector

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            vec_mem_ops=self.vec_mem_ops,
            vec_lanes=self.vec_lanes,
            scalar_mem_ops=self.scalar_mem_ops,
            overhead=self.overhead,
        )
        return d


@dataclass
class ModuleMetrics:
    total: Counts = field(default_factory=Counts)
    functions: dict[str, Counts] = field(default_factory=dict)


def _lane_count(ty: str) -> int | None:
    m = VECTOR_TY_RE.match(ty.strip())
    return int(m.group(1)) if m else None


def _tally(line: str, c: Counts) -> None:
    stripped = line.strip()
    if not stripped or stripped.startswith(";"):
        return
    c.instructions += 1
    if "!sandboxvec" in stripped:
        c.sandboxvec_md += 1

    m = LOAD_RE.match(line)
    if m:
        lanes = _lane_count(m.group(1))
        if lanes:
            c.vec_loads += 1
            c.vec_load_lanes += lanes
        else:
            c.scalar_loads += 1
        return
    m = STORE_RE.match(line)
    if m:
        lanes = _lane_count(m.group(1))
        if lanes:
            c.vec_stores += 1
            c.vec_store_lanes += lanes
        else:
            c.scalar_stores += 1
        return
    if MASKED_LOAD_RE.search(stripped):
        c.vec_loads += 1
    elif MASKED_STORE_RE.search(stripped):
        c.vec_stores += 1

    if "= extractelement " in stripped:
        c.extractelement += 1
    elif "= insertelement " in stripped:
        c.insertelement += 1
    elif "= shufflevector " in stripped:
        c.shufflevector += 1
    elif "= bitcast " in stripped:
        c.bitcast += 1


def parse_module(ir_text: str) -> ModuleMetrics:
    mm = ModuleMetrics()
    cur: Counts | None = None
    for line in ir_text.splitlines():
        if cur is None:
            if DEFINE_RE.match(line):
                name_m = FUNC_NAME_RE.search(line)
                name = name_m.group(1) if name_m else f"<anon{len(mm.functions)}>"
                # Duplicate names cannot happen in valid IR, but be defensive.
                while name in mm.functions:
                    name += "'"
                cur = Counts()
                mm.functions[name] = cur
            continue
        if line.startswith("}"):
            cur = None
            continue
        _tally(line, cur)

    for c in mm.functions.values():
        for fld in (
            "vec_loads",
            "vec_stores",
            "vec_load_lanes",
            "vec_store_lanes",
            "scalar_loads",
            "scalar_stores",
            "extractelement",
            "insertelement",
            "shufflevector",
            "bitcast",
            "instructions",
            "sandboxvec_md",
        ):
            setattr(mm.total, fld, getattr(mm.total, fld) + getattr(c, fld))
    return mm


def gain(before: Counts, after: Counts) -> dict:
    """How much vectorization `after` has relative to `before`."""
    return {
        "vec_mem_ops": after.vec_mem_ops - before.vec_mem_ops,
        "vec_lanes": after.vec_lanes - before.vec_lanes,
        "scalar_mem_ops_removed": before.scalar_mem_ops - after.scalar_mem_ops,
        "overhead_added": after.overhead - before.overhead,
    }
