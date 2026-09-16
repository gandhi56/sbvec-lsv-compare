"""Split a -sbvec-passes pipeline into per-vectorizer-pass verification stages.

The SandboxVectorizer has no region-level verifier pass (and
-sbvec-always-verify only exists in assertion builds), so the harness verifies
from the outside: for every vectorizer region pass in the pipeline it runs a
truncated pipeline that stops right after that pass, followed by `tr-accept`
so the pass's raw output is kept even if the real pipeline would later revert
it for cost reasons.  The output of each stage is then run through the IR
verifier.

    seed-collection(enable-diff-types)<tr-save,bundle-vec(bottom-up),load-store-vec,tr-accept-or-revert>

becomes

    bundle-vec      seed-collection(enable-diff-types)<tr-save,bundle-vec(bottom-up),tr-accept>
    load-store-vec  seed-collection(enable-diff-types)<tr-save,bundle-vec(bottom-up),load-store-vec,tr-accept>

Limitation: a stage accepts every region's changes, while the real pipeline
may revert some of them, so later regions within a stage can see different
IR than they would in the full pipeline.  The full pipeline's output is still
verified separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Region passes that transform IR, i.e. the ones worth verifying after.
VECTORIZER_PASSES = frozenset({"bundle-vec", "load-store-vec", "pack-reuse"})


@dataclass
class PassSpec:
    name: str
    args: str = ""  # including the parentheses, e.g. "(bottom-up)"
    region_passes: list["PassSpec"] | None = None

    def render(self) -> str:
        s = self.name + self.args
        if self.region_passes is not None:
            s += "<" + ",".join(p.render() for p in self.region_passes) + ">"
        return s


@dataclass
class Stage:
    after: str  # the vectorizer pass this stage ends with, e.g. "bundle-vec"
    pipeline: str
    index: int = field(default=0)  # position among stages, for unique labels

    @property
    def label(self) -> str:
        return f"{self.index}-{self.after}"


def _split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced pipeline: {text}")
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if depth != 0:
        raise ValueError(f"unbalanced pipeline: {text}")
    parts.append("".join(cur))
    return [p for p in parts if p]


def _parse_one(text: str) -> PassSpec:
    i = 0
    while i < len(text) and text[i] not in "(<":
        i += 1
    spec = PassSpec(text[:i])
    if i < len(text) and text[i] == "(":
        close = text.index(")", i)
        spec.args = text[i:close + 1]
        i = close + 1
    if i < len(text):
        if text[i] != "<" or not text.endswith(">"):
            raise ValueError(f"cannot parse pass: {text}")
        spec.region_passes = [_parse_one(p) for p in _split_top(text[i + 1:-1])]
    return spec


def parse(pipeline: str) -> list[PassSpec]:
    return [_parse_one(p) for p in _split_top("".join(pipeline.split()))]


def stages(pipeline: str) -> list[Stage]:
    """One stage per vectorizer region pass, in pipeline order."""
    fn_passes = parse(pipeline)
    out: list[Stage] = []
    for fi, fp in enumerate(fn_passes):
        for ri, rp in enumerate(fp.region_passes or []):
            if rp.name not in VECTORIZER_PASSES:
                continue
            head = PassSpec(fp.name, fp.args,
                            [*fp.region_passes[:ri + 1], PassSpec("tr-accept")])
            text = ",".join([*(p.render() for p in fn_passes[:fi]), head.render()])
            out.append(Stage(rp.name, text, len(out)))
    return out
