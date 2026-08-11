"""Find the lit tests that exercise the LoadStoreVectorizer through `opt`."""

from __future__ import annotations

import os
import re
import subprocess

from .runline import LSV_PASS_NAME, iter_run_lines

# `llc -amdgpu-load-store-vectorizer=0` and friends are codegen flags, not the
# IR pass; they must not be picked up.
CANDIDATE_RE = re.compile(r"-{1,2}passes\s*[= ]\s*'?\"?[^ ]*" + re.escape(LSV_PASS_NAME))

DEFAULT_ROOTS = ("llvm/test/Transforms/LoadStoreVectorizer", "llvm/test")
TEST_SUFFIXES = (".ll",)


def find_tests(llvm_root: str, roots: list[str] | None = None) -> list[str]:
    """Return absolute paths of test files with an opt-based LSV RUN line."""
    search_roots = [os.path.join(llvm_root, r) for r in (roots or ["llvm/test"])]
    hits: list[str] = []
    for root in search_roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(TEST_SUFFIXES):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", errors="replace") as f:
                        blob = f.read()
                except OSError:
                    continue
                if LSV_PASS_NAME not in blob:
                    continue
                if any(CANDIDATE_RE.search(rl.text) for rl in iter_run_lines(path)):
                    hits.append(path)
    return sorted(set(hits))


def registered_targets(llvm_bin: str) -> set[str]:
    """Targets compiled into this build, from `llc --version`."""
    llc = os.path.join(llvm_bin, "llc")
    if not os.path.exists(llc):
        return set()
    try:
        out = subprocess.run(
            [llc, "--version"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    targets, seen_header = set(), False
    for line in out.splitlines():
        if "Registered Targets" in line:
            seen_header = True
            continue
        if not seen_header:
            continue
        m = re.match(r"\s+([A-Za-z0-9_.\-]+)\s+-\s", line)
        if m:
            targets.add(m.group(1))
    return targets


def canonical_target(triple: str) -> str:
    arch = triple.split("-", 1)[0].lower()
    arch = re.sub(r"\d+\.\d+$", "", arch)  # amdgpu7.01 -> amdgpu
    if arch in ("x86_64", "x86_64h", "amd64"):
        return "x86-64"
    if re.fullmatch(r"i[3-6]86|x86", arch):
        return "x86"
    if arch.startswith("amdgcn"):
        return "amdgcn"
    if arch.startswith("amdgpu"):
        return "amdgpu"
    if arch.startswith("r600"):
        return "r600"
    if arch.startswith("nvptx"):
        return "nvptx64" if "64" in arch else "nvptx"
    if arch in ("aarch64", "arm64", "aarch64_be", "arm64_32"):
        return "aarch64"
    if arch.startswith("thumb") or arch.startswith("arm"):
        return "arm"
    return arch
