"""Run each LSV lit test twice -- once through the LoadStoreVectorizer and once
through the SandboxVectorizer's LoadStoreVec -- and compare the resulting IR."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field

from . import blockers, metrics
from .discover import canonical_target
from .runline import (
    OptCommand,
    UnsupportedRunLine,
    collect_setup_lines,
    expand_substitutions,
    expand_token,
    extract_triple,
    iter_run_lines,
    parse_opt_command,
    swap_pass,
    SBVEC_PASS_NAME,
)

# The default SandboxVectorizer pipeline is
#   seed-collection<tr-save,bottom-up-vec,load-store-vec,tr-accept-or-revert>
# For an apples-to-apples LSV comparison we drop bottom-up-vec.
PIPELINES = {
    "lsv-only": "seed-collection(enable-diff-types)"
    "<tr-save,load-store-vec,tr-accept-or-revert>",
    "lsv-only-notxn": "seed-collection(enable-diff-types)<load-store-vec>",
    "default": "seed-collection(enable-diff-types)"
    "<tr-save,bottom-up-vec,load-store-vec,tr-accept-or-revert>",
}

ABLATIONS: list[tuple[str, list[str]]] = [
    ("allow-non-pow2", ["-sbvec-allow-non-pow2"]),
    ("wide-vec-regs", ["-sbvec-vec-reg-bits=1024"]),
    ("load-seeds", ["-sbvec-collect-seeds=stores,loads"]),
    ("ignore-cost", ["-sbvec-cost-threshold=-1000000"]),
    (
        "bigger-bundles",
        ["-sbvec-seed-bundle-size-limit=1024", "-sbvec-seed-groups-limit=4096"],
    ),
    (
        "all-knobs",
        [
            "-sbvec-allow-non-pow2",
            "-sbvec-vec-reg-bits=1024",
            "-sbvec-collect-seeds=stores,loads",
            "-sbvec-cost-threshold=-1000000",
            "-sbvec-seed-bundle-size-limit=1024",
            "-sbvec-seed-groups-limit=4096",
        ],
    ),
]

CATEGORIES = (
    "parity",
    "full-miss",
    "partial-miss",
    "sbvec-better",
    "no-opportunity",
)


@dataclass
class ProcResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def crashed(self) -> bool:
        return self.timed_out or self.returncode < 0 or self.returncode > 1


@dataclass
class FunctionResult:
    name: str
    input: dict
    lsv: dict
    sbvec: dict
    lsv_gain: dict
    sbvec_gain: dict
    category: str
    blockers: list[str] = field(default_factory=list)
    recovered_by: str | None = None
    ablation: dict[str, object] = field(default_factory=dict)


@dataclass
class CaseResult:
    test: str
    run_line: int
    status: str
    detail: str = ""
    triple: str | None = None
    cmd_lsv: str = ""
    cmd_sbvec: str = ""
    input_total: dict = field(default_factory=dict)
    lsv_total: dict = field(default_factory=dict)
    sbvec_total: dict = field(default_factory=dict)
    functions: list[FunctionResult] = field(default_factory=list)
    # (ablation label, first error line) for knob configurations that crashed.
    ablation_failures: list[tuple[str, str]] = field(default_factory=list)


class Runner:
    def __init__(
        self,
        opt_binary: str,
        llvm_root: str,
        pipeline: str,
        timeout: int,
        registered: set[str],
        allow_unsupported: bool,
        keep_ir_dir: str | None,
        run_verify: bool,
        ablate: bool,
    ):
        self.opt = opt_binary
        self.llvm_root = llvm_root
        self.pipeline = pipeline
        self.timeout = timeout
        self.registered = registered
        self.allow_unsupported = allow_unsupported
        self.keep_ir_dir = keep_ir_dir
        self.run_verify = run_verify
        self.ablate = ablate

    # -- process helpers -------------------------------------------------
    def _run(self, argv: list[str], cwd: str | None = None) -> ProcResult:
        try:
            p = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                errors="replace",
            )
            return ProcResult(argv, p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired:
            return ProcResult(argv, -9, "", "timeout", timed_out=True)
        except OSError as e:
            return ProcResult(argv, -1, "", str(e))

    def _sbvec_flags(self, extra: list[str] | None = None) -> list[str]:
        return [f"-sbvec-passes={self.pipeline}", *(extra or [])]

    # -- main entry point ------------------------------------------------
    def run_test(self, test_path: str) -> list[CaseResult]:
        rel = os.path.relpath(test_path, self.llvm_root)
        run_lines = iter_run_lines(test_path)
        results: list[CaseResult] = []
        for idx, rl in enumerate(run_lines):
            try:
                cmd = parse_opt_command(rl.text)
            except UnsupportedRunLine:
                continue
            results.append(self._run_case(rel, test_path, run_lines, idx, rl, cmd))
        return results

    def _run_case(self, rel, test_path, run_lines, idx, rl, cmd: OptCommand) -> CaseResult:
        res = CaseResult(test=rel, run_line=rl.first_line, status="ok")
        # The triple can come from -mtriple or from the module itself.
        triple = extract_triple(cmd.args) or module_triple(test_path)
        res.triple = triple
        if triple:
            target = canonical_target(triple)
            if self.registered and target not in self.registered:
                if not self.allow_unsupported:
                    res.status = "skipped"
                    res.detail = f"target '{target}' not built into this opt"
                    return res
                res.detail = f"warning: target '{target}' not built; generic TTI used"

        if cmd.expects_crash:
            res.status = "skipped"
            res.detail = "RUN line expects a crash"
            return res

        with tempfile.TemporaryDirectory(prefix="sbveccmp-") as tmp:
            tmp_prefix = os.path.join(tmp, "t")
            try:
                setup = collect_setup_lines(run_lines, idx)
                for line in setup:
                    sh = expand_substitutions(line, test_path, tmp_prefix)
                    p = subprocess.run(
                        sh, shell=True, cwd=tmp, capture_output=True, text=True,
                        timeout=self.timeout,
                    )
                    if p.returncode != 0:
                        res.status = "skipped"
                        res.detail = f"setup command failed: {sh}"
                        return res
                input_path = expand_token(cmd.input_token, test_path, tmp_prefix)
            except UnsupportedRunLine as e:
                res.status = "skipped"
                res.detail = str(e)
                return res
            except subprocess.SubprocessError as e:
                res.status = "skipped"
                res.detail = f"setup failed: {e}"
                return res

            if not os.path.exists(input_path):
                res.status = "skipped"
                res.detail = f"input not found: {input_path}"
                return res

            return self._compare(res, cmd, input_path, tmp)

    def _compare(self, res: CaseResult, cmd: OptCommand, input_path: str, tmp: str):
        base = self._run([self.opt, "-S", "-o", "-", input_path])
        if not base.ok:
            res.status = "input-parse-failed"
            res.detail = _first_error(base)
            return res

        lsv_argv = cmd.render(self.opt, input_path, [])
        sb_args = swap_pass(cmd.args, SBVEC_PASS_NAME)
        sb_argv = [self.opt, *sb_args, *self._sbvec_flags(), input_path]
        res.cmd_lsv = shlex.join(lsv_argv)
        res.cmd_sbvec = shlex.join(sb_argv)

        lsv = self._run(lsv_argv)
        sb = self._run(sb_argv)

        if not lsv.ok:
            res.status = "lsv-failed"
            res.detail = _first_error(lsv)
            return res
        if sb.timed_out:
            res.status = "sbvec-timeout"
            return res
        if not sb.ok:
            res.status = "sbvec-crash" if sb.crashed else "sbvec-error"
            res.detail = _first_error(sb)
            return res

        if self.run_verify and sb.stdout != base.stdout:
            v = self._run_stdin([self.opt, "-passes=verify", "-disable-output"], sb.stdout)
            if not v.ok:
                res.status = "sbvec-verify-failed"
                res.detail = _first_error(v)
                # keep going: still report the metrics below

        m_in = metrics.parse_module(base.stdout)
        m_lsv = metrics.parse_module(lsv.stdout)
        m_sb = metrics.parse_module(sb.stdout)
        res.input_total = m_in.total.as_dict()
        res.lsv_total = m_lsv.total.as_dict()
        res.sbvec_total = m_sb.total.as_dict()

        blocker_map = blockers.analyse(base.stdout)
        empty = metrics.Counts()
        for fn, c_in in m_in.functions.items():
            c_lsv = m_lsv.functions.get(fn, empty)
            c_sb = m_sb.functions.get(fn, empty)
            g_lsv = metrics.gain(c_in, c_lsv)
            g_sb = metrics.gain(c_in, c_sb)
            fr = FunctionResult(
                name=fn,
                input=c_in.as_dict(),
                lsv=c_lsv.as_dict(),
                sbvec=c_sb.as_dict(),
                lsv_gain=g_lsv,
                sbvec_gain=g_sb,
                category=_categorise(g_lsv["vec_lanes"], g_sb["vec_lanes"]),
            )
            if fr.category in ("full-miss", "partial-miss"):
                fr.blockers = _derived_blockers(c_in, c_lsv, c_sb) + blocker_map.get(
                    fn, []
                )
            res.functions.append(fr)

        if self.ablate and any(
            f.category in ("full-miss", "partial-miss") for f in res.functions
        ):
            self._ablate(res, sb_args, input_path, m_in)

        if self.keep_ir_dir:
            self._dump(res, base.stdout, lsv.stdout, sb.stdout, sb.stderr)
        return res

    def _run_stdin(self, argv: list[str], data: str) -> ProcResult:
        try:
            p = subprocess.run(
                argv, input=data, capture_output=True, text=True,
                timeout=self.timeout, errors="replace",
            )
            return ProcResult(argv, p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired:
            return ProcResult(argv, -9, "", "timeout", timed_out=True)

    def _ablate(self, res: CaseResult, sb_args, input_path, m_in):
        """Re-run the missed cases under relaxed knobs to pin down the cause."""
        missed = {
            f.name: f for f in res.functions
            if f.category in ("full-miss", "partial-miss")
        }
        empty = metrics.Counts()
        for label, flags in ABLATIONS:
            argv = [self.opt, *sb_args, *self._sbvec_flags(flags), input_path]
            r = self._run(argv)
            if not r.ok:
                kind = "timeout" if r.timed_out else "crash"
                for f in missed.values():
                    f.ablation[label] = kind
                res.ablation_failures.append((label, _first_error(r)))
                continue
            mm = metrics.parse_module(r.stdout)
            for name, f in missed.items():
                c = mm.functions.get(name, empty)
                g = metrics.gain(m_in.functions[name], c)["vec_lanes"]
                f.ablation[label] = g
                if f.recovered_by is None and g >= f.lsv_gain["vec_lanes"] > 0:
                    f.recovered_by = label
            if all(f.recovered_by for f in missed.values()):
                break

    def _dump(self, res: CaseResult, base, lsv, sb, sb_err):
        d = os.path.join(
            self.keep_ir_dir,
            res.test.replace("/", "__").removesuffix(".ll") + f".L{res.run_line}",
        )
        os.makedirs(d, exist_ok=True)
        for name, blob in (
            ("input.ll", base),
            ("lsv.ll", lsv),
            ("sbvec.ll", sb),
            ("sbvec.stderr", sb_err),
            ("commands.txt", f"{res.cmd_lsv}\n{res.cmd_sbvec}\n"),
        ):
            with open(os.path.join(d, name), "w") as f:
                f.write(blob)


def _derived_blockers(c_in, c_lsv, c_sb) -> list[str]:
    """Tags read straight off the measured IR, so they cannot be wrong."""
    tags = []
    lsv_ld = c_lsv.vec_load_lanes - c_in.vec_load_lanes
    sb_ld = c_sb.vec_load_lanes - c_in.vec_load_lanes
    lsv_st = c_lsv.vec_store_lanes - c_in.vec_store_lanes
    sb_st = c_sb.vec_store_lanes - c_in.vec_store_lanes
    if lsv_ld > 0 and sb_ld <= 0:
        tags.append("load-chain-missed")
    elif lsv_ld > sb_ld > 0:
        tags.append("load-chain-shorter")
    if lsv_st > 0 and sb_st <= 0:
        tags.append("store-chain-missed")
    elif lsv_st > sb_st > 0:
        tags.append("store-chain-shorter")
    return tags


def _categorise(lsv_gain: int, sb_gain: int) -> str:
    if lsv_gain <= 0 and sb_gain <= 0:
        return "no-opportunity"
    if sb_gain > lsv_gain:
        return "sbvec-better"
    if sb_gain == lsv_gain:
        return "parity"
    if sb_gain <= 0:
        return "full-miss"
    return "partial-miss"


ERROR_MARKERS = ("Assertion failed", "UNREACHABLE", "error:", "Segmentation fault",
                 "LLVM ERROR", "Stack dump")


def _first_error(p: ProcResult) -> str:
    """The most informative stderr line, skipping leading warnings."""
    lines = [l.strip() for l in (p.stderr or "").splitlines() if l.strip()]
    for line in lines:
        if any(m in line for m in ERROR_MARKERS):
            return line[:300]
    return lines[0][:300] if lines else f"exit {p.returncode}"


MODULE_TRIPLE_RE = re.compile(r'^\s*target\s+triple\s*=\s*"([^"]+)"', re.M)


def module_triple(path: str) -> str | None:
    try:
        with open(path, "r", errors="replace") as f:
            head = f.read(65536)
    except OSError:
        return None
    m = MODULE_TRIPLE_RE.search(head)
    return m.group(1) if m else None
