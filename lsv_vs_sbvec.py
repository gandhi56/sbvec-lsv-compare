#!/usr/bin/env python3
"""Compare LLVM's LoadStoreVectorizer with the SandboxVectorizer's LoadStoreVec
across the LoadStoreVectorizer lit tests.

Subcommands
  discover  list the lit tests that drive the LoadStoreVectorizer through opt
  compare   run both vectorizers on every such test and report the deltas
  rewrite   swap the pass in the RUN lines (diff by default, --apply to write)

Nothing under llvm-project is modified unless you pass `rewrite --apply`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbveccmp import report, rewrite as rewrite_mod  # noqa: E402
from sbveccmp.discover import find_tests, registered_targets  # noqa: E402
from sbveccmp.runner import ABLATIONS, PIPELINES, Runner  # noqa: E402

DEFAULT_LLVM_ROOT = os.path.expanduser("~/llvm-project")


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llvm-root", default=DEFAULT_LLVM_ROOT,
                   help="llvm-project checkout (default: %(default)s)")
    p.add_argument("--build-dir", default=None,
                   help="build directory (default: <llvm-root>/build)")
    p.add_argument("--tests", nargs="*", default=None,
                   help="explicit test files instead of discovery")
    p.add_argument("--test-root", action="append", default=None,
                   help="subtree to search, relative to llvm-root "
                        "(default: llvm/test/Transforms/LoadStoreVectorizer)")
    p.add_argument("--filter", default=None,
                   help="only tests whose path contains this substring")


def resolve_paths(args) -> tuple[str, str]:
    llvm_root = os.path.abspath(os.path.expanduser(args.llvm_root))
    build = args.build_dir or os.path.join(llvm_root, "build")
    build = os.path.abspath(os.path.expanduser(build))
    opt = os.path.join(build, "bin", "opt")
    if not os.path.exists(opt):
        sys.exit(f"error: opt not found at {opt} (use --build-dir)")
    return llvm_root, opt


def gather_tests(args, llvm_root: str) -> list[str]:
    if args.tests:
        tests = [os.path.abspath(t) for t in args.tests]
    else:
        roots = args.test_root or ["llvm/test/Transforms/LoadStoreVectorizer"]
        tests = find_tests(llvm_root, roots)
    if args.filter:
        tests = [t for t in tests if args.filter in t]
    return tests


# ---------------------------------------------------------------- discover
def cmd_discover(args) -> int:
    llvm_root, _ = resolve_paths(args)
    tests = gather_tests(args, llvm_root)
    for t in tests:
        print(os.path.relpath(t, llvm_root))
    print(f"\n{len(tests)} test files", file=sys.stderr)
    return 0


# ----------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    llvm_root, opt = resolve_paths(args)
    tests = gather_tests(args, llvm_root)
    if not tests:
        sys.exit("error: no tests found")

    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)
    ir_dir = os.path.join(out_dir, "ir") if args.keep_ir else None
    if ir_dir:
        os.makedirs(ir_dir, exist_ok=True)

    pipeline = PIPELINES.get(args.pipeline, args.pipeline)
    targets = registered_targets(os.path.dirname(opt))

    runner = Runner(
        opt_binary=opt,
        llvm_root=llvm_root,
        pipeline=pipeline,
        timeout=args.timeout,
        registered=targets,
        allow_unsupported=args.allow_unsupported_target,
        keep_ir_dir=ir_dir,
        run_verify=not args.no_verify,
        ablate=not args.no_ablate,
    )

    t0 = time.time()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(runner.run_test, tests):
            results.extend(res)
            done += 1
            if not args.quiet:
                print(f"\r[{done}/{len(tests)}] {done * 100 // len(tests)}%",
                      end="", file=sys.stderr, flush=True)
    if not args.quiet:
        print(file=sys.stderr)
    elapsed = time.time() - t0

    results.sort(key=lambda r: (r.test, r.run_line))
    report.write_json(results, os.path.join(out_dir, "results.json"))
    report.write_per_test_csv(results, os.path.join(out_dir, "per_test.csv"))
    report.write_per_function_csv(results, os.path.join(out_dir, "per_function.csv"))
    meta = {
        "opt": opt,
        "llvm-root": llvm_root,
        "sbvec pipeline": pipeline,
        "registered targets": ",".join(sorted(targets)) or "unknown",
        "test files": len(tests),
        "wall time": f"{elapsed:.1f}s",
    }
    text = report.write_summary(results, os.path.join(out_dir, "summary.md"), meta)
    report.write_missed(results, os.path.join(out_dir, "missed.md"))

    print(text)
    print(f"Wrote {out_dir}/summary.md, missed.md, per_test.csv, "
          f"per_function.csv, results.json")
    return 0


# ----------------------------------------------------------------- rewrite
def cmd_rewrite(args) -> int:
    llvm_root, opt = resolve_paths(args)
    tests = gather_tests(args, llvm_root)
    if not tests:
        sys.exit("error: no tests found")

    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)
    pipeline = PIPELINES.get(args.pipeline, args.pipeline)
    patch_path = os.path.join(out_dir, "rewrite.patch")

    changed, patch = rewrite_mod.run_rewrite(
        tests, llvm_root, pipeline, args.sbvec_flag or [], args.apply, patch_path
    )
    if not args.apply:
        print(patch)
        print(f"{changed} files would change; patch written to {patch_path}",
              file=sys.stderr)
        print("re-run with --apply to edit llvm-project in place "
              "(revert with `git checkout -- llvm/test`)", file=sys.stderr)
        return 0

    print(f"{changed} files rewritten in place; patch saved to {patch_path}")
    if args.update_checks:
        updated, handwritten, failures = rewrite_mod.update_checks(
            tests, llvm_root, opt
        )
        print(f"CHECK lines regenerated: {len(updated)} tests")
        print(f"hand-written CHECK lines (need manual triage): {len(handwritten)}")
        for f in handwritten:
            print(f"  {f}")
        if failures:
            print("update_test_checks.py failures:")
            for f in failures:
                print(f"  {f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="list matching lit tests")
    add_common(p)
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("compare", help="run both vectorizers and report")
    add_common(p)
    p.add_argument("--out", default="./results", help="output directory")
    p.add_argument("--pipeline", default="lsv-only",
                   help=f"sbvec pipeline preset {sorted(PIPELINES)} or a literal "
                        "-sbvec-passes string")
    p.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4)
    p.add_argument("--timeout", type=int, default=120, help="seconds per opt run")
    p.add_argument("--keep-ir", action="store_true",
                   help="save input/lsv/sbvec IR under <out>/ir")
    p.add_argument("--no-ablate", action="store_true",
                   help="skip the relaxed-knob re-runs "
                        f"({', '.join(n for n, _ in ABLATIONS)})")
    p.add_argument("--no-verify", action="store_true",
                   help="skip running the IR verifier on sbvec output")
    p.add_argument("--allow-unsupported-target", action="store_true",
                   help="run tests whose triple is not built into this opt "
                        "(they fall back to generic TTI)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("rewrite", help="swap the pass in RUN lines")
    add_common(p)
    p.add_argument("--out", default="./results", help="where to write the patch")
    p.add_argument("--pipeline", default="lsv-only")
    p.add_argument("--sbvec-flag", action="append",
                   help="extra opt flag to add to rewritten RUN lines "
                        "(repeatable), e.g. --sbvec-flag=-sbvec-allow-non-pow2")
    p.add_argument("--apply", action="store_true",
                   help="edit the tests in place instead of printing a diff")
    p.add_argument("--update-checks", action="store_true",
                   help="with --apply, regenerate autogenerated CHECK lines")
    p.set_defaults(func=cmd_rewrite)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
