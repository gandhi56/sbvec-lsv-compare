# LoadStoreVectorizer vs SandboxVectorizer LoadStoreVec

Swaps LLVM's `load-store-vectorizer` for the SandboxVectorizer's `load-store-vec`
region pass across the LoadStoreVectorizer lit tests, measures how much each one
vectorizes, and reports where the SandboxVectorizer falls short.

Lives outside `llvm-project`; nothing in the checkout is modified unless you run
`rewrite --apply`.

## Requirements

- An `opt` built with assertions (`LLVM_ENABLE_ASSERTIONS=ON`), default
  `<llvm-root>/build/bin/opt`.
- Targets used by the tests must be in `LLVM_TARGETS_TO_BUILD`. Tests whose
  triple is not built are **skipped** (with `--allow-unsupported-target` they run
  against generic TTI, which is not representative). To cover everything:
  `-DLLVM_TARGETS_TO_BUILD="X86;AMDGPU;NVPTX;AArch64"`.
- Python 3.10+, no third-party packages.

## Usage

```bash
python3 lsv_vs_sbvec.py discover
```

```bash
python3 lsv_vs_sbvec.py compare --out ./results --keep-ir
```

```bash
python3 lsv_vs_sbvec.py rewrite --out ./results          # prints a diff
```

```bash
python3 lsv_vs_sbvec.py rewrite --apply --update-checks   # edits llvm-project
```

Useful flags: `--llvm-root`, `--build-dir`, `--filter SUBSTR`, `--tests f.ll ...`,
`--test-root llvm/test/CodeGen`, `-j N`, `--timeout SECS`, `--no-ablate`,
`--pipeline {lsv-only,lsv-only-notxn,default}`.

Revert an applied rewrite with `git checkout -- llvm/test`.

## How the comparison works

For every RUN line that drives `opt -passes=...load-store-vectorizer...`:

1. Preceding setup RUN lines (`cp`, `printf`, …) are executed in a temp dir so
   `%t` inputs exist.
2. The command is normalised (`-S -o -`, redirects and `FileCheck` stripped) and
   run three times: unchanged input (baseline), LSV, and SandboxVectorizer.
3. The SandboxVectorizer invocation is the original command with the pass name
   swapped plus
   `-sbvec-passes="seed-collection(enable-diff-types)<tr-save,load-store-vec,tr-accept-or-revert>"`
   — the upstream default pipeline minus `bottom-up-vec`, so only LoadStoreVec is
   measured. All other flags (`-mtriple`, `-mcpu`, `-mattr`, `-aa-pipeline`) are
   preserved.
4. The output IR of both is verified with `opt -passes=verify` and counted.

### Why IR counting, not `-stats`

LSV has `NumVectorInstructions` / `NumScalarsVectorized`; the SandboxVectorizer
has no `STATISTIC` counters at all, so `-stats` cannot compare them. Both outputs
are instead parsed with the same counter, measuring vector/scalar loads and
stores, lanes, and the `extractelement`/`insertelement`/`shufflevector` overhead
introduced. Metrics are reported as a delta against the unmodified input, so
inputs that already contain vector accesses do not skew the numbers.

If matching `STATISTIC` counters are ever added to `LoadStoreVec.cpp`, `-stats`
becomes a useful cross-check — the IR counts stay the source of truth here.

## Missed-opportunity attribution

Each function is categorised per RUN line: `parity`, `sbvec-better`,
`partial-miss`, `full-miss`, `no-opportunity` (by vector lanes created).

Misses are attributed two ways:

- **Empirical ablation** — the miss is re-run under relaxed knobs
  (`-sbvec-allow-non-pow2`, `-sbvec-vec-reg-bits=1024`,
  `-sbvec-collect-seeds=stores,loads`, `-sbvec-cost-threshold=-1000000`,
  larger seed bundle/group limits, and all combined). The first configuration
  that reaches LSV's lane count is the attributed cause. This is proof, not a
  guess. Configurations that crash are reported separately.
- **Static hints** — heuristics mirroring the bail-outs in
  `LoadStoreVec::runOnRegion` (stored value not a load/constant, load with ≥2
  uses, cross-BB, loads-only chain with no store seed, mixed types, non-pow2
  chain, …), plus tags derived from the measured IR (`load-chain-missed`,
  `store-chain-shorter`, …).

`missed.md` groups findings by cause, ranked by lost lanes, each with a
copy-pasteable repro command.

## Outputs

| file | contents |
|---|---|
| `summary.md` | totals, per-function categories, ablation and blocker tables, crash list |
| `missed.md` | missed opportunities grouped by attributed cause, with repros |
| `per_test.csv` | one row per RUN line |
| `per_function.csv` | one row per function per RUN line, with ablation results |
| `results.json` | everything, for further analysis |
| `ir/` | with `--keep-ir`: input / LSV / SandboxVectorizer IR and stderr per case |

## Caveats

- RUN lines using `%if`, expecting a crash (`not --crash`), or with unmodelled
  lit substitutions are skipped and listed in `per_test.csv`.
- `rewrite --update-checks` only regenerates tests carrying the
  `utils/update_test_checks.py` header; hand-written CHECK lines are listed for
  manual triage.
- A `parity` verdict means equal lane counts, not identical IR.
