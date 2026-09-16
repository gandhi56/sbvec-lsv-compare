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
- Python 3.11+ (for `tomllib`), no third-party packages.

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
`--config PATH`, `--pipeline NAME`.

Revert an applied rewrite with `git checkout -- llvm/test`.

## Pipeline configuration

The SandboxVectorizer pipelines live in [`pipelines.toml`](pipelines.toml):

```toml
[pipelines]
default  = "seed-collection(enable-diff-types)<tr-save,bundle-vec(bottom-up),load-store-vec,tr-accept-or-revert>"
lsv-only = "seed-collection(enable-diff-types)<tr-save,load-store-vec,tr-accept-or-revert>"

[defaults]          # used when --pipeline is omitted
compare = "default"
rewrite = "lsv-only"
```

Add entries to `[pipelines]` and select them with `--pipeline NAME`, or point
`--config` at a different file. `--pipeline` also accepts a literal
`-sbvec-passes` string (anything containing `<` or `(`); unknown names are an
error. The pipeline used is recorded in `summary.md`.

## How the comparison works

For every RUN line that drives `opt -passes=...load-store-vectorizer...`:

1. Preceding setup RUN lines (`cp`, `printf`, …) are executed in a temp dir so
   `%t` inputs exist.
2. The command is normalised (`-S -o -`, redirects and `FileCheck` stripped) and
   run three times: unchanged input (baseline), LSV, and SandboxVectorizer.
3. The SandboxVectorizer invocation is the original command with the pass name
   swapped plus `-sbvec-passes="<pipeline>"`, the selected pipeline from
   `pipelines.toml`. All other flags (`-mtriple`, `-mcpu`, `-mattr`, `-aa-pipeline`) are
   preserved.
4. The SandboxVectorizer IR is verified after each vectorizer pass (see
   below) and at the end, then both outputs are counted.

### IR verification after each vectorizer pass

The SandboxVectorizer has no verifier region pass (`-sbvec-always-verify`
exists only in assertion builds and only covers `bundle-vec`). So the harness
cuts the pipeline after each vectorizer region pass (`bundle-vec`,
`load-store-vec`, `pack-reuse`), adds `tr-accept`, and runs
`opt -passes=verify` on the result:

```
bundle-vec      seed-collection(enable-diff-types)<tr-save,bundle-vec(bottom-up),tr-accept>
load-store-vec  seed-collection(enable-diff-types)<tr-save,bundle-vec(bottom-up),load-store-vec,tr-accept>
```

`tr-accept` keeps the pass's output even when the full pipeline would revert
it as unprofitable, so invalid IR is caught before cost checks can hide it.
The first failing pass is reported as `verify_failed_after`. RUN lines are
marked `sbvec-verify-failed` when the final IR is invalid, and
`sbvec-stage-verify-failed` when only an intermediate stage is invalid or
crashes. Because every region is accepted, later regions within a stage may see
different IR than in the full pipeline. `--no-verify` turns all of this off.

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
| `ir/` | with `--keep-ir`: input / LSV / SandboxVectorizer IR and stderr per case, plus `sbvec.after-<N>-<pass>.ll` for each verification stage |

## Sample results

Against llvm-project `bba25a8d818e`, `X86;AMDGPU` only (the 20 NVPTX/AArch64
tests were skipped), across 108 RUN lines:

| metric | LSV | SandboxVectorizer |
|---|---:|---:|
| vector lanes created | 1874 | 1632 (87%) |
| vector memory ops created | 542 | 274 (51%) |
| scalar memory ops removed | 878 | 592 (67%) |

Per function: 103 parity, 46 sbvec-better, 41 partial-miss, 92 full-miss.
Leading attributed causes: stored value is neither a load nor a constant (43),
loads-only chains with no store seed (21), non-power-of-2 chains (9).

The run also surfaced three defects: an assertion in `Type.cpp` on
`AMDGPU/pointer-elements.ll`, invalid IR (`vector element #1 is not of type
'i32'`) on `AMDGPU/merge-stores.ll`, and an assertion in `VecUtils.h`
(`isa<LoadOrStoreT>(Bndl[0])`) whenever `-sbvec-collect-seeds` includes loads,
because `LoadStoreVec::runOnRegion` assumes the region's aux bundle holds
stores.

## Caveats

- RUN lines using `%if`, expecting a crash (`not --crash`), or with unmodelled
  lit substitutions are skipped and listed in `per_test.csv`.
- `rewrite --update-checks` only regenerates tests carrying the
  `utils/update_test_checks.py` header; hand-written CHECK lines are listed for
  manual triage.
- A `parity` verdict means equal lane counts, not identical IR.

## License

Apache License v2.0 with LLVM Exceptions, matching llvm-project — see
[LICENSE.TXT](LICENSE.TXT).
