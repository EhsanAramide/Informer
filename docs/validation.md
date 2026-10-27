# Validation record

Final owner review approved all five chunks and closed this journey. The dated
entries below retain the results and review status recorded at each handoff.
Closure is documentation-only; the last verified suite remains 66 passed and
2 CUDA skips. IonMPX temporal integration is a separate development journey.

## Chunk 1 — reference characterization (2026-10-25)

Status: implemented, pending owner review. No packaged implementation exists yet;
these results characterize the retained reference, not package/reference parity
or installability. Upstream model and utility bytes are unchanged.

Run from this repository:

```sh
python -m pytest -q tests
```

Equivalent command used from the parent checkout:
`python -m pytest -q externals/Informer/tests`.
Result: **10 passed in 34.78 seconds**. The initial test-first run had one passing
source-integrity check and nine expected failures because the isolated worker
had not been implemented. After adding the worker, all checks passed.

### Verified coverage

- Frozen model/utility Git blob hashes at reference revision
  `29f2a739226a509202a092b464163da81fa74960`.
- Eight model cases: Informer/InformerStack × full/ProbSparse × CPU float32/float64.
  Encoder/decoder/forecast widths are respectively 5/2/3, encoder length 16,
  decoder context length 8, forecast length 4, batch size 2. These are synthetic
  test choices, not recommended scientific settings.
- Finite predictions, input and parameter gradients, non-mutated inputs and
  evaluation state, optional/nested attention returns, seeded exact output replay,
  and strict in-memory state-dictionary save/load with exact prediction recovery.
- Evaluation-time ProbSparse RNG consumption; causal full-attention masking;
  distillation BatchNorm updates; circular value-embedding endpoint behavior.
- Known failures for odd positional-embedding width and length-one ProbSparse.

Each numerical case runs in a fresh `python -I -B` subprocess with a temporary
working directory. Only the worker explicitly adds the reference root to its
import path. Its imported model location is checked; generic `models` and `utils`
imports never enter the parent test process. No benchmark data, network access,
IonMPX imports, or checkpoint files are needed. Child stderr is retained by
pytest; use `-rP` or `-s` to inspect captured output from passing tests.

### Tested environment and limits

| Component | Observed version/status |
|---|---|
| Python | 3.14.7 |
| PyTorch | 2.14.0 |
| NumPy | 2.5.3 |
| pytest | 9.1.1 |
| Device | CPU; CUDA unavailable |

CUDA was not exercised; no CUDA coverage is implied. Other versions within the
declared future dependency ranges remain unverified. This baseline does not yet
test fixed/learned calendar embeddings, all masks and length combinations,
training-mode model parity, or utilities. Those belong to Chunks 3–4. Timestamp
and utility limitations in the architecture guide are source-inspection findings,
not claims of successful utility execution on modern dependencies.

The future extracted package must be compared against this reference using
identical weights, inputs, modes, and reset random states. Neither passing this
baseline nor later parity tests establishes forecasting performance.

## Chunk 2 — package extraction and installation (2026-10-26)

Status: implemented, pending owner review. Result: **22 passed in 94.44 seconds**
for `python -m pytest -q externals/Informer/tests` from the parent checkout.

The result includes the ten Chunk 1 baseline tests plus:

- Public root and component imports with version `0.0.0`.
- A blocked-pandas process proving that core imports remain available and the
  explicit timestamp module reports the `informer[time]` remedy.
- Eight exact reference/package cases: both models × full/ProbSparse attention ×
  CPU float32/float64. Identical state dictionaries and reset RNG states produce
  identical predictions, attention tensors, input gradients, and parameter
  gradients.
- Wheel and source-distribution builds followed by separate temporary-environment
  installations and imports from outside the checkout. Neither historical
  top-level namespace is loaded by installed-package use.

Build tests use the locally installed build backend (`--no-isolation`) and install
artifacts with `--no-deps`; this avoids network access and makes dependency
availability an explicit test-environment precondition. It does not replace a
future clean dependency-resolution matrix. CUDA remains unavailable and untested.

The maintained source differs from the frozen model/utilities only where needed
for relative imports, package initializers, and actionable optional-pandas import
handling. Algorithms, constructor signatures, defaults, parameter names, state
keys, and numerical operations were not changed.

## Chunk 3 — model customization and compatibility (2026-10-26)

Status: implemented, pending owner review. Result: **39 passed, 2 skipped in
98.68 seconds** for the complete standalone test suite. Both skipped cases are
conditional CUDA float32/float64 smoke tests; CUDA is unavailable on this host.

New verification covers:

- Complete class, constructor, and forward docstrings for all public model,
  attention, embedding, encoder, and decoder components.
- Fixed categorical, learned categorical, and continuous timestamp embeddings.
- Distillation enabled and disabled for Informer and InformerStack.
- Causal full-attention masks, normalized attention rows, controlled ProbSparse
  RNG replay, custom attention projection widths, and attention gradients.
- Dropout train/evaluation behavior, strict state restoration, distinct
  encoder/decoder/output widths, and CPU float32/float64 behavior.
- Six additional exact package/reference comparisons spanning the three timestamp
  embedding modes and both distillation settings. Together with Chunk 2 cases,
  exact comparisons cover both model classes and both attention families.

Exact parity includes predictions, returned attention tensors, input gradients,
parameter gradients, state-dictionary names, and loaded parameter values. Tests
use matching modes, inputs, weights, and reset RNG states. They do not claim that
training-mode batches are independent: distillation deliberately retains the
reference BatchNorm behavior. They also do not make ProbSparse deterministic
without RNG control.

No model algorithm, operation, signature, default, initialization, mask rule,
sampling rule, or state key changed in this chunk. Source changes are docstrings;
the expanded behavior is established through tests and documentation. Passing
compatibility checks demonstrates preservation, not forecasting accuracy.

## Chunk 4 — utility compatibility (2026-10-27)

Status: implemented, pending owner review. Full standalone suite: **66 passed,
2 skipped in 103.93 seconds**. CUDA remains unavailable; both skips are the
existing CUDA model checks. The 27 new utility tests first produced 25 passes
and two failures demonstrating removed NumPy/pandas APIs. Following compatibility
changes, the targeted suite passed all 27 tests with no warning summary.

Environment: Python 3.14.7, NumPy 2.5.3, pandas 2.3.3, PyTorch 2.14.0, pytest
9.1.1. pandas 3.x is declared in the supported range but was not tested here.

Verified numerical fixtures cover global metrics and the reference CORR formula,
population scaling, inverse last-feature broadcasting, tensor gradients, ISO
week 53 across year boundaries, leap-year endpoints, continuous channel order,
categorical minute bins, and input mutation. Temporary directories contain all
early-stopping checkpoints. Tests also cover patience/delta/ties, both learning
rate schedules, dotdict, and Boolean causal/selected-query masks. Expected
division warnings are asserted locally rather than suppressed globally.

Compatibility changes are limited to:

1. `np.Inf` → `np.inf`, preserving positive infinity initialization.
2. `DatetimeIndex.week` → ISO week extraction converted to a NumPy integer array,
   preserving calendar-week values and the original scaling formula.
3. Removing positional `convert_dtype=True` from categorical Series.apply calls;
   fixed fixtures verify the same values, column order, and DataFrame mutation.

The missing-pandas test now asserts core-import isolation in its dedicated child
process, so importing pandas for calendar tests during collection cannot affect
that assertion. Original root sources still pass their frozen hash checks.
Utility input limits and formulas are recorded in [utilities.md](utilities.md).
Chunk 5 remains unstarted.

## Chunk 5 — examples and installation closure (2026-10-27)

Status: implementation complete, pending final owner review. Full standalone
suite: **66 passed, 2 skipped in 122.07 seconds**. Both skips require CUDA, which
is unavailable. The test count is unchanged because installation tests now also
exercise the four example invocations against each of the two artifacts.

Verified wheel and source-distribution installation from temporary working
directories, using isolated Python (`-I`), no PYTHONPATH, and confirmation that
the imported package resides inside the temporary environment. Both installations
run standard Informer, InformerStack, custom attention components, and components
with pandas calendar features. All examples verify shapes and gradients. Wheel
contents exclude generic model/utility namespaces; the source distribution
includes docs, examples, attribution, and frozen reference sources/tests.

Core imports and missing-extra guidance are checked with an import blocker that
makes pandas unavailable to the installed package. A separate normal process
exercises the calendar example with pandas present. This simulates absence;
it does not claim a physically pandas-free dependency environment. Temporary
environments reuse system dependencies and install artifacts with --no-deps and
--no-build-isolation. No clean network dependency-resolution matrix was run.

Tested environment remains Python 3.14.7, PyTorch 2.14.0, NumPy 2.5.3, pandas
2.3.3, and pytest 9.1.1. Packaging uses setuptools 82.0.1, build 1.6.0, and wheel
0.48.0. Other declared dependency combinations and CUDA behavior remain unverified.
The original license-file/README-badge discrepancy remains recorded in attribution.

The package README now documents installation, inputs, examples, reproducibility,
checkpoint compatibility, historical benchmark status, and limitations. Original
README content is retained in README.upstream.md. IonMPX overview links identify
the standalone dependency and future shared node-wise temporal integration.
Canonical context was not edited. No publication, benchmark reproduction,
forecasting comparison, or IonMPX temporal implementation occurred in this journey.
