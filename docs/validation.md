# Validation record

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
