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
