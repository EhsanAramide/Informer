# Standalone Informer packaging journey

## Status and workflow

Chunks 1–4 are implemented; Chunk 4 awaits owner review. The namespaced package
is independently buildable, while the root implementation remains frozen.
The owner approved this five-chunk, documentation-first and test-driven plan.
Each chunk stops for review; the next requires explicit confirmation.

Source baseline: `29f2a739226a509202a092b464163da81fa74960`. Preserve the root
`models/`, `utils/`, runners, scripts, and requirements as historical reference
material, not maintained package imports. Their model/utility Git blob hashes
are recorded in `tests/reference_manifest.json` and checked without requiring
Git at test runtime. Never silently refresh this baseline to accommodate a fix.

| Chunk | Deliverables | Review status |
|---|---|---|
| 1 | Scientific/package contract; isolated reference forward/backward, sampling, checkpoint and limitation tests | Reviewed and approved |
| 2 | Namespaced source extraction, build metadata, optional imports, wheel/source-distribution installation and reference parity | Reviewed and approved |
| 3 | Public component documentation and expanded model/customization compatibility matrix | Reviewed and approved |
| 4 | Utility compatibility, time/metric/scaler behavior, temporary checkpoint tests, equivalent modern API replacements | Implemented; pending review |
| 5 | Synthetic examples, API guide, isolated installation, tested versions, limitations and final documentation closure | Not started |

## Package contract

See [validation.md](validation.md) for the tested environment, baseline results,
and explicit exclusions. Review approval is required before Chunk 5 begins.

Distribution/import name: `informer`, version `0.0.0`, using setuptools
and `src/informer`. Export `Informer` and `InformerStack` at package root and
expose all current modules beneath `informer.models` and `informer.utils`.
Keep model constructor/forward arguments, defaults, return structures, parameter
names, and state-dictionary keys. Do not export top-level `models`/`utils` aliases.

Target Python 3.11+, PyTorch `>=2,<3`, NumPy `>=2,<3`; optional `time` extra uses
pandas `>=2,<4`. Test dependencies include pytest and build tooling. Declared
ranges are not evidence that every version combination was tested. Core model
imports must not require pandas, plotting, scikit-learn, benchmark data, or IonMPX.
All current utility modules remain accessible, including existing training helpers,
without adding a new training framework. Optional imports must fail with useful
installation guidance. Legacy dependency pins remain historical evidence only.

Current maintained layout:

```text
src/informer/
  __init__.py
  models/       # model, attention, encoder, decoder, embeddings
  utils/        # masks, time features, metrics, original tools
tests/          # reference isolation and packaged API checks
examples/       # synthetic, network-free usage
docs/           # contract, API, development, validation
```

## Change control and acceptance

Import restructuring, metadata, validation, and demonstrated equivalent modern
API replacements are permitted. Changes to attention, initialization, masking,
sampling, normalization, scaling conventions, or metric formulas must be
explained and approved before adoption. Known reference limitations are evidence,
not behaviors to silently repair or newly impose on unrelated inputs.

Compare reference and packaged implementations in isolated processes using the
same weights, tensors, modes, and random-number state. Test both models, supported
attention/embedding/distillation choices, CPU float32/64, conditional CUDA,
gradients, attention returns, and checkpoint restoration. Wheel and source
distribution installation is verified outside the checkout; source-tree imports
alone do not demonstrate installability. Test core imports without the optional
time dependency and calendar utilities with it.

Complete public documentation and runnable examples must accompany final review.
Reference runners remain historical; their modernization, benchmark reproduction,
and package publication are excluded. Preserve the license and attribution;
record conflicting repository notices rather than silently choosing terms.

IonMPX packing, embedding selection, layer handling, decoder-input construction,
JD, losses, optimization, and temporal/training orchestration are later journeys.
All work in this journey lives in this independent repository. Only final
dependency links may be updated in IonMPX at Chunk 5; canonical changes need fresh
permission. Completion means reviewed software compatibility, not forecast skill.
