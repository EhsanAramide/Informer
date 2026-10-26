# Scientific reference and preservation contract

This repository packages the reference Informer architecture; it does not
redesign it. The [original paper](https://arxiv.org/abs/2012.07436) proposes
ProbSparse attention, encoder distillation, and a generative-style decoder that
predicts the horizon in one forward call. Code-level parity targets the pinned
reference revision in [development.md](development.md), not an inferred rewrite
of the paper's equations.

## Model data flow

1. Encoder input values are projected by a width-three circular temporal Conv1d;
   positional and calendar/time-feature embeddings are added, then dropout applied.
2. Encoder attention/feed-forward blocks process history. Optional distillation
   applies circular convolution, BatchNorm1d, ELU, and max pooling between blocks.
3. Decoder embedding processes caller-supplied historical context and placeholders.
   Decoder self-attention is masked; cross-attention to encoder output is full.
4. A linear projection produces `c_out` channels; the last `out_len` positions
   are returned. The model does not construct placeholders or select targets.

InformerStack runs separate encoders on progressively shorter suffixes of the
embedded history and concatenates their outputs in time. Its attention diagnostics
are nested by encoder and block. This is not an IonMPX layer representation.

## Behavior to preserve and test

- ProbSparse draws sampled key indices using `torch.randint` even in evaluation.
  Reset the RNG before parity calls. Zero dropout and `.eval()` alone do not
  promise deterministic sampling. Different draws need not always change outputs.
- Distillation BatchNorm updates running buffers in training and uses them in
  evaluation. Training-time normalization couples sequences within a batch.
- Value embedding uses circular temporal padding, not causal convolution. A causal
  attention mask does not make the entire input embedding causal. Do not supply
  unknown future target values and assume masking prevents leakage.
- The reference `ProbAttention` allocates a dropout module but its update path
  does not apply it. Do not silently insert dropout during extraction.
- `attn='prob'` selects ProbSparse; the reference otherwise selects full attention.
  `InformerStack` does not forward its encoder mask into the subencoders.
  Record these semantics rather than strengthening masks implicitly.
- `seq_len` and `label_len` are accepted by the constructors but do not determine
  slicing inside forward. The supplied tensor lengths matter. The constructor's
  `device` argument does not move parameters; use ordinary PyTorch device transfer.
- Positional embedding tables have default maximum length 5000. Odd `d_model`
  fails in the reference sinusoidal initialization. Length-one ProbSparse inputs
  produce zero sampled keys and fail; do not silently change sample-count rules.
- Heads use floor-divided projection widths, not an explicit divisibility check.
  Minimum lengths for distillation/stack suffixes need further characterization
  in Chunk 3; unrestricted shape support is not claimed.

## Time and utility conventions

Categorical markers use month/day/weekday/hour and, at `freq='t'`, a minute bin
in 0–3 (`minute // 15`), not the raw minute. Continuous time features use a
frequency-dependent channel order. The root helper consumes a DataFrame with
a `date` column despite inconsistent wording in its original docstring; the
categorical path mutates it. No cadence inference or resampling is provided.

Original utilities include ratios undefined for zero denominators and a scaler
without a constant-feature safeguard. `EarlyStopping` uses `np.Inf`, removed in
NumPy 2; `WeekOfYear` uses the legacy pandas `.week` accessor. These are identified
compatibility candidates for Chunk 4, not fixes made in this baseline. Reference
behavior and limitations must remain distinct from later supported package cases.
