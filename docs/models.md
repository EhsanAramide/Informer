# Model and component customization

This guide describes the maintained model API without changing the reference
algorithms. The synthetic dimensions below demonstrate configuration only; they
are not IonMPX architecture choices or forecasting recommendations.

## Whole models

Encoder, decoder, and output channels are independent constructor choices:

```python
import torch
from informer import Informer

model = Informer(
    enc_in=12,
    dec_in=4,
    c_out=3,
    seq_len=96,
    label_len=48,
    out_len=24,
    d_model=64,
    n_heads=4,
    e_layers=2,
    d_layers=1,
    d_ff=128,
    attn="prob",
    embed="timeF",
    freq="h",
    distil=True,
    output_attention=True,
)

# Continuous hourly markers contain four features in the reference convention.
x_enc = torch.randn(2, 96, 12)
x_mark_enc = torch.randn(2, 96, 4)
x_dec = torch.cat([torch.randn(2, 48, 4), torch.zeros(2, 24, 4)], dim=1)
x_mark_dec = torch.randn(2, 72, 4)
prediction, encoder_attention = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
assert prediction.shape == (2, 24, 3)
```

The future decoder block contains placeholders, not future target observations.
The package deliberately does not assemble it because known-future covariates and
target conventions belong to the caller. Circular value convolution makes correct
decoder-input construction especially important.

`InformerStack` uses the same call, but `e_layers` is a list such as `[3, 2, 1]`.
It returns one diagnostic list per suffix encoder when attention is requested.
`Informer` instead returns one flat list. Attention diagnostics cover encoders,
not decoder attention.

## Attention components

`FullAttention` and `ProbAttention` consume already split tensors shaped
`(batch, sequence, heads, head_width)`. `AttentionLayer` adds model-width linear
projections around either kernel:

```python
from informer.models.attn import AttentionLayer, FullAttention

attention = AttentionLayer(
    FullAttention(mask_flag=False, attention_dropout=0.0),
    d_model=64,
    n_heads=4,
    d_keys=12,
    d_values=16,
)
output, weights = attention(query, key, value, attn_mask=None)
```

ProbSparse calls `torch.randint` during both training and evaluation. For exact
replay, capture/reset the PyTorch RNG state immediately around each forward call.
Its `attention_dropout` argument is retained, but the reference context-update
path does not apply that dropout module. Full attention does apply attention
dropout. Explicit masks are objects exposing a Boolean `.mask` tensor, as provided
by `TriangularCausalMask` and `ProbMask`.

## Timestamp embedding modes

Three existing modes are preserved:

| `embed` | Marker representation | Learnable timestamp projection |
|---|---|---|
| `"fixed"` | categorical indices | no |
| any non-`"fixed"`, non-`"timeF"` string | categorical indices | yes |
| `"timeF"` | continuous features | yes, linear |

Categorical columns are month, day, weekday, hour, plus quarter-hour minute bin
when `freq="t"`. Continuous widths come from the reference frequency map: hourly
uses four, minutely five, secondly six, monthly/annual one, weekly two, and
daily/business-day three. `TimeFeatureEmbedding.embed_type` exists for signature
compatibility but does not influence its linear projection.

`TokenEmbedding` uses width-three circular Conv1d; `PositionalEmbedding` uses a
fixed table with maximum length 5000 by default and requires an even `d_model` in
the reference implementation. `DataEmbedding` adds value, position, and timestamp
components before dropout.

## Distillation, mode, dtype, and device

With `distil=True`, convolution/BatchNorm/ELU/max-pool blocks appear between
encoder layers. BatchNorm updates running state in training mode, so package
parity requires matching both model mode and state. With `distil=False`, encoder
sequence length is not reduced between attention blocks.

The model does not infer or move its dtype/device from the constructor's `device`
argument; that argument is retained but unused. Use normal PyTorch transfer:

```python
model = model.to(device="cuda", dtype=torch.float64)
```

All floating input tensors must match the parameter dtype/device. Categorical
markers are converted to integer indices. CPU float32 and float64 are verified;
CUDA checks are conditional and were skipped in the current environment.

## Checkpoints and reproducibility

The namespace extraction preserves parameter names and state-dictionary keys.
Save constructor options with each state dictionary because weights alone do not
encode widths, depths, attention choice, embedding mode, or distillation. Exact
ProbSparse comparisons additionally require the same RNG state. Whole-model
pickle compatibility across the old generic namespace and new package namespace
is not promised.

Known limitations and ignored arguments are catalogued in
[architecture.md](architecture.md); validation evidence is in
[validation.md](validation.md).
