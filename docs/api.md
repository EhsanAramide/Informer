# Planned standalone API and reference tensor contract

The package is available from this repository using its standard build metadata.
Chunk 2 establishes imports and reference parity; comprehensive component
docstrings and customization coverage remain Chunk 3 work.

```python
from informer import Informer, InformerStack
from informer.models.attn import ProbAttention, FullAttention, AttentionLayer
from informer.models.encoder import Encoder, EncoderLayer, EncoderStack, ConvLayer
from informer.models.decoder import Decoder, DecoderLayer
from informer.models.embed import DataEmbedding
from informer.utils.masking import TriangularCausalMask, ProbMask
```

Also expose existing definitions through `informer.utils.timefeatures`,
`informer.utils.metrics`, and `informer.utils.tools`. The latter retains
`StandardScaler`, `EarlyStopping`, `adjust_learning_rate`, and `dotdict`.
Expose the existing embedding classes through `informer.models.embed`.
Module paths are supported customization points; underscore-prefixed helpers
remain implementation details. No convenience API may change reference behavior.

## Forward contract

`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None,
dec_self_mask=None, dec_enc_mask=None)` retains the upstream interface.

| Input/output | Shape and meaning |
|---|---|
| `x_enc` | `(B, L_enc, enc_in)` historical encoder channels |
| `x_mark_enc` | `(B, L_enc, M)` corresponding categorical/continuous markers |
| `x_dec` | `(B, L_dec, dec_in)` known context plus future placeholders |
| `x_mark_dec` | `(B, L_dec, M)` markers including known forecast timestamps |
| Prediction | Last `out_len` decoder positions with `c_out` channels |

For intended use, `L_dec = label_len + out_len` and `L_dec >= out_len`.
The upstream model does not enforce this: requesting more output positions than
supplied returns the available suffix. We characterize valid configurations
without claiming a new strict validation API in Chunk 1.

Encoder, decoder, and output channel widths need not be equal. Parameters and
floating inputs must have compatible dtype/device. Categorical markers are cast
to integer indices internally; continuous markers must match the embedding's
floating dtype. The forecast returns a tensor unless `output_attention=True`,
which returns `(prediction, encoder_attention)`. Attention lists are flat for
Informer and nested for InformerStack; they do not include decoder diagnostics.

Preserve constructor signatures and defaults, including the different `e_layers`
forms (integer for Informer, list for InformerStack). Checkpoints use unchanged
state-dictionary names. Save constructor options alongside weights; state dicts
alone do not encode architecture. Whole-object pickle compatibility is not an
acceptance promise across the namespace move.

Masks, RNG behavior, ignored constructor arguments, and known shape limitations
are documented in [architecture.md](architecture.md). Timestamp conversion is
optional and pandas-dependent; core model imports must remain independent of it.
Modern installed-package behavior and full usage examples are later chunks.

## Installation boundary

Build and install the project itself for the core models, or install its `time`
extra when timestamp utilities are needed. Importing `informer`, model modules,
masks, metrics, or general tools does not import pandas. Importing
`informer.utils.timefeatures` without pandas raises a message directing the user
to install `informer[time]`. Historical root-level `models` and `utils` are not
installed and are not compatibility aliases.
