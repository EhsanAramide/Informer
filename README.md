# Informer for Python

An independently installable package of **Informer** and **InformerStack**, based
on [Informer: Beyond Efficient Transformer for Long Sequence Time-Series
Forecasting](https://arxiv.org/abs/2012.07436) (AAAI 2021). It exposes the original
architecture and components with reference parity tests and modern utility fixes.

## Install

Python 3.11+ is required. From this repository, in a virtual environment:

```sh
python -m pip install .
python -m pip install '.[time]'       # optional pandas calendar utilities
python -m pip install -e '.[time,test]'  # development and full test suite
```

From IonMPX, use `python -m pip install ./externals/Informer`.
This fork is not published to PyPI; install this checkout or its built artifacts.
Core dependencies are PyTorch >=2,<3 and NumPy >=2,<3; the optional time extra
adds pandas >=2,<4. Actual tested versions are in [validation](docs/validation.md).

## Use the models

```python
import torch
from informer import Informer, InformerStack

torch.manual_seed(7)
model = Informer(
    enc_in=5, dec_in=2, c_out=3, seq_len=16, label_len=8, out_len=4,
    d_model=8, n_heads=2, e_layers=2, d_layers=1, d_ff=16,
    embed="timeF", freq="h", dropout=0,
).eval()
history = torch.randn(2, 16, 5)
decoder = torch.cat((history[:, -8:, :2], torch.zeros(2, 4, 2)), dim=1)
# Synthetic continuous markers; use known timestamps for real observations.
forecast = model(history, torch.zeros(2, 16, 4), decoder, torch.zeros(2, 12, 4))
assert forecast.shape == (2, 4, 3)
```

Input layout is `(batch, time, channels)`. Encoder, decoder, and output widths may
differ. The caller supplies decoder history and placeholders; never put unknown
future targets into decoder inputs. InformerStack uses a list of encoder depths,
such as `e_layers=[2, 1]`, with the same forward interface.

Public components live in `informer.models` (attention, embeddings, encoder,
decoder) and `informer.utils` (masks, calendars, metrics, scaling, training
helpers). No top-level models/utils aliases are installed. See the
[API contract](docs/api.md), [model guide](docs/models.md), and
[utility guide](docs/utilities.md).

## Runnable examples

After installation, use these scripts or their absolute paths from any directory.
They need no datasets or network downloads:

```sh
python examples/forecast.py
python examples/forecast.py --stack
python examples/components.py
python examples/components.py --time  # requires the time extra
```

Examples check shapes and backpropagation on synthetic inputs. Dimensions are
illustrative; these checks do not establish forecasting performance.

## Reproducibility and limits

- ProbSparse samples keys even in evaluation. Reset PyTorch RNG state for exact
  replay; evaluation mode alone does not make sampling deterministic.
- Distillation uses BatchNorm, whose training behavior depends on batch statistics.
  Value embeddings use circular temporal convolution.
- Save constructor settings alongside state dictionaries. Parameter names and
  state keys match the reference; whole-object pickle migration is not promised.
- Use ordinary PyTorch dtype/device transfer and matching floating inputs.
  Constructor device, seq_len, and label_len arguments do not move tensors or
  construct decoder inputs.
- Reference limits include odd positional widths, length-one ProbSparse,
  ignored masks in some paths, and unguarded metric/scaler divisions. Read the
  [scientific contract](docs/architecture.md) and utility guide before adaptation.

## Development and provenance

```sh
python -m pytest -q tests
python -m build
```

Maintained code lives under src/informer. Root models/, utils/, benchmark runners,
scripts, and requirements remain historical reference material excluded from
installed library imports. Their dependency pins are not installation instructions.
The original README, including its benchmark claims, is retained as
[README.upstream.md](README.upstream.md).
Reference revision: `29f2a739226a509202a092b464163da81fa74960`.

The root LICENSE contains Apache 2.0 text, while the original README badge says
CC BY-NC-SA. This discrepancy is recorded without selecting new terms; see
[attribution](ATTRIBUTION.md). No publication or benchmark reproduction is implied.

The [development record](docs/development.md) tracks review status.
IonMPX temporal integration, representation packing, optimization, and experiment
orchestration are separate work.
