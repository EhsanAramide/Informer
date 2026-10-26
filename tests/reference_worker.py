"""Run numerical reference checks in a disposable, isolated Python process.

Invocation: python -I -B reference_worker.py CHECKOUT model NAME DTYPE ATTENTION
or: python -I -B reference_worker.py CHECKOUT behavior.
Only this child process adds the historical checkout to its import path.
"""

import io
import json
from pathlib import Path
import sys


def check_model(name: str, dtype_name: str, attention: str) -> dict:
    """Characterize a model using unequal input widths and a seeded CPU fixture.

    Args:
        name: Reference model class, Informer or InformerStack.
        dtype_name: PyTorch floating-point dtype name.
        attention: Reference attention selector, full or prob.

    Returns:
        JSON-compatible evidence after all assertions succeed. Checkpoints stay
        in memory; decoder future values are zero placeholders, not targets.
    """
    import torch
    from models import model as reference

    torch.manual_seed(100)
    dtype = getattr(torch, dtype_name)
    options = dict(enc_in=5, dec_in=2, c_out=3, seq_len=16, label_len=8,
                   out_len=4, factor=2, d_model=8, n_heads=2,
                   e_layers=2 if name == "Informer" else [2, 1],
                   d_layers=1, d_ff=16, dropout=0, attn=attention,
                   embed="timeF", freq="t", output_attention=True,
                   distil=True, device=torch.device("cpu"))
    model = getattr(reference, name)(**options).to(dtype=dtype).eval()
    inputs = (
        torch.randn(2, 16, 5, dtype=dtype, requires_grad=True),
        torch.randn(2, 16, 5, dtype=dtype),
        torch.cat((torch.randn(2, 8, 2, dtype=dtype),
                   torch.zeros(2, 4, 2, dtype=dtype)), dim=1).requires_grad_(),
        torch.randn(2, 12, 5, dtype=dtype),
    )
    originals = tuple(value.detach().clone() for value in inputs)
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    torch.manual_seed(321)
    before_rng = torch.get_rng_state()
    prediction, diagnostics = model(*inputs)
    assert tuple(prediction.shape) == (2, 4, 3)
    assert prediction.dtype == dtype and torch.isfinite(prediction).all()
    assert torch.equal(before_rng, torch.get_rng_state()) == (attention == "full")
    assert len(diagnostics) == 2
    if name == "InformerStack":
        assert [len(item) for item in diagnostics] == [2, 1]
        diagnostics = [item for group in diagnostics for item in group]
    for item in diagnostics:
        assert item.ndim == 4 and item.shape[:2] == (2, 2)
        assert torch.isfinite(item).all()
        torch.testing.assert_close(item.sum(-1), torch.ones_like(item.sum(-1)))
    prediction.square().mean().backward()
    for value in (inputs[0], inputs[2]):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    for key, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, key
            assert torch.isfinite(parameter.grad).all(), key
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)

    # Evaluate return selection without changing the sampled attention pattern.
    model.output_attention = False
    torch.manual_seed(321)
    torch.testing.assert_close(model(*inputs), prediction, rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = getattr(reference, name)(**options).to(dtype=dtype).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert list(restored.state_dict()) == list(original_state)
    torch.manual_seed(321)
    restored_prediction, _ = restored(*inputs)
    torch.testing.assert_close(restored_prediction, prediction, rtol=0, atol=0)
    return dict(model=name, dtype=dtype_name, prediction_shape=list(prediction.shape),
                checks_passed=True)


def check_behavior() -> dict:
    """Assert sampling, causal masking, BatchNorm, circular padding and limits.

    These checks document reference behavior, including unsupported inputs;
    they do not repair or replace the historical algorithms.
    """
    import torch
    from models.attn import FullAttention, ProbAttention
    from models.embed import PositionalEmbedding, TokenEmbedding
    from models.encoder import ConvLayer

    torch.manual_seed(101)
    values = torch.randn(2, 16, 2, 4)
    attention = ProbAttention(False, factor=2, attention_dropout=0).eval()
    state = torch.get_rng_state()
    first, _ = attention(values, values, values, None)
    assert not torch.equal(state, torch.get_rng_state())
    torch.set_rng_state(state)
    repeated, _ = attention(values, values, values, None)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)

    causal = FullAttention(True, attention_dropout=0, output_attention=True)
    _, scores = causal(values, values, values, None)
    assert torch.count_nonzero(torch.triu(scores, diagonal=1)) == 0
    torch.testing.assert_close(scores.sum(-1), torch.ones_like(scores.sum(-1)))

    distillation = ConvLayer(4)
    assert isinstance(distillation.norm, torch.nn.BatchNorm1d)
    sequence = torch.randn(2, 16, 4) + 3
    before = distillation.norm.running_mean.clone()
    distillation(sequence)
    assert distillation.norm.num_batches_tracked.item() == 1
    assert not torch.equal(before, distillation.norm.running_mean)
    before = distillation.norm.running_mean.clone()
    distillation.eval()(sequence)
    torch.testing.assert_close(before, distillation.norm.running_mean, rtol=0, atol=0)

    embedding = TokenEmbedding(1, 2)
    assert embedding.tokenConv.padding_mode == "circular"
    with torch.no_grad():
        embedding.tokenConv.weight.fill_(1)
        embedding.tokenConv.bias.zero_()
    endpoint = torch.zeros(1, 5, 1)
    endpoint[:, -1] = 1
    torch.testing.assert_close(embedding(endpoint)[:, 0], torch.ones(1, 2))
    try:
        PositionalEmbedding(7)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the reference odd-width positional limitation")
    try:
        singleton = torch.ones(1, 1, 1, 2)
        attention(singleton, singleton, singleton, None)
    except (IndexError, RuntimeError):
        pass
    else:
        raise AssertionError("Expected the reference length-one ProbSparse limitation")
    return dict(checks_passed=True)


def main() -> None:
    """Select the pinned checkout and emit only JSON evidence on stdout."""
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import torch
    from models import model

    assert Path(model.__file__).resolve() == root / "models/model.py"
    torch.set_num_threads(1)
    if sys.argv[2] == "model":
        report = check_model(*sys.argv[3:])
    elif sys.argv[2] == "behavior":
        report = check_behavior()
    else:
        raise ValueError(f"Unknown reference case: {sys.argv[2]}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
