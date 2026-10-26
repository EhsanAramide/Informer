"""Exercise maintained model components through their public namespaces."""

import io

import pytest
import torch

from informer import Informer, InformerStack
from informer.models.attn import AttentionLayer, FullAttention, ProbAttention
from informer.models.embed import DataEmbedding
from informer.utils.masking import TriangularCausalMask


def categorical_markers(batch: int, length: int, minute: bool) -> torch.Tensor:
    """Create valid reference-order categorical timestamp markers."""
    values = torch.zeros(batch, length, 5 if minute else 4, dtype=torch.long)
    values[..., 0] = torch.randint(1, 13, (batch, length))
    values[..., 1] = torch.randint(1, 32, (batch, length))
    values[..., 2] = torch.randint(0, 7, (batch, length))
    values[..., 3] = torch.randint(0, 24, (batch, length))
    if minute:
        values[..., 4] = torch.randint(0, 4, (batch, length))
    return values


@pytest.mark.parametrize(
    ("embed_type", "frequency", "marker_width"),
    [("fixed", "h", 4), ("learned", "t", 5), ("timeF", "h", 4)],
)
def test_embedding_modes(
    embed_type: str, frequency: str, marker_width: int
) -> None:
    """Support fixed, learned, and continuous timestamp embeddings."""
    torch.manual_seed(1)
    embedding = DataEmbedding(3, 8, embed_type, frequency, dropout=0)
    values = torch.randn(2, 7, 3)
    markers = (torch.randn(2, 7, marker_width) if embed_type == "timeF"
               else categorical_markers(2, 7, frequency == "t"))
    result = embedding(values, markers)
    assert result.shape == (2, 7, 8)
    assert torch.isfinite(result).all()
    if embed_type == "fixed":
        assert not embedding.temporal_embedding.month_embed.emb.weight.requires_grad
    elif embed_type == "learned":
        assert embedding.temporal_embedding.month_embed.weight.requires_grad


def test_full_attention_causal_mask_and_custom_scale() -> None:
    """Apply a causal mask and preserve normalized attention rows."""
    torch.manual_seed(2)
    values = torch.randn(2, 6, 2, 4, dtype=torch.float64, requires_grad=True)
    mask = TriangularCausalMask(2, 6, device=values.device)
    attention = FullAttention(True, scale=0.25, attention_dropout=0,
                              output_attention=True)
    output, weights = attention(values, values, values, mask)
    assert output.shape == values.shape
    assert weights.shape == (2, 2, 6, 6)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_probsparse_sampling_replays_only_with_rng_reset() -> None:
    """Document evaluation-time sampling and controlled reproducibility."""
    torch.manual_seed(3)
    values = torch.randn(2, 12, 2, 4)
    attention = ProbAttention(False, factor=2, attention_dropout=0,
                              output_attention=True).eval()
    state = torch.get_rng_state()
    first = attention(values, values, values, None)
    assert not torch.equal(state, torch.get_rng_state())
    torch.set_rng_state(state)
    second = attention(values, values, values, None)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


def test_attention_layer_custom_head_widths() -> None:
    """Expose low-level projected attention with caller-selected head widths."""
    inner = FullAttention(False, attention_dropout=0, output_attention=True)
    layer = AttentionLayer(inner, d_model=8, n_heads=2, d_keys=3, d_values=5)
    query = torch.randn(2, 7, 8, requires_grad=True)
    key = torch.randn(2, 9, 8, requires_grad=True)
    output, weights = layer(query, key, key, None)
    assert output.shape == (2, 7, 8)
    assert weights.shape == (2, 2, 7, 9)
    output.sum().backward()
    assert query.grad is not None and key.grad is not None


@pytest.mark.parametrize("model_class", [Informer, InformerStack])
@pytest.mark.parametrize("distil", [False, True])
def test_model_distillation_and_state_restoration(model_class, distil: bool) -> None:
    """Run both distillation settings and restore an exact state dictionary."""
    options = dict(enc_in=3, dec_in=2, c_out=1, seq_len=12, label_len=6,
                   out_len=3, factor=2, d_model=8, n_heads=2,
                   e_layers=2 if model_class is Informer else [2, 1], d_layers=1,
                   d_ff=16, dropout=0, attn="full", embed="timeF", freq="h",
                   output_attention=True, distil=distil, device=torch.device("cpu"))
    torch.manual_seed(4)
    model = model_class(**options).eval()
    inputs = (torch.randn(2, 12, 3), torch.randn(2, 12, 4),
              torch.randn(2, 9, 2), torch.randn(2, 9, 4))
    expected = model(*inputs)[0]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = model_class(**options).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    actual = restored(*inputs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 3, 1)


def test_dropout_mode_and_seed_control() -> None:
    """Show deterministic evaluation and seeded training-mode dropout replay."""
    embedding = DataEmbedding(2, 8, "timeF", "h", dropout=0.5)
    values = torch.randn(2, 8, 2)
    markers = torch.randn(2, 8, 4)
    embedding.eval()
    torch.testing.assert_close(embedding(values, markers), embedding(values, markers))
    embedding.train()
    torch.manual_seed(22)
    first = embedding(values, markers)
    torch.manual_seed(22)
    second = embedding(values, markers)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_model_smoke(dtype: torch.dtype) -> None:
    """Conditionally verify dtype/device behavior without claiming all GPUs."""
    device = torch.device("cuda")
    model = Informer(3, 2, 1, 12, 6, 3, d_model=8, n_heads=2,
                     e_layers=1, d_layers=1, d_ff=16, dropout=0,
                     attn="full", embed="timeF", freq="h", distil=False,
                     device=device).to(device=device, dtype=dtype).eval()
    result = model(torch.randn(1, 12, 3, device=device, dtype=dtype),
                   torch.randn(1, 12, 4, device=device, dtype=dtype),
                   torch.randn(1, 9, 2, device=device, dtype=dtype),
                   torch.randn(1, 9, 4, device=device, dtype=dtype))
    assert result.device == device and result.dtype == dtype
