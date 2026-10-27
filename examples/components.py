"""Customize attention; optionally use calendar features with --time."""
import argparse
import torch
from informer.models.attn import AttentionLayer, FullAttention
from informer.models.embed import DataEmbedding


def main():
    """Compose public components with synthetic or known-timestamp markers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time', action='store_true', help='Requires informer[time]')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    if args.time:
        from informer.utils.timefeatures import time_features
        import pandas as pd
        frame = pd.DataFrame({'date': pd.date_range('2002-12-21', periods=8, freq='h')})
        markers = torch.as_tensor(time_features(frame, freq='h'), dtype=torch.float32)[None]
    else:
        markers = torch.zeros(1, 8, 4)
    values = torch.randn(1, 8, 3, requires_grad=True)
    embedding = DataEmbedding(3, 8, embed_type='timeF', freq='h', dropout=0)
    attention = AttentionLayer(
        FullAttention(mask_flag=False, attention_dropout=0, output_attention=True),
        d_model=8, n_heads=2, d_keys=3, d_values=5,
    )
    states = embedding(values, markers)
    output, weights = attention(states, states, states, attn_mask=None)
    assert output.shape == (1, 8, 8) and weights.shape == (1, 2, 8, 8)
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    print(f'Components: output={tuple(output.shape)}, calendar={args.time}, gradients verified')


if __name__ == '__main__':
    main()
