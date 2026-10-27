"""Synthetic forecasting and gradients: python forecast.py [--stack]."""
import argparse
import torch
from informer import Informer, InformerStack


def main():
    """Forecast with distinct channel widths and history-only decoder context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', action='store_true', help='Use InformerStack')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model_type = InformerStack if args.stack else Informer
    model = model_type(
        enc_in=5, dec_in=2, c_out=3, seq_len=16, label_len=8, out_len=4,
        d_model=8, n_heads=2, e_layers=[2, 1] if args.stack else 2,
        d_layers=1, d_ff=16, factor=2, dropout=0, embed='timeF', freq='h',
        output_attention=True, device=torch.device('cpu'),
    ).eval()
    history = torch.randn(2, 16, 5, requires_grad=True)
    # Decoder future positions contain placeholders, never future targets.
    decoder = torch.cat((history[:, -8:, :2], torch.zeros(2, 4, 2)), dim=1)
    # Synthetic continuous markers; real observations require real timestamps.
    forecast, attention = model(history, torch.zeros(2, 16, 4),
                                decoder, torch.zeros(2, 12, 4))
    assert forecast.shape == (2, 4, 3)
    forecast.square().mean().backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert model.projection.weight.grad is not None
    print(f'{model_type.__name__}: forecast={tuple(forecast.shape)}, '
          f'attention_groups={len(attention)}, gradients verified')


if __name__ == '__main__':
    main()
