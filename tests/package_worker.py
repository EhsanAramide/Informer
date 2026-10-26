"""Compare maintained package code to the frozen reference in isolation."""

import json
from pathlib import Path
import sys


def flatten_diagnostics(value):
    """Yield tensors from either model's flat or nested attention structure."""
    for item in value:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def main() -> None:
    """Run exact output and gradient parity with controlled random state."""
    root = Path(sys.argv[1]).resolve()
    name, dtype_name, attention = sys.argv[2:5]
    scenario = sys.argv[5] if len(sys.argv) > 5 else "base"
    sys.path.insert(0, str(root))
    import torch
    from models import model as reference

    # Namespaced imports coexist safely with the reference's generic imports.
    sys.path.insert(0, str(root / "src"))
    from informer.models import model as packaged

    assert Path(reference.__file__).resolve() == root / "models/model.py"
    assert Path(packaged.__file__).resolve() == root / "src/informer/models/model.py"
    torch.set_num_threads(1)
    dtype = getattr(torch, dtype_name)
    settings = {
        "base": ("timeF", "t", True),
        "fixed_no_distil": ("fixed", "h", False),
        "learned_distil": ("learned", "t", True),
        "timef_no_distil": ("timeF", "h", False),
    }
    embed, frequency, distil = settings[scenario]
    options = dict(enc_in=5, dec_in=2, c_out=3, seq_len=16, label_len=8,
                   out_len=4, factor=2, d_model=8, n_heads=2,
                   e_layers=2 if name == "Informer" else [2, 1],
                   d_layers=1, d_ff=16, dropout=0, attn=attention,
                   embed=embed, freq=frequency, output_attention=True,
                   distil=distil, device=torch.device("cpu"))
    torch.manual_seed(77)
    old = getattr(reference, name)(**options).to(dtype=dtype).eval()
    new = getattr(packaged, name)(**options).to(dtype=dtype).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    assert list(old.state_dict()) == list(new.state_dict())

    def markers(length):
        if embed == "timeF":
            width = 5 if frequency == "t" else 4
            return torch.randn(2, length, width, dtype=dtype)
        values = torch.zeros(2, length, 5 if frequency == "t" else 4,
                             dtype=torch.long)
        values[..., 0] = torch.randint(1, 13, (2, length))
        values[..., 1] = torch.randint(1, 32, (2, length))
        values[..., 2] = torch.randint(0, 7, (2, length))
        values[..., 3] = torch.randint(0, 24, (2, length))
        if frequency == "t":
            values[..., 4] = torch.randint(0, 4, (2, length))
        return values

    base = [torch.randn(2, 16, 5, dtype=dtype),
            markers(16),
            torch.cat((torch.randn(2, 8, 2, dtype=dtype),
                       torch.zeros(2, 4, 2, dtype=dtype)), dim=1),
            markers(12)]
    old_inputs = [value.clone().requires_grad_(index in (0, 2))
                  for index, value in enumerate(base)]
    new_inputs = [value.clone().requires_grad_(index in (0, 2))
                  for index, value in enumerate(base)]
    torch.manual_seed(901)
    old_output, old_attention = old(*old_inputs)
    torch.manual_seed(901)
    new_output, new_attention = new(*new_inputs)
    torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
    old_items = list(flatten_diagnostics(old_attention))
    new_items = list(flatten_diagnostics(new_attention))
    assert len(new_items) == len(old_items)
    for actual, expected in zip(new_items, old_items):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old_output.square().mean().backward()
    new_output.square().mean().backward()
    for actual, expected in zip(new_inputs, old_inputs):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    for (new_name, actual), (old_name, expected) in zip(
            new.named_parameters(), old.named_parameters()):
        assert new_name == old_name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    print(json.dumps({"model": name, "dtype": dtype_name,
                      "attention": attention, "scenario": scenario,
                      "checks_passed": True}))


if __name__ == "__main__":
    main()
