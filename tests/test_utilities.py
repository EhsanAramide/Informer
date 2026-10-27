"""Numerical and behavioral contracts for the preserved utility modules."""

import numpy as np
import pytest
import torch

from informer.utils import metrics
from informer.utils.masking import ProbMask, TriangularCausalMask
from informer.utils.tools import EarlyStopping, StandardScaler, adjust_learning_rate, dotdict


def test_metrics_numerical_values():
    """Compare reductions with independently calculated fixture values."""
    true = np.array([[1., 2.], [3., 4.]])
    pred = np.array([[2., 2.], [1., 8.]])
    np.testing.assert_allclose(metrics.metric(pred, true),
                               [1.75, 5.25, np.sqrt(5.25), 2 / 3, 11 / 18])
    assert metrics.RSE(pred, true) == pytest.approx(np.sqrt(21 / 5))
    assert metrics.CORR(pred, true) == pytest.approx(0)
    assert metrics.CORR(true, true) == pytest.approx(np.sqrt(2))


def test_metric_undefined_cases():
    """Keep visible division warnings and undefined results."""
    with pytest.warns(RuntimeWarning):
        assert np.isinf(metrics.MAPE(np.ones(2), np.zeros(2)))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(metrics.RSE(np.ones(2), np.ones(2)))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(metrics.CORR(np.ones((2, 2)), np.ones((2, 2))))


@pytest.mark.parametrize('tensor', [False, True])
def test_scaler_roundtrip_and_last_feature(tensor):
    """Preserve population scaling, input values, gradients, and inverse fallback."""
    data = np.array([[1., 4.], [3., 8.], [5., 12.]])
    scaler = StandardScaler()
    assert scaler.fit(data) is None
    np.testing.assert_allclose(scaler.mean, [3, 8])
    np.testing.assert_allclose(scaler.std, np.sqrt([8 / 3, 32 / 3]))
    values = torch.tensor(data, requires_grad=True) if tensor else data.copy()
    result = scaler.transform(values)
    restored = scaler.inverse_transform(result)
    if tensor:
        torch.testing.assert_close(restored, values)
        restored.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))
        np.testing.assert_allclose(values.detach(), data)
        torch.testing.assert_close(scaler.inverse_transform(result[:, -1:]), values[:, -1:])
    else:
        np.testing.assert_allclose(restored, data)
        np.testing.assert_array_equal(values, data)
        np.testing.assert_allclose(scaler.inverse_transform(result[:, -1:]), data[:, -1:])


def test_scaler_limits():
    """Document constant scales and unsupported tensor-fitted tensor transforms."""
    scaler = StandardScaler()
    scaler.fit(np.ones((3, 2)))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(scaler.transform(np.ones((3, 2)))).all()
    scaler.fit(torch.ones(3, 2))
    with pytest.raises(TypeError):
        scaler.transform(torch.ones(3, 2))


def test_early_stopping_checkpoint_and_ties(tmp_path):
    """Exercise improvement, equal-loss reset, patience, and saved model state."""
    model = torch.nn.Linear(2, 1)
    stopper = EarlyStopping(patience=2, verbose=True)
    assert stopper.val_loss_min == float('inf')
    stopper(2., model, str(tmp_path))
    saved = torch.load(tmp_path / 'checkpoint.pth', weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(saved[key], value)
    stopper(3., model, str(tmp_path))
    assert stopper.counter == 1
    stopper(2., model, str(tmp_path))
    assert stopper.counter == 0
    stopper(3., model, str(tmp_path))
    stopper(4., model, str(tmp_path))
    assert stopper.early_stop
    stopper(1., model, str(tmp_path))
    assert stopper.early_stop and stopper.val_loss_min == 1
    bounded = EarlyStopping(delta=0.1)
    bounded(2., model, str(tmp_path))
    bounded(1.95, model, str(tmp_path))
    assert bounded.counter == 1


def test_learning_rate_and_dotdict():
    """Update all groups according to reference schedules and dictionary behavior."""
    optimizer = torch.optim.SGD([{'params': [torch.nn.Parameter(torch.ones(1))]},
                                 {'params': [torch.nn.Parameter(torch.ones(1))]}], lr=1)
    args = dotdict(lradj='type1', learning_rate=0.1)
    adjust_learning_rate(optimizer, 3, args)
    assert [group['lr'] for group in optimizer.param_groups] == [0.025, 0.025]
    args.lradj = 'type2'
    for epoch, expected in [(2, 5e-5), (3, 5e-5), (4, 1e-5), (6, 5e-6),
                            (8, 1e-6), (10, 5e-7), (15, 1e-7), (20, 5e-8)]:
        adjust_learning_rate(optimizer, epoch, args)
        assert all(group['lr'] == expected for group in optimizer.param_groups)
    del args.lradj
    assert args.lradj is None
    with pytest.raises(KeyError):
        del args.lradj
    with pytest.raises(UnboundLocalError):
        adjust_learning_rate(optimizer, 1, args)


def test_mask_endpoints():
    """Select different causal query rows independently for each batch/head."""
    causal = TriangularCausalMask(2, 4).mask
    expected = torch.arange(4)[None, :] > torch.arange(4)[:, None]
    assert torch.equal(causal, expected.expand(2, 1, 4, 4))
    index = torch.tensor([[[0, 2], [1, 3]], [[3, 1], [2, 0]]])
    scores = torch.empty(2, 2, 2, 4)
    selected = ProbMask(2, 2, 4, index, scores).mask
    assert torch.equal(selected, torch.arange(4) > index.unsqueeze(-1))
