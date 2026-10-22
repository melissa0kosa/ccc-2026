import numpy as np
import pandas as pd
import torch

from ccc_2026 import io, numpy_nn
from ccc_2026.cv import cross_validate
from ccc_2026.nn import MLP, fit, predict, predict_classes, seed_everything


def test_compare_to_sample_tolerates_float_noise(tmp_path):
    a = io.write_lines(tmp_path / "a.out", ["1,2.0000001", "x"])
    b = io.write_lines(tmp_path / "b.out", ["1,2", "x"])
    assert io.compare_to_sample(a, b) == []
    c = io.write_lines(tmp_path / "c.out", ["1,3", "x"])
    assert io.compare_to_sample(c, b)


def test_read_csv_sniffs_separator(tmp_path):
    (tmp_path / "s.in").write_text("a;b\n1;2\n")
    (tmp_path / "w.in").write_text("a b\n1 2\n")
    assert list(io.read_csv(tmp_path / "s.in").columns) == ["a", "b"]
    assert io.read_csv(tmp_path / "w.in")["b"].iloc[0] == 2


def test_write_csv_uses_lf(tmp_path):
    p = io.write_csv(tmp_path / "o.csv", pd.DataFrame({"x": [1, 2]}))
    assert b"\r\n" not in p.read_bytes()


def test_numpy_forward_matches_torch():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    x = np.random.default_rng(0).normal(size=(5, 4))
    ws, bs = numpy_nn.from_torch_state_dict(net.state_dict())
    ours = numpy_nn.forward(x, ws, bs, "relu", "linear")
    ref = net(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    assert np.allclose(ours, ref, atol=1e-5)


def test_numpy_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 50, dtype=float)
    y = (X[:, 0] != X[:, 1]).astype(int)
    m = numpy_nn.NumpyMLP([2, 16, 2], output="softmax", seed=1).fit(X, y, epochs=300, lr=1e-2,
                                                                     batch_size=16)
    assert (m.predict(X) == y).mean() == 1.0


def test_torch_fit_regression_and_classification():
    seed_everything(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 0.1 * rng.normal(size=600)
    model, _ = fit(MLP(5, 1, hidden=(64, 64)), X[:500], y[:500], X[500:], y[500:],
                   epochs=60, verbose=0)
    rmse = np.sqrt(np.mean((predict(model, X[500:]).ravel() - y[500:]) ** 2))
    assert rmse < 0.6

    yc = (y > 0).astype(int)
    model, _ = fit(MLP(5, 2, hidden=(64,)), X[:500], yc[:500], X[500:], yc[500:],
                   task="classification", epochs=60, verbose=0)
    assert (predict_classes(model, X[500:]) == yc[500:]).mean() > 0.9


def test_cross_validate_runs():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 2
    score, oof = cross_validate(Ridge(), X, y, "rmse", "regression", verbose=False)
    assert score < 0.1 and len(oof) == 100
