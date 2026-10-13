"""Brax PPO policies as plain NumPy: export after GPU training, run anywhere.

The exported .npz holds the policy MLP weights and the observation normalizer,
so scripts/play.py can evaluate GPU-trained policies in the CPU env without JAX.
"""
import numpy as np


def export_brax_policy(params, path, action_size):
    """Save Brax PPO params (normalizer, policy, value) as an .npz for NumpyPolicy."""
    normalizer, policy = params[0], params[1]
    layers = policy["params"]
    names = sorted(layers, key=lambda k: int(k.split("_")[-1]))  # hidden_0, hidden_1, ...
    arrays = {"obs_mean": np.asarray(normalizer.mean), "obs_std": np.asarray(normalizer.std),
              "n_layers": np.array(len(names)), "action_size": np.array(action_size)}
    for i, name in enumerate(names):
        arrays[f"w{i}"] = np.asarray(layers[name]["kernel"])
        arrays[f"b{i}"] = np.asarray(layers[name]["bias"])
    np.savez(path, **arrays)


class NumpyPolicy:
    """Deterministic Brax PPO policy (swish MLP, tanh-normal output) in NumPy."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        self.action_size = int(z["action_size"])

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x * 0.5 * (1.0 + np.tanh(0.5 * x))  # swish = x * sigmoid(x), overflow-free
        return np.tanh(x[: self.action_size])  # tanh-normal mode = tanh(mean)
