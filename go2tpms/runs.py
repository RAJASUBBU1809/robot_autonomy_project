"""Run folders (runs/<name>/), their config.json, and checkpoint lookup.

SB3 runs (scripts/train.py):      policy.zip, vecnormalize.pkl,
                                  ckpt_<steps>_steps.zip + ckpt_vecnormalize_<steps>_steps.pkl
MJX runs (scripts/train_mjx.py):  policy_mjx.npz, ckpt_mjx_<steps>.npz, ckpt/<steps>/ (orbax)
"""
import glob
import json
import os
import re

from .paths import RUNS_DIR


def run_dir(name, root=RUNS_DIR):
    return os.path.join(str(root), name)


def load_config(out):
    path = os.path.join(out, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_config(out, config):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _newest(pattern, regex):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(regex, os.path.basename(p)).group(1)))


def latest_sb3_checkpoint(out):
    """(policy path, vecnormalize path) of the newest SB3 checkpoint, or None."""
    newest = _newest(os.path.join(out, "ckpt_*_steps.zip"), r"ckpt_(\d+)_steps")
    if newest is None:
        return None
    steps = re.search(r"ckpt_(\d+)_steps", os.path.basename(newest)).group(1)
    return newest, os.path.join(out, f"ckpt_vecnormalize_{steps}_steps.pkl")


def sb3_policy_files(out, ckpt=None):
    """ckpt: None (final policy), 'latest', or a step count."""
    if ckpt is None:
        return os.path.join(out, "policy.zip"), os.path.join(out, "vecnormalize.pkl")
    if ckpt == "latest":
        found = latest_sb3_checkpoint(out)
        if found is None:
            raise SystemExit(f"No checkpoints in {out}")
        return found
    return (os.path.join(out, f"ckpt_{ckpt}_steps.zip"),
            os.path.join(out, f"ckpt_vecnormalize_{ckpt}_steps.pkl"))


def mjx_policy_file(out, ckpt=None):
    """ckpt: None (final policy), 'latest', or a step count."""
    if ckpt is None:
        return os.path.join(out, "policy_mjx.npz")
    if ckpt == "latest":
        found = _newest(os.path.join(out, "ckpt_mjx_*.npz"), r"ckpt_mjx_(\d+)")
        if found is None:
            raise SystemExit(f"No checkpoints in {out}")
        return found
    return os.path.join(out, f"ckpt_mjx_{ckpt}.npz")


def latest_orbax_checkpoint(ckpt_dir):
    """Newest Brax/orbax checkpoint folder (named by step count), or None."""
    steps = [d for d in glob.glob(os.path.join(ckpt_dir, "*")) if re.fullmatch(r"\d+", os.path.basename(d))]
    return max(steps, key=lambda d: int(os.path.basename(d))) if steps else None
