"""Train the Go2 walking policy on a GPU with MJX + Brax PPO (e.g. on Kaggle).

    python scripts/train_mjx.py --name tpms_eng_mjx                       # TPMS foot + energy-phase reward
    python scripts/train_mjx.py --name solid_eng_mjx --foot solid
    python scripts/train_mjx.py --name test_mjx --steps 200000 --envs 64  # tiny CPU test
    python scripts/train_mjx.py --name tpms_eng_mjx --resume              # continue from last checkpoint
    python scripts/train_mjx.py --name tpms_eng_mjx_v3 --init-from tpms_eng_mjx_v2 --steps 50000000 --eng-weight 3e-3
                                                         # fine-tune a walking policy with a new reward
    python scripts/train_mjx.py --name rigid_power_x --foot rigid --init-from rigid_base --eng-type power --eng-weight 1e-3
                                                         # other energy term (phase / power / torque)

--resume restores the network weights + observation normalizer of the newest
checkpoint and trains --steps more (Brax restarts its step counter and optimizer).
--init-from does the same from another run's newest checkpoint (a run name under runs/ or
init/, or a folder path), with this run's own foot and reward. Needed for strong energy-phase
weights: from scratch, the random early gait costs so much that the policy learns to stand still.

Writes runs/<name>/: config.json, policy_mjx.npz (final), ckpt_mjx_<steps>.npz,
Brax/orbax checkpoints in ckpt/, and TensorBoard logs in tb/. Copy the folder to
your PC and use scripts/play.py exactly as for SB3 runs.
"""
import argparse
import functools
import os
import time

import jax
import numpy as np

from go2tpms.compliant_foot import FOOT_TYPES
from go2tpms.go2_env import ENG_TYPES
from go2tpms.go2_mjx_env import Go2MjxEnv
from go2tpms.numpy_policy import NumpyPolicy, export_brax_policy
from go2tpms.paths import RUNS_DIR
from go2tpms.runs import latest_orbax_checkpoint, load_config, run_dir, save_config


def find_init_checkpoint(init_from, out_root):
    """Newest orbax checkpoint of a run given by name (runs/, init/ or --out) or folder path."""
    candidates = [init_from, os.path.join(out_root, init_from), str(RUNS_DIR / init_from),
                  str(RUNS_DIR.parent / "init" / init_from)]
    for c in candidates:
        ckpt = latest_orbax_checkpoint(os.path.abspath(os.path.join(c, "ckpt")))
        if ckpt:
            return ckpt
    raise SystemExit(f"No checkpoint found for --init-from {init_from} (looked in {candidates})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--foot", default="tpms60", choices=list(FOOT_TYPES))
    parser.add_argument("--eng-weight", type=float, default=3e-3)
    parser.add_argument("--eng-type", default="phase", choices=list(ENG_TYPES),
                        help="energy term: phase (paper Eq. 15), power (|torque*speed|), torque (torque^2)")
    parser.add_argument("--steps", type=int, default=100_000_000)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--foot-armature", type=float, default=-1.0,
                        help="override foot slide armature (needed for --sim-dt 0.002)")
    parser.add_argument("--action-scale", type=float, default=0.5, help="rad per unit action")
    parser.add_argument("--evals", type=int, default=20, help="number of eval/checkpoint points")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="PPO learning rate (try 1e-4 if a fine-tune collapses)")
    parser.add_argument("--out", default=str(RUNS_DIR), help="folder for run folders")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", help="start from another run's newest checkpoint (fine-tune)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Imported here so `python train_mjx.py -h` is fast.
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from tensorboardX import SummaryWriter

    out = run_dir(args.name, root=args.out)
    ckpt_dir = os.path.abspath(os.path.join(out, "ckpt"))  # orbax needs an absolute path
    os.makedirs(ckpt_dir, exist_ok=True)

    saved = load_config(out)
    if args.resume and saved:
        args.foot, args.eng_weight = saved["foot"], saved["eng_weight"]
        args.eng_type = saved.get("eng_type", "phase")
        args.sim_dt, args.foot_armature = saved.get("sim_dt", 0.001), saved.get("foot_armature", -1.0)
        args.action_scale = saved.get("action_scale", 0.25)  # older runs used 0.25
    init_from = saved.get("init_from") if args.resume and saved else args.init_from
    config = {"foot": args.foot, "eng_weight": args.eng_weight, "eng_type": args.eng_type, "trainer": "mjx",
              "sim_dt": args.sim_dt, "foot_armature": args.foot_armature,
              "action_scale": args.action_scale, "lr": args.lr, "seed": args.seed}
    if init_from:
        config["init_from"] = init_from
    save_config(out, config)

    overrides = dict(foot_type=args.foot, eng_weight=args.eng_weight, eng_type=args.eng_type, sim_dt=args.sim_dt,
                     foot_armature=args.foot_armature, action_scale=args.action_scale)
    env = Go2MjxEnv(config_overrides=overrides)
    eval_env = Go2MjxEnv(config_overrides=overrides)

    restore = None
    if args.resume:
        restore = latest_orbax_checkpoint(ckpt_dir)
        if restore is None:
            raise SystemExit(f"No checkpoint to resume in {ckpt_dir}")
        print(f"Resuming from {restore}")
    elif args.init_from:
        restore = find_init_checkpoint(args.init_from, args.out)
        init_scale = load_config(os.path.dirname(os.path.dirname(restore))).get("action_scale")
        if init_scale is not None and init_scale != args.action_scale:
            raise SystemExit(f"{args.init_from} was trained with action_scale={init_scale}, "
                             f"this run uses {args.action_scale}")
        print(f"Fine-tuning from {restore}")

    n_dev = jax.local_device_count()
    num_minibatches = 32
    envs = max(n_dev, args.envs // n_dev * n_dev)
    batch_size = max(1, envs // num_minibatches)  # batch_size * minibatches = envs
    print(f"Devices: {jax.devices()}  envs={envs}  foot={args.foot}  eng={args.eng_type} {args.eng_weight}  "
          f"steps={args.steps:,}  sim_dt={args.sim_dt}  action_scale={args.action_scale}  lr={args.lr}  seed={args.seed}")

    writer = SummaryWriter(os.path.join(out, "tb"))
    t_start = time.time()
    episode_len = env._config.episode_length

    def progress(step, metrics):
        elapsed = time.time() - t_start
        for k, v in metrics.items():
            v = float(v)
            writer.add_scalar(k, v, step)
            # Brax reports episode sums; also log per-step means like train.py does.
            if k.startswith("eval/episode_rew/") or k.startswith("eval/episode_env/"):
                length = float(metrics.get("eval/avg_episode_length", episode_len))
                writer.add_scalar(k.replace("eval/episode_", ""), v / max(length, 1.0), step)
        writer.flush()
        rew = float(metrics.get("eval/episode_reward", np.nan))
        length = float(metrics.get("eval/avg_episode_length", np.nan))
        speed = float(metrics.get("eval/episode_env/speed", np.nan)) / max(length, 1.0)
        print(f"[{elapsed / 60:6.1f} min] step {step:>11,}  episode reward {rew:8.2f}  "
              f"length {length:6.1f}  mean speed {speed:5.2f} m/s", flush=True)

    checked = []

    def save_policy(step, make_policy, params):
        path = os.path.join(out, f"ckpt_mjx_{step}.npz")
        export_brax_policy(params, path, env.action_size)
        if not checked:  # once per run: exported NumPy policy == Brax policy
            obs = np.random.default_rng(0).normal(size=(32, env.observation_size)).astype(np.float32)
            a_brax, _ = make_policy(params, deterministic=True)(jax.numpy.asarray(obs), jax.random.PRNGKey(0))
            a_np = np.stack([NumpyPolicy(path)(o) for o in obs])
            err = float(np.max(np.abs(np.asarray(a_brax) - a_np)))
            print(f"Export check: max |NumPy policy - Brax policy| = {err:.1e}  "
                  f"{'OK' if err < 1e-4 else 'MISMATCH'}", flush=True)
            checked.append(err)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )
    make_policy, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_timesteps=args.steps,
        num_envs=envs,
        num_eval_envs=128,
        episode_length=episode_len,
        action_repeat=1,
        unroll_length=20,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=4,
        learning_rate=args.lr,
        entropy_cost=1e-2,
        discounting=0.97,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        reward_scaling=1.0,
        normalize_observations=True,
        # (The 20 s limit is handled as 'truncation' by Brax's EpisodeWrapper; GAE
        # bootstraps through it, so a time-out is not treated as a failure.)
        deterministic_eval=True,
        num_evals=args.evals,
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=save_policy,
        save_checkpoint_path=ckpt_dir,
        restore_checkpoint_path=restore,
        seed=args.seed,
    )
    export_brax_policy(params, os.path.join(out, "policy_mjx.npz"), env.action_size)
    writer.close()
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Saved to {out}/")


if __name__ == "__main__":
    main()
