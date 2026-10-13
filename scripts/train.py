"""Train a Go2 walking policy with PPO (Stable-Baselines3) on the CPU.

    python scripts/train.py --name tpms_eng                      # TPMS foot + energy-phase reward
    python scripts/train.py --name solid_base --foot solid --eng-weight 0
    python scripts/train.py --name test --steps 200000 --envs 2  # quick smoke test
    python scripts/train.py --name tpms_eng --resume             # continue a stopped run

Checkpoints, the final policy and TensorBoard logs are saved in runs/<name>/.
Watch training with:  tensorboard --logdir runs
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from go2tpms.compliant_foot import FOOT_TYPES
from go2tpms.go2_env import Go2Env
from go2tpms.runs import latest_sb3_checkpoint, load_config, run_dir, save_config


class InfoLogger(BaseCallback):
    """Logs the mean of each reward term and of power/speed over every rollout."""

    def _on_rollout_start(self):
        self.sums = defaultdict(float)
        self.count = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info["reward_terms"].items():
                self.sums["rew/" + k] += v
            self.sums["env/mech_power_W"] += info["mech_power"]
            self.sums["env/speed"] += info["speed"]
            self.sums["env/speed_error"] += abs(info["speed"] - info["cmd"])
            self.sums["env/max_foot_compression_mm"] += float(np.max(info["foot_compression_mm"]))
            self.count += 1
        return True

    def _on_rollout_end(self):
        for k, v in self.sums.items():
            self.logger.record(k, v / max(1, self.count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--foot", default="tpms60", choices=list(FOOT_TYPES))
    parser.add_argument("--eng-weight", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=20_000_000,
                        help="total env steps (with --resume: extra steps)")
    # 6 is fastest on a 4-core / 8-thread laptop; more envs than cores slows down.
    parser.add_argument("--envs", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--action-scale", type=float, default=0.5, help="rad per unit action")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<name>/")
    args = parser.parse_args()

    out = run_dir(args.name)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)  # the envs use the other cores

    saved = load_config(out)
    if args.resume and saved:
        args.foot, args.eng_weight = saved["foot"], saved["eng_weight"]
        args.action_scale = saved.get("action_scale", 0.25)  # older runs used 0.25
    else:
        save_config(out, {"foot": args.foot, "eng_weight": args.eng_weight, "trainer": "sb3",
                          "action_scale": args.action_scale})

    env_kwargs = dict(foot_type=args.foot, eng_weight=args.eng_weight, action_scale=args.action_scale)
    venv = make_vec_env(Go2Env, n_envs=args.envs, env_kwargs=env_kwargs,
                        vec_env_cls=SubprocVecEnv if args.envs > 1 else None)

    if args.resume:
        policy_path = os.path.join(out, "policy.zip")
        norm_path = os.path.join(out, "vecnormalize.pkl")
        if not os.path.exists(policy_path):  # stopped hard: use the newest checkpoint
            ckpt = latest_sb3_checkpoint(out)
            if ckpt is None:
                raise SystemExit(f"Nothing to resume in {out}/")
            policy_path, norm_path = ckpt
        print(f"Resuming from {policy_path}")
        venv = VecNormalize.load(norm_path, venv)
        model = PPO.load(policy_path, env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        n_steps = max(64, 8192 // args.envs // 4 * 4)  # ~8k samples per update
        model = PPO(
            "MlpPolicy", venv,
            n_steps=n_steps,
            batch_size=n_steps * args.envs // 4, n_epochs=5,  # 4 minibatches per epoch
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=[256, 256, 128],
                               activation_fn=torch.nn.ELU, log_std_init=-1.0),
            tensorboard_log=os.path.join(out, "tb"),
            device="cpu", verbose=1,
        )

    checkpoint = CheckpointCallback(save_freq=max(1, 500_000 // args.envs), save_path=out,
                                    name_prefix="ckpt", save_vecnormalize=True)
    print(f"Training '{args.name}' on {args.envs} envs: foot={args.foot}, "
          f"eng_weight={args.eng_weight}, steps={args.steps:,}")
    try:
        model.learn(total_timesteps=args.steps, callback=[checkpoint, InfoLogger()],
                    reset_num_timesteps=not args.resume, tb_log_name="ppo")
    finally:  # also saves if you stop with Ctrl+C
        model.save(os.path.join(out, "policy.zip"))
        venv.save(os.path.join(out, "vecnormalize.pkl"))
        print(f"Saved to {out}/")
    venv.close()


if __name__ == "__main__":  # required on Windows for parallel envs
    main()
