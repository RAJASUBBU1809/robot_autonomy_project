"""Watch a trained policy, or measure its power use like the paper's Table 5.

    python scripts/play.py --name tpms_eng --speed 0.6      # watch in the viewer
    python scripts/play.py --name tpms_eng --evaluate       # table for 0.2 ... 1.0 m/s
    python scripts/play.py --name tpms_eng --ckpt latest    # use the newest checkpoint
    python scripts/play.py --name tpms_eng --ckpt 3000000 --evaluate
    python scripts/play.py --name tpms_eng --foot solid --evaluate  # same policy, other foot

The foot type defaults to the one the run was trained with (runs/<name>/config.json).
Works for SB3 runs (train.py) and GPU runs (train_mjx.py, policy_mjx.npz) alike;
both are evaluated in the same CPU env, so their numbers are comparable.
"""
import argparse
import time

import numpy as np

from go2tpms.compliant_foot import FOOT_TYPES
from go2tpms.go2_env import Go2Env
from go2tpms.numpy_policy import NumpyPolicy
from go2tpms.runs import load_config, mjx_policy_file, run_dir, sb3_policy_files


def load(name, foot, ckpt=None, render=False, heat_scale=1.0):
    """Returns (env, act) where act(obs) -> action, for SB3 or MJX-trained runs."""
    out = run_dir(name)
    config = load_config(out)
    # Evaluate with the action scale the policy was trained with (older runs: 0.25).
    env = Go2Env(foot_type=foot, action_scale=config.get("action_scale", 0.25),
                 heat_scale=heat_scale, render_mode="human" if render else None)
    if config.get("trainer") == "mjx":
        path = mjx_policy_file(out, ckpt)
        print(f"Loading {path}  (foot={foot})")
        return env, NumpyPolicy(path)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    policy_path, norm_path = sb3_policy_files(out, ckpt)
    print(f"Loading {policy_path}  (foot={foot})")
    norm = VecNormalize.load(norm_path, DummyVecEnv([lambda: env]))
    norm.training, norm.norm_reward = False, False
    policy = PPO.load(policy_path, device="cpu")

    def act(obs):
        action, _ = policy.predict(norm.normalize_obs(obs[None]), deterministic=True)
        return action[0]

    return env, act


def watch(env, act, speed):
    """Real-time viewer. Restarts when the robot falls; close the window to stop."""
    obs, _ = env.reset(seed=0, options={"cmd": speed})
    env.render()
    t, speeds, powers, elec = 0.0, [], [], []
    while env.viewer.is_running():
        start = time.time()
        obs, _, terminated, truncated, info = env.step(act(obs))
        speeds.append(info["speed"])
        powers.append(info["mech_power"])
        elec.append(info["elec_power"])
        t += env.dt
        if t >= 2.0:  # 2 s averages (one control step alone varies a lot within a stride)
            print(f"cmd {speed:.2f} m/s  actual {np.mean(speeds):.2f} m/s  "
                  f"mech power {np.mean(powers):5.1f} W  electrical (est.) {np.mean(elec):5.1f} W")
            t, speeds, powers, elec = 0.0, [], [], []
        if terminated or truncated:
            print("  robot fell, restarting" if terminated else "  episode end, restarting")
            obs, _ = env.reset(options={"cmd": speed})
        time.sleep(max(0.0, env.dt - (time.time() - start)))
    env.close()


METRICS = ("speed", "mech_power", "elec_power", "brake_power", "torque_sq")


def run(env, act, speed, seconds, settle=2.0):
    """Walk at one commanded speed; averages after the settle time, or (None, fall time).

    Returns a dict: speed, track_err (mean |speed - cmd|), mech_power (positive motor work),
    elec_power (estimated, incl. motor heat), brake_power, torque_sq, mech_std, compression
    (max, mm), eng_<type> (mean unweighted energy term of each type).
    """
    obs, _ = env.reset(seed=0, options={"cmd": speed})
    log = {k: [] for k in METRICS + ("compression", "eng_phase", "eng_power", "eng_torque")}
    for i in range(int((settle + seconds) / env.dt)):
        obs, _, terminated, _, info = env.step(act(obs))
        if i * env.dt >= settle:
            for k in METRICS:
                log[k].append(info[k])
            log["compression"].append(info["foot_compression_mm"].max())
            for k, v in info["eng_raw"].items():
                log[f"eng_{k}"].append(v)
        if terminated:
            return None, i * env.dt  # fell
    result = {k: float(np.mean(v)) for k, v in log.items()}
    result["compression"] = float(np.max(log["compression"]))
    result["mech_std"] = float(np.std(log["mech_power"]))
    result["track_err"] = float(np.mean(np.abs(np.array(log["speed"]) - speed)))
    return result, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--foot", choices=list(FOOT_TYPES),
                        help="default: the foot the run was trained with")
    parser.add_argument("--ckpt", help="checkpoint step count, or 'latest' (default: final policy)")
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--heat-scale", type=float, default=1.0,
                        help="scale of the motor heat estimate in electrical power (0 = none)")
    args = parser.parse_args()

    if args.foot is None:
        args.foot = load_config(run_dir(args.name)).get("foot", "tpms60")

    if not args.evaluate:
        watch(*load(args.name, args.foot, args.ckpt, render=True, heat_scale=args.heat_scale), args.speed)
        return

    env, act = load(args.name, args.foot, args.ckpt, heat_scale=args.heat_scale)
    weight = env.model.body_subtreemass[1] * 9.81
    print(f"Policy '{args.name}', foot={args.foot}  (30 s per speed, like the paper)")
    print("Power [W]: mech = positive motor work, elec = estimated electrical (motor work + heat,"
          " no regeneration), brake = negative motor work")
    print("cmd [m/s]  actual [m/s]  mech power    elec   brake   CoT mech  CoT elec  max compr. [mm]")
    for speed in (0.2, 0.4, 0.6, 0.8, 1.0):
        r, fell_at = run(env, act, speed, seconds=30.0)
        if r is None:
            print(f"   {speed:.1f}       robot fell after {fell_at:.1f} s")
            continue
        v = max(r["speed"], 1e-3)
        print(f"   {speed:.1f}       {r['speed']:5.2f}     {r['mech_power']:6.1f} ± {r['mech_std']:4.1f}"
              f"  {r['elec_power']:6.1f}  {r['brake_power']:5.1f}    {r['mech_power'] / (weight * v):5.3f}"
              f"     {r['elec_power'] / (weight * v):5.3f}     {r['compression']:5.2f}")


if __name__ == "__main__":
    main()
