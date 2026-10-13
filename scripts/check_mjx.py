"""Check that the MJX env (go2_mjx_env.py) matches the CPU env (go2_env.py).

    venv_mjx\\Scripts\\python.exe scripts\\check_mjx.py            # all checks
    venv_mjx\\Scripts\\python.exe scripts\\check_mjx.py --foot solid
    venv_mjx\\Scripts\\python.exe scripts\\check_mjx.py --foot solid --sim-dt 0.002 --foot-armature 0.29
    venv_mjx\\Scripts\\python.exe scripts\\check_mjx.py --foot rigid --eng-type power --eng-weight 1e-3

1. foot force: JAX formula vs compliant_foot.CompliantFeet.force_at
2. stand: zero action for 3 s, foot compression (compare with stand_test.py)
3. parity: same start state + same random actions in both envs, 25 control steps
"""
import argparse
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from go2tpms.compliant_foot import FOOT_TYPES, CompliantFeet
from go2tpms.go2_env import ENG_TYPES, Go2Env
from go2tpms.go2_mjx_env import Go2MjxEnv


def check_force(env, foot):
    if FOOT_TYPES[foot].get("rigid"):
        print("1. Foot force: rigid foot, no spring (slide joint locked) -> skipped")
        return

    class _J:
        qposadr = dofadr = [0]

    class _M:
        def joint(self, name):
            return _J()

    ref = CompliantFeet(_M(), foot)
    d, v = np.meshgrid(np.linspace(-0.001, 0.009, 101), np.linspace(-0.05, 0.05, 41))
    f_ref = ref.force_at(np.maximum(d, 0.0), v)
    f_jax = np.asarray(env.foot_force(jp.array(d, dtype=jp.float32), jp.array(v, dtype=jp.float32)))
    err = np.max(np.abs(f_jax - f_ref))
    print(f"[1] foot force max |JAX - NumPy| = {err:.2e} N  (forces up to {f_ref.max():.0f} N)  "
          f"{'OK' if err < 1e-3 * f_ref.max() else 'MISMATCH'}")


def check_stand(env, reset, step):
    state = reset(jax.random.PRNGKey(0))
    zero = jp.zeros(env.action_size)
    t0 = time.time()
    state = step(state, zero)
    jax.block_until_ready(state.obs)
    print(f"[2] (jit compile + first step: {time.time() - t0:.0f} s)")
    for i in range(1, int(3.0 / env.dt) + 1):
        state = step(state, zero)
        if i % 50 == 0:
            comp = np.maximum(np.asarray(state.data.qpos[env._slide_q]), 0) * 1000
            print(f"    t={i * env.dt:4.2f}s  z={float(state.data.qpos[2]):.3f} m  "
                  f"compression [mm] {np.round(comp, 2)}  done={float(state.done):.0f}")


def check_parity(env, reset, step, foot, eng_type, eng_weight, n_steps=25):
    cpu = Go2Env(foot_type=foot, eng_type=eng_type, eng_weight=eng_weight)
    obs_c, _ = cpu.reset(seed=3, options={"cmd": 0.6})
    state = reset(jax.random.PRNGKey(0))
    data = state.data.replace(qpos=jp.array(cpu.data.qpos, dtype=jp.float32),
                              qvel=jp.array(cpu.data.qvel, dtype=jp.float32))
    data = mjx.forward(env.mjx_model, data)
    info = {**state.info, "cmd": jp.array(0.6, dtype=jp.float32)}
    state = state.replace(data=data, info=info, obs=env._get_obs(data, info))
    print(f"[3] parity, step 0: max |obs diff| = {np.max(np.abs(np.asarray(state.obs) - obs_c)):.2e}")

    rng = np.random.default_rng(0)
    print("    step   max|obs diff|   reward cpu / mjx     mech power cpu / mjx [W]   compression cpu / mjx [mm]")
    for i in range(1, n_steps + 1):
        a = rng.uniform(-1, 1, env.action_size).astype(np.float32)
        obs_c, r_c, term_c, _, info_c = cpu.step(a)
        state = step(state, jp.array(a))
        o_m = np.asarray(state.obs)
        if abs(r_c - float(state.reward)) > 0.05:
            diffs = {k: (round(float(v), 3), round(float(state.metrics[f"rew/{k}"]), 3))
                     for k, v in info_c["reward_terms"].items()
                     if abs(v - float(state.metrics[f"rew/{k}"])) > 0.01}
            print(f"    step {i}: reward terms differing (cpu, mjx): {diffs}")
        if i % 5 == 0 or i == 1:
            comp_m = np.max(np.maximum(np.asarray(state.data.qpos[env._slide_q]), 0)) * 1000
            print(f"    {i:4d}   {np.max(np.abs(o_m - obs_c)):10.3e}   {r_c:7.3f} / {float(state.reward):7.3f}"
                  f"    {info_c['mech_power']:7.2f} / {float(state.metrics['env/mech_power_W']):7.2f}"
                  f"        {info_c['foot_compression_mm'].max():5.2f} / {comp_m:5.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foot", default="tpms60", choices=list(FOOT_TYPES))
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--foot-armature", type=float, default=-1.0)
    parser.add_argument("--eng-type", default="phase", choices=list(ENG_TYPES))
    parser.add_argument("--eng-weight", type=float, default=3e-3)
    args = parser.parse_args()
    print("JAX devices:", jax.devices())

    env = Go2MjxEnv(config_overrides={"foot_type": args.foot, "sim_dt": args.sim_dt,
                                      "foot_armature": args.foot_armature,
                                      "eng_type": args.eng_type, "eng_weight": args.eng_weight})
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    check_force(env, args.foot)
    check_stand(env, reset, step)
    if args.sim_dt == 0.001 and args.foot_armature < 0:
        check_parity(env, reset, step, args.foot, args.eng_type, args.eng_weight)
    else:
        print("[3] parity skipped (CPU env uses 1 ms and the default foot armature)")


if __name__ == "__main__":
    main()
