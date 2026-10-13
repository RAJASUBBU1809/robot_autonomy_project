"""Benchmark the MJX env: how many env steps per second does this GPU give?

    python scripts/bench_mjx.py                 # 4096 envs, 1 ms and 2 ms physics step
    python scripts/bench_mjx.py --envs 8192

Prints env-steps/s and the time a 100M-step training run would need (training
adds ~20-30% on top for the PPO updates). The laptop CPU does ~900 env-steps/s.
The 2 ms option raises the foot slide armature so the stiff foot spring stays
stable (sqrt(k/m)*dt below the explicit limit); validate with check_mjx.py first.
"""
import argparse
import time

import jax
import jax.numpy as jp

from go2tpms.go2_mjx_env import Go2MjxEnv

CONFIGS = {
    "1 ms (default)": dict(sim_dt=0.001),
    "2 ms, foot armature 0.29": dict(sim_dt=0.002, foot_armature=0.29),
}


def bench(name, overrides, n_envs, n_steps):
    env = Go2MjxEnv(config_overrides=overrides)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(0)
    state = reset(jax.random.split(key, n_envs))

    t0 = time.time()
    actions = jax.random.uniform(key, (n_envs, env.action_size), minval=-1, maxval=1)
    state = step(state, actions)
    jax.block_until_ready(state.obs)
    compile_s = time.time() - t0

    t0 = time.time()
    fallen = jp.zeros(())
    for i in range(n_steps):
        key, sub = jax.random.split(key)
        actions = jax.random.uniform(sub, (n_envs, env.action_size), minval=-1, maxval=1)
        state = step(state, actions)
        fallen = fallen + state.done.sum()
    jax.block_until_ready(state.obs)
    sec = time.time() - t0
    sps = n_envs * n_steps / sec
    nan = int(jp.isnan(state.data.qpos).any(axis=1).sum())
    print(f"{name:26s} {sps:9,.0f} env-steps/s  ({sps * env.n_substeps:11,.0f} physics steps/s)  "
          f"100M steps ~ {1e8 / sps / 3600:5.1f} h   compile {compile_s:4.0f} s   "
          f"envs with NaN: {nan}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50, help="control steps to time")
    args = parser.parse_args()
    print("JAX devices:", jax.devices())
    print(f"{args.envs} envs, random actions, {args.steps} control steps (50 Hz)")
    for name, overrides in CONFIGS.items():
        bench(name, overrides, args.envs, args.steps)


if __name__ == "__main__":
    main()
