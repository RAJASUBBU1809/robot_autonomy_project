"""MJX (GPU) version of go2_env.Go2Env, for training with Brax PPO.

Same model, PD servos, compliant feet, observations, rewards and termination as
the CPU env, written in JAX so thousands of robots run in parallel on a GPU.
Only differences: leg/body collision geoms are turned off (only the 4 feet touch
the floor; a body touching the floor means a fall, which terminates anyway) and
the solver runs fewer iterations (10 instead of 100; Newton usually converges in 2-3).

    from go2tpms.go2_mjx_env import Go2MjxEnv
    env = Go2MjxEnv(config_overrides={"foot_type": "solid"})
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .compliant_foot import FOOT_TYPES, LEGS, flat_below, lock_feet
from .go2_env import ENG_TYPES, make_pd_servos
from .paths import SCENE_XML

MODEL_PATH = str(SCENE_XML)

REWARD_TERMS = ("track_vx", "track_yaw", "lin_vel_z", "ang_vel_xy", "orientation",
                "torques", "action_rate", "air_time", "energy")


def default_config():
    return config_dict.create(
        ctrl_dt=0.02,          # 50 Hz policy, as in go2_env.py
        sim_dt=0.001,          # 1 ms physics step, as in build_model.py
        episode_length=1000,   # 20 s
        action_repeat=1,
        foot_type="tpms60",
        eng_weight=3e-3,  # 1e-4 was ~0.2 % of the reward: too weak to shape the gait
        eng_type="phase",  # "phase" (paper Eq. 15), "power" or "torque": see go2_env.py
        cmd_min=0.2,
        cmd_max=1.0,
        kp=40.0,
        kd=1.0,
        action_scale=0.5,      # rad; 0.25 caps the speed at ~0.8 m/s (see go2_env.py)
        foot_armature=-1.0,    # < 0: keep the model's value (0.05)
        # Playground uses 4/8 for soft-contact robots; our stiff foot contact + foot
        # spring diverges with 4 (robot launched into the air), 10 is stable.
        solver_iterations=10,
        ls_iterations=10,
    )


def mjx_model_tweaks(m, foot_geom_ids, floor_id, iterations, ls_iterations):
    """MJX-only speedups, applied in place to an MjModel before mjx.put_model."""
    keep = set(int(g) for g in foot_geom_ids) | {int(floor_id)}
    for g in range(m.ngeom):
        if g not in keep:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
    m.opt.iterations = iterations
    m.opt.ls_iterations = ls_iterations


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Go2MjxEnv(mjx_env.MjxEnv):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        cfg = self._config
        assert cfg.eng_type in ENG_TYPES, cfg.eng_type

        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        m.opt.timestep = cfg.sim_dt
        self._torque_limit = make_pd_servos(m, cfg.kp, cfg.kd)
        self._floor_id = m.geom("floor").id
        self._foot_geom_ids = np.array([m.geom(leg).id for leg in LEGS])
        mjx_model_tweaks(m, self._foot_geom_ids, self._floor_id,
                         cfg.solver_iterations, cfg.ls_iterations)
        slide = [m.joint(f"{leg}_foot_slide") for leg in LEGS]
        if cfg.foot_armature >= 0:
            for j in slide:
                m.dof_armature[j.dofadr[0]] = cfg.foot_armature
        if FOOT_TYPES[cfg.foot_type].get("rigid"):
            lock_feet(m)
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl="jax")
        self._n_substeps = int(round(cfg.ctrl_dt / cfg.sim_dt))

        act_joints = m.actuator_trnid[:, 0]
        self._qadr = jp.array(m.jnt_qposadr[act_joints])
        self._vadr = jp.array(m.jnt_dofadr[act_joints])
        self._slide_q = jp.array([j.qposadr[0] for j in slide])
        self._slide_v = jp.array([j.dofadr[0] for j in slide])
        assert [m.actuator(i).name[:2] for i in range(m.nu)] == [l for l in LEGS for _ in range(3)]

        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self._init_qpos = jp.array(d.qpos)
        self._default_q = jp.array(d.qpos[m.jnt_qposadr[act_joints]])
        # Contact margin of foot-floor pairs: the CPU env counts a foot as touching
        # when MuJoCo lists the contact, i.e. when dist < margin.
        self._contact_margin = float(max(m.geom_margin[self._foot_geom_ids[0]],
                                         m.geom_margin[self._floor_id]))
        self._foot_radius = float(m.geom_size[self._foot_geom_ids[0], 0])
        self._foot_geom_ids = jp.array(self._foot_geom_ids)

        # Foot spring constants (same law as compliant_foot.CompliantFeet).
        spec = FOOT_TYPES[cfg.foot_type]
        self._a0, self._a1, self._a2 = spec["coef"]
        loss, v_switch = spec["loss"], 0.005
        self._d_flat = flat_below(self._a1, self._a2)
        self._c_mean, self._c_amp, self._inv_vs = 1.0 - 0.5 * loss, 0.5 * loss, 1.0 / v_switch

    # ------------------------------------------------------------- properties
    @property
    def xml_path(self):
        return MODEL_PATH

    @property
    def action_size(self):
        return self._mj_model.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    # ---------------------------------------------------------------- helpers
    def foot_force(self, q_slide, v_slide):
        """Spring force of each foot (N), same as CompliantFeet.force_at."""
        d = jp.maximum(q_slide * 1000.0, self._d_flat)
        f = self._a0 + d * (self._a1 + self._a2 * d)
        return f * (self._c_mean + self._c_amp * jp.tanh(v_slide * self._inv_vs))

    def _foot_contacts(self, data):
        # Floor = plane z=0, so the sphere-floor contact distance is z - radius; MuJoCo
        # lists the contact (what the CPU env checks) when that is below the margin.
        dist = data.geom_xpos[self._foot_geom_ids, 2] - self._foot_radius
        return dist < self._contact_margin

    def _base_frame(self, data):
        rot = quat_to_mat(data.qpos[3:7])
        lin_vel = rot.T @ data.qvel[0:3]
        ang_vel = data.qvel[3:6]  # free-joint angular velocity is in the body frame
        gravity = rot.T @ jp.array([0.0, 0.0, -1.0])
        return lin_vel, ang_vel, gravity

    def _get_obs(self, data, info):
        lin_vel, ang_vel, gravity = self._base_frame(data)
        q = data.qpos[self._qadr] - self._default_q
        qd = data.qvel[self._vadr]
        obs = jp.concatenate([
            lin_vel * 2.0, ang_vel * 0.25, gravity, jp.array([info["cmd"] * 2.0]),
            q, qd * 0.05, info["last_action"],
        ])
        return jp.nan_to_num(obs)

    # --------------------------------------------------------------- env API
    def reset(self, rng):
        rng, k_noise, k_cmd = jax.random.split(rng, 3)
        nu = self.action_size
        qpos = self._init_qpos.at[self._qadr].add(
            jax.random.uniform(k_noise, (nu,), minval=-0.05, maxval=0.05))
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=jp.zeros(self._mj_model.nv),
                                 ctrl=self._default_q, impl="jax")
        data = mjx.forward(self._mjx_model, data)
        cmd = jax.random.uniform(k_cmd, (), minval=self._config.cmd_min,
                                 maxval=self._config.cmd_max)
        info = {
            "rng": rng,
            "cmd": cmd,
            "last_action": jp.zeros(nu),
            "air_time": jp.zeros(4),
            "prev_contact": jp.ones(4, dtype=bool),  # starts standing
            "prev_leg_power": jp.zeros(4),
            "prev_elastic_power": jp.zeros(4),
        }
        metrics = {f"rew/{k}": jp.zeros(()) for k in REWARD_TERMS}
        metrics.update({k: jp.zeros(()) for k in (
            "env/mech_power_W", "env/speed", "env/speed_error", "env/max_foot_compression_mm")})
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        cfg, info = self._config, state.info
        nu = self.action_size
        action = jp.clip(action, -1.0, 1.0)
        data = state.data.replace(ctrl=self._default_q + cfg.action_scale * action)
        nv = self._mj_model.nv

        def substep(carry, _):
            data, p_sum, p_pos, p_abs, tsq, e_pow = carry
            v_slide = data.qvel[self._slide_v]
            f = self.foot_force(data.qpos[self._slide_q], v_slide)
            data = data.replace(qfrc_applied=jp.zeros(nv).at[self._slide_v].set(-f))
            data = mjx.step(self._mjx_model, data)
            tau = data.actuator_force
            p = tau * data.qvel[self._vadr]
            return (data, p_sum + p, p_pos + jp.maximum(p, 0.0), p_abs + jp.abs(p), tsq + tau * tau,
                    e_pow + f * v_slide), None

        zeros = jp.zeros(nu)
        (data, p_sum, p_pos, p_abs, tsq, e_pow), _ = jax.lax.scan(
            substep, (data, zeros, zeros, zeros, zeros, jp.zeros(4)), None, length=self._n_substeps)
        n = self._n_substeps
        leg_power = p_sum.reshape(4, 3).sum(axis=1) / n
        elastic_power = e_pow / n
        torque_sq = tsq.sum() / n
        mech_power = p_pos.sum() / n
        abs_power = p_abs.sum() / n

        lin_vel, ang_vel, gravity = self._base_frame(data)
        contact = self._foot_contacts(data)
        dt = self.dt

        # Feet air time (as go2_env.py).
        air_time = info["air_time"] + dt
        first_contact = contact & ~info["prev_contact"]
        air_reward = jp.sum((air_time - 0.25) * first_contact) * (info["cmd"] > 0.1)
        air_time = jp.where(contact, 0.0, air_time)

        # Energy term (see go2_env.py). Energy-phase: paper Eq. 15, summed over the four legs.
        mismatch = (leg_power - info["prev_leg_power"]) - (elastic_power - info["prev_elastic_power"])
        r_eng = {"phase": -jp.sum(mismatch ** 2), "power": -abs_power,
                 "torque": -torque_sq}[cfg.eng_type]

        cmd = info["cmd"]
        r = {
            "track_vx": 1.0 * jp.exp(-((cmd - lin_vel[0]) ** 2 + lin_vel[1] ** 2) / 0.25),
            "track_yaw": 0.5 * jp.exp(-(ang_vel[2] ** 2) / 0.25),
            "lin_vel_z": -2.0 * lin_vel[2] ** 2,
            "ang_vel_xy": -0.05 * jp.sum(ang_vel[:2] ** 2),
            "orientation": -1.0 * jp.sum(gravity[:2] ** 2),
            "torques": -1e-4 * torque_sq,
            "action_rate": -0.01 * jp.sum((action - info["last_action"]) ** 2),
            "air_time": 1.0 * air_reward,
            "energy": cfg.eng_weight * r_eng,
        }
        reward = sum(r.values()) * dt * 50.0

        fallen = (gravity[2] > -0.5) | (data.qpos[2] < 0.12)
        fallen = fallen | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        reward = jp.where(fallen, reward - 2.0, reward)
        reward = jp.nan_to_num(reward)

        info = {**info, "last_action": action, "air_time": air_time, "prev_contact": contact,
                "prev_leg_power": leg_power, "prev_elastic_power": elastic_power}
        metrics = dict(state.metrics)  # keep keys added by Brax wrappers
        metrics.update({f"rew/{k}": v for k, v in r.items()})
        metrics.update({
            "env/mech_power_W": mech_power,
            "env/speed": lin_vel[0],
            "env/speed_error": jp.abs(lin_vel[0] - cmd),
            "env/max_foot_compression_mm": jp.max(jp.maximum(data.qpos[self._slide_q], 0.0)) * 1000.0,
        })
        obs = self._get_obs(data, info)
        return state.replace(data=data, obs=obs, reward=reward,
                             done=fallen.astype(jp.float32), metrics=metrics, info=info)
