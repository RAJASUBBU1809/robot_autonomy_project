"""Gymnasium environment: Go2 walking forward on compliant feet.

Policy runs at 50 Hz and outputs joint-angle offsets from the standing pose
(action_scale 0.5 rad; with 0.25 the thigh offsets saturate and the robot cannot
go faster than ~0.8 m/s); a PD controller turns them into motor torques at every 1 ms physics step. The PD
runs inside MuJoCo: the Go2 torque motors are turned into PD servos with the
torque limit as force limit, which is the same law but much faster than Python.

Reward = command tracking + regularisation + an energy term, chosen by eng_type:
  "phase"  energy-phase term, paper Eq. 15: -sum over legs of (change in leg power minus change
           in foot elastic power)^2. With the rigid foot (no spring) it penalises changes in leg power.
  "power"  mechanical power, -sum over joints of |torque * joint speed| (Fu et al. 2021).
  "torque" -sum of torque^2 (on top of the small "torques" term every run has).
Set eng_weight=0 to train without it. The weights of the three types are not comparable
(different units); scripts/calibrate_eng.py matches them. Default phase 3e-3: with 1e-4 the
term was only ~0.2 % of the reward and did not change the gait.
"""
import gymnasium as gym
import mujoco
import numpy as np

from .compliant_foot import LEGS, CompliantFeet
from .paths import SCENE_XML

MODEL_PATH = str(SCENE_XML)
ENG_TYPES = ("phase", "power", "torque")

# Motor heat (copper loss) per joint torque^2, for the electrical power estimate.
# Go2 joint motor teardown (simplexitypd.com/blog/unitree-go2-motor-teardown): line resistance
# 0.44 ohm (delta; equivalent wye 0.22 ohm), q-axis torque constant 0.26 Nm/A at the rotor,
# planetary gear 6.22:1. Copper loss = 1.5 * R_wye * i_q^2 and joint torque = Kt * gear * i_q.
# The calf has an extra linkage: its torque limit is 45.43 / 23.7 = 1.92x the hip/thigh limit.
# Uncertain (Unitree's own Go1 motor data gives a different Kt), so results are also reported
# with heat_scale 0 and 2.
MOTOR_R_WYE, MOTOR_KT, MOTOR_GEAR = 0.22, 0.26, 6.22


def heat_coef(torque_limit):
    """W per (Nm)^2 of joint torque, per actuator (~0.126 hip/thigh, ~0.034 calf)."""
    gear = MOTOR_GEAR * torque_limit / torque_limit.min()
    return 1.5 * MOTOR_R_WYE / (MOTOR_KT * gear) ** 2


def make_pd_servos(m, kp, kd):
    """Turn the Go2 torque motors into PD servos, in place; returns the torque limits.

    force = kp*(ctrl - q) - kd*qdot, clipped to the motor's torque limit, so ctrl
    becomes the target joint angle. Shared by the CPU env and the MJX env.
    """
    torque_limit = m.actuator_ctrlrange[:, 1].copy()
    m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_gainprm[:, :] = 0.0
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_biasprm[:, :] = 0.0
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kd
    m.actuator_forcerange[:, 0] = -torque_limit
    m.actuator_forcerange[:, 1] = torque_limit
    m.actuator_forcelimited[:] = 1
    m.actuator_ctrllimited[:] = 0
    return torque_limit


class Go2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, foot_type="tpms60", eng_weight=3e-3, eng_type="phase", cmd_range=(0.2, 1.0),
                 episode_seconds=20.0, kp=40.0, kd=1.0, action_scale=0.5,
                 control_dt=0.02, heat_scale=1.0, render_mode=None):
        assert eng_type in ENG_TYPES, eng_type
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.feet = CompliantFeet(self.model, foot_type)
        self.eng_weight, self.eng_type = eng_weight, eng_type
        self.cmd_range = cmd_range
        self.kp, self.kd, self.action_scale = kp, kd, action_scale
        self.frame_skip = int(round(control_dt / self.model.opt.timestep))
        self.dt = self.frame_skip * self.model.opt.timestep
        self.max_steps = int(episode_seconds / self.dt)
        self.render_mode = render_mode
        self.viewer = None

        m = self.model
        act_joints = m.actuator_trnid[:, 0]
        self.qadr = m.jnt_qposadr[act_joints]
        self.vadr = m.jnt_dofadr[act_joints]
        # Actuators are ordered FL(hip, thigh, calf), FR, RL, RR: 3 per leg.
        assert [m.actuator(i).name[:2] for i in range(m.nu)] == [l for l in LEGS for _ in range(3)]
        self.floor_id = m.geom("floor").id
        self.foot_geom_ids = np.array([m.geom(leg).id for leg in LEGS])

        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        self.init_qpos = self.data.qpos.copy()
        self.default_q = self.data.qpos[self.qadr].copy()
        self.torque_limit = make_pd_servos(m, kp, kd)
        self.heat_coef = heat_scale * heat_coef(self.torque_limit)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(m.nu,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 1 + 3 * m.nu
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _base_frame(self):
        """Base linear/angular velocity in the body frame, and gravity direction."""
        d = self.data
        R = d.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, R)
        rot = rot.reshape(3, 3)
        lin_vel = rot.T @ d.qvel[0:3]
        ang_vel = d.qvel[3:6]  # free-joint angular velocity is already in the body frame
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        return lin_vel, ang_vel, gravity

    def _foot_contacts(self):
        contact = np.zeros(4, dtype=bool)
        for c in self.data.contact[: self.data.ncon]:
            for i, g in enumerate(self.foot_geom_ids):
                if (c.geom1 == g and c.geom2 == self.floor_id) or (c.geom2 == g and c.geom1 == self.floor_id):
                    contact[i] = True
        return contact

    def _get_obs(self):
        lin_vel, ang_vel, gravity = self._base_frame()
        q = self.data.qpos[self.qadr] - self.default_q
        qd = self.data.qvel[self.vadr]
        return np.concatenate([
            lin_vel * 2.0, ang_vel * 0.25, gravity, [self.cmd * 2.0],
            q, qd * 0.05, self.last_action,
        ]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = self.init_qpos
        d.qpos[self.qadr] += self.np_random.uniform(-0.05, 0.05, self.model.nu)
        mujoco.mj_forward(self.model, d)

        if options and "cmd" in options:
            self.cmd = float(options["cmd"])
        else:
            self.cmd = float(self.np_random.uniform(*self.cmd_range))
        self.steps = 0
        self.last_action = np.zeros(self.model.nu)
        self.air_time = np.zeros(4)
        self.prev_contact = np.ones(4, dtype=bool)  # starts standing: no fake touchdowns
        self.prev_leg_power = np.zeros(4)
        self.prev_elastic_power = np.zeros(4)
        return self._get_obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(action, -1.0, 1.0)
        q_target = self.default_q + self.action_scale * action

        d.ctrl[:] = q_target  # PD servo targets (see __init__)

        p_sum = np.zeros(m.nu)       # joint power, summed over substeps
        p_pos_sum = np.zeros(m.nu)   # positive motor work only
        p_abs_sum = np.zeros(m.nu)
        elec_sum = np.zeros(m.nu)    # estimated electrical power, no regeneration
        torque_sq = np.zeros(m.nu)
        elastic_power = np.zeros(4)
        vadr, feet, heat = self.vadr, self.feet, self.heat_coef
        for _ in range(self.frame_skip):
            feet.apply(d)
            mujoco.mj_step(m, d)
            tau = d.actuator_force
            p = tau * d.qvel[vadr]
            tsq = tau * tau
            p_sum += p
            p_pos_sum += np.maximum(p, 0.0)
            p_abs_sum += np.abs(p)
            elec_sum += np.maximum(p + heat * tsq, 0.0)
            torque_sq += tsq
            elastic_power += feet.power
        n = self.frame_skip
        leg_power = p_sum.reshape(4, 3).sum(axis=1) / n  # mean power per leg, W
        elastic_power /= n                                # mean elastic power per foot, W
        torque_sq = torque_sq.sum() / n
        mech_power = p_pos_sum.sum() / n
        abs_power = p_abs_sum.sum() / n

        lin_vel, ang_vel, gravity = self._base_frame()
        contact = self._foot_contacts()

        # Feet air time: reward steps that last longer than 0.25 s (encourages a clean gait).
        self.air_time += self.dt
        first_contact = contact & ~self.prev_contact
        air_reward = np.sum((self.air_time - 0.25) * first_contact) if self.cmd > 0.1 else 0.0
        self.air_time[contact] = 0.0
        self.prev_contact = contact

        # Energy term (see module docstring). Energy-phase: paper Eq. 15, summed over the legs.
        mismatch = (leg_power - self.prev_leg_power) - (elastic_power - self.prev_elastic_power)
        eng_raw = {"phase": -np.sum(mismatch ** 2), "power": -abs_power, "torque": -torque_sq}
        self.prev_leg_power, self.prev_elastic_power = leg_power, elastic_power

        r = {
            "track_vx": 1.0 * np.exp(-((self.cmd - lin_vel[0]) ** 2 + lin_vel[1] ** 2) / 0.25),
            "track_yaw": 0.5 * np.exp(-(ang_vel[2] ** 2) / 0.25),
            "lin_vel_z": -2.0 * lin_vel[2] ** 2,
            "ang_vel_xy": -0.05 * np.sum(ang_vel[:2] ** 2),
            "orientation": -1.0 * np.sum(gravity[:2] ** 2),
            "torques": -1e-4 * torque_sq,
            "action_rate": -0.01 * np.sum((action - self.last_action) ** 2),
            "air_time": 1.0 * air_reward,
            "energy": self.eng_weight * eng_raw[self.eng_type],
        }
        reward = sum(r.values()) * self.dt * 50.0  # same scale whatever control_dt is
        self.last_action = action
        self.steps += 1

        fallen = gravity[2] > -0.5 or d.qpos[2] < 0.12
        terminated = bool(fallen)
        truncated = self.steps >= self.max_steps
        if terminated:
            reward -= 2.0

        info = {
            "mech_power": mech_power,           # W, positive motor work only
            "brake_power": (abs_power - p_sum.sum() / n) / 2,  # W, negative motor work (braking)
            "elec_power": elec_sum.sum() / n,   # W, estimated: motor work + copper heat, no regen
            "torque_sq": torque_sq,
            "eng_raw": eng_raw,                 # unweighted energy terms of all three types
            "speed": lin_vel[0],
            "cmd": self.cmd,
            "foot_compression_mm": self.feet.delta * 1000.0,
            "reward_terms": r,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Camera follows the robot (Esc in the viewer switches to a free camera).
            with self.viewer.lock():
                cam = self.viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = self.model.body("base").id
                cam.distance, cam.azimuth, cam.elevation = 1.6, 135.0, -20.0
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
