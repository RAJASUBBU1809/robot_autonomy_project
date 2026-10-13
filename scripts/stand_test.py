"""Stand the Go2 on its compliant feet and print how much each foot compresses.

    python scripts/stand_test.py            # opens the viewer
    python scripts/stand_test.py --headless # just prints numbers
    python scripts/stand_test.py --foot solid
"""
import argparse
import time

import mujoco
import numpy as np

from go2tpms.compliant_foot import FOOT_TYPES, CompliantFeet
from go2tpms.paths import SCENE_XML

KP, KD = 80.0, 2.0  # PD gains for holding the standing pose

parser = argparse.ArgumentParser()
parser.add_argument("--foot", default="tpms60", choices=list(FOOT_TYPES))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=3.0)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
data = mujoco.MjData(model)
feet = CompliantFeet(model, args.foot)

# Actuated joints and their standing-pose targets.
act_joints = model.actuator_trnid[:, 0]
qadr = model.jnt_qposadr[act_joints]
vadr = model.jnt_dofadr[act_joints]

mujoco.mj_resetDataKeyframe(model, data, 0)
q_target = data.qpos[qadr].copy()
data.qpos[2] += 0.05  # start 5 cm up, so it drops onto its feet


def control_step():
    data.ctrl[:] = KP * (q_target - data.qpos[qadr]) - KD * data.qvel[vadr]
    feet.apply(data)
    mujoco.mj_step(model, data)


def report():
    comp = " ".join(f"{d * 1000:5.2f}" for d in feet.delta)
    force = " ".join(f"{f:6.1f}" for f in feet.force)
    print(f"t={data.time:5.2f}s  height={data.qpos[2]:.3f} m  "
          f"compression [mm] FL FR RL RR: {comp}  force [N]: {force}")


steps_per_report = int(0.25 / model.opt.timestep)

if args.headless:
    for i in range(int(args.seconds / model.opt.timestep)):
        control_step()
        if i % steps_per_report == 0:
            report()
else:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = 0
        while viewer.is_running():
            start = time.time()
            control_step()
            if i % steps_per_report == 0:
                report()
            i += 1
            if i % 10 == 0:  # redraw every 10 steps, roughly real time
                viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))
