"""Build a Go2 model with compliant feet (paper-style emulation).

Copies Menagerie's unitree_go2 folder (meshes, scene.xml, license) to
models/go2_model/ and writes a modified go2.xml there, where each foot is a
30 mm sphere on a short slide joint. The spring force on the joint is applied
from Python by go2tpms/compliant_foot.py (and its JAX copy in go2_mjx_env.py).

    python scripts/build_model.py

Re-run after changing anything below. Needs the Menagerie clone in mujoco_menagerie/.
"""
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from go2tpms import compliant_foot
from go2tpms.compliant_foot import LEGS
from go2tpms.paths import GO2_XML, MENAGERIE_GO2, MODEL_DIR, SCENE_XML

SRC = str(MENAGERIE_GO2 / "go2.xml")
DST = str(GO2_XML)

FOOT_RADIUS = 0.030   # m, hemisphere radius used in the paper
FOOT_MASS = 0.03      # kg, approx. mass of a 60%-density TPMS hemisphere
TRAVEL = 0.008        # m, max compression (paper: 6.3 mm at 300 N)
TIMESTEP = 0.001      # s, small enough for the stiff foot spring
# Ground contact of the foot sphere: nearly rigid, so all foot compliance comes from
# the emulated TPMS/solid spring. (Menagerie's soft foot contact sinks ~15 mm.)
FOOT_SOLREF = " ".join(f"{v:g}" for v in compliant_foot.FOOT_SOLREF)   # "0.004 1"
FOOT_SOLIMP = " ".join(f"{v:g}" for v in compliant_foot.FOOT_SOLIMP)   # "0.9 0.95 0.001"


def foot_axes_at_home():
    """Slide axis per leg = world vertical, seen from the calf in the standing pose."""
    m = mujoco.MjModel.from_xml_path(SRC)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    axes, foot_z = {}, None
    for leg in LEGS:
        R = d.body(f"{leg}_calf").xmat.reshape(3, 3)
        axes[leg] = R.T @ np.array([0.0, 0.0, 1.0])
        foot_z = d.geom(leg).xpos[2]
    return axes, foot_z


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def main():
    if not MENAGERIE_GO2.exists():
        raise SystemExit(f"{MENAGERIE_GO2} not found. Clone it first:\n"
                         "  git clone https://github.com/google-deepmind/mujoco_menagerie.git")
    shutil.copytree(MENAGERIE_GO2, MODEL_DIR, dirs_exist_ok=True)  # meshes, scene.xml, ...

    axes, old_foot_z = foot_axes_at_home()
    tree = ET.parse(SRC)
    root = tree.getroot()

    # Small timestep for the stiff spring. Pyramidal friction cone (as in Menagerie's
    # go2_mjx.xml): ~2.4x faster than elliptic here, fine for RL.
    option = root.find("option")
    option.set("timestep", str(TIMESTEP))
    option.set("cone", "pyramidal")

    for leg in LEGS:
        calf = root.find(f".//body[@name='{leg}_calf']")
        # Remove the original visual foot mesh and collision sphere.
        for g in list(calf.findall("geom")):
            if g.get("mesh") == "foot" or g.get("name") == leg:
                calf.remove(g)

        foot = ET.SubElement(calf, "body", name=f"{leg}_foot", pos="-0.002 0 -0.213")
        ET.SubElement(foot, "inertial", pos="0 0 0", mass=str(FOOT_MASS),
                      diaginertia=fmt([0.4 * FOOT_MASS * FOOT_RADIUS**2] * 3))
        ET.SubElement(foot, "joint", name=f"{leg}_foot_slide", type="slide",
                      axis=fmt(axes[leg]), range=f"0 {TRAVEL}",
                      damping="1", armature="0.05", frictionloss="0")
        # Collision sphere (keeps the original name, so "FL" etc. still work).
        ET.SubElement(foot, "geom", name=leg, **{"class": "foot"},
                      pos="0 0 0", size=str(FOOT_RADIUS), condim="3",
                      solref=FOOT_SOLREF, solimp=FOOT_SOLIMP)
        # Visible sphere so you can see the new foot in the viewer.
        ET.SubElement(foot, "geom", **{"class": "visual"}, type="sphere",
                      size=str(FOOT_RADIUS), rgba="0.55 0.75 0.95 1")

    # Keyframe: add one 0 per foot slide joint, raise the base so feet touch the floor.
    key = root.find(".//key[@name='home']")
    q = [float(x) for x in key.get("qpos").split()]
    q[2] += FOOT_RADIUS - old_foot_z
    base, legs = q[:7], q[7:]
    new_q = base + sum((legs[3 * i:3 * i + 3] + [0.0] for i in range(4)), [])
    key.set("qpos", fmt(new_q))

    ET.indent(tree, space="  ")
    tree.write(DST)

    # Check that the new model compiles.
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    print(f"Wrote {DST}: {m.njnt} joints, {m.nu} actuators, "
          f"total mass {m.body_subtreemass[1]:.2f} kg, timestep {m.opt.timestep} s")


if __name__ == "__main__":
    main()
