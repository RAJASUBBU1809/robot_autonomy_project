"""Compliant foot model based on Lee et al. (2026), IJPEM-GT.

Loading curve fitted in the paper:  F[N] = a0 + a1*d + a2*d^2,  d in mm.
Hysteresis: while the foot is extending (unloading), the force is reduced by
the measured energy-loss ratio, so one load/unload cycle loses that fraction.

Call feet.apply(data) before every mujoco.mj_step(model, data).
Check the curve against the paper's numbers:  python -m go2tpms.compliant_foot
"""
import math

import numpy as np

FOOT_TYPES = {
    # Section 2.3.4 + Table 4: TPMS-D, 60% relative density (the paper's optimum)
    "tpms60": {"coef": (4.6105, -3.8891, 7.9422), "loss": 0.123},
    # Table 2 + Table 4: solid hemisphere of the same resin (the paper's baseline)
    "solid": {"coef": (9.639, -2.472, 15.505), "loss": 0.304},
    # No spring: the slide joint is locked (lock_feet), like a normal rigid robot foot.
    "rigid": {"coef": (0.0, 0.0, 0.0), "loss": 0.0, "rigid": True},
}
LEGS = ("FL", "FR", "RL", "RR")
# Ground contact of the foot sphere (used by scripts/build_model.py): nearly rigid, so all foot
# compliance comes from the emulated spring.
FOOT_SOLREF = (0.004, 1.0)
FOOT_SOLIMP = (0.9, 0.95, 0.001)
# Lock of the rigid foot's slide joint: same time constant, higher impedance, so it holds the
# standing load with < 0.1 mm instead of ~0.55 mm with the contact values.
LOCK_SOLREF = (0.004, 1.0)
LOCK_SOLIMP = (0.99, 0.999, 0.001)
_trapz = getattr(np, "trapezoid", None) or np.trapz  # works on old and new NumPy


def flat_below(a1, a2):
    """Compression (mm) of the parabola's minimum; the fitted force is held flat below it."""
    return max(0.0, -a1 / (2 * a2)) if a2 else 0.0


def lock_feet(model):
    """Rigid foot: lock the four foot slide joints at zero compression (in place, on an MjModel).
    A joint limit with range (0, 0) and a stiff solref/solimp; works in MJX too."""
    for leg in LEGS:
        j = model.joint(f"{leg}_foot_slide").id
        model.jnt_limited[j] = 1
        model.jnt_range[j] = (0.0, 0.0)
        model.jnt_solref[j] = LOCK_SOLREF
        model.jnt_solimp[j, :3] = LOCK_SOLIMP


class CompliantFeet:
    def __init__(self, model, foot_type="tpms60", v_switch=0.005):
        spec = FOOT_TYPES[foot_type]
        self.a0, self.a1, self.a2 = spec["coef"]
        self.loss = spec["loss"]
        self.v_switch = v_switch  # m/s, smooths the load/unload switch
        # Below the parabola's minimum the fit is not physical, so hold it flat there.
        self.d_flat = flat_below(self.a1, self.a2)  # mm
        if spec.get("rigid"):
            lock_feet(model)

        joints = [model.joint(f"{leg}_foot_slide") for leg in LEGS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])

        self._adr = list(zip(self.qadr.tolist(), self.dadr.tolist()))
        self._data = None         # last MjData seen by apply() (for .delta)
        self.force = np.zeros(4)  # spring force, N
        self.power = np.zeros(4)  # elastic power, W (+ storing, - releasing)

    def loading_force(self, delta_m):
        d = np.maximum(np.asarray(delta_m) * 1000.0, self.d_flat)
        return self.a0 + self.a1 * d + self.a2 * d * d

    def force_at(self, delta_m, vel):
        unloading = 0.5 * (1.0 - np.tanh(vel / self.v_switch))  # 0 loading, 1 unloading
        return self.loading_force(delta_m) * (1.0 - self.loss * unloading)

    def apply(self, data):
        """Same law as force_at(). Written with plain floats because it runs every
        1 ms step, and for 4 values that is ~6x faster than NumPy."""
        qpos, qvel, qfrc = data.qpos, data.qvel, data.qfrc_applied
        a0, a1, a2, d_flat = self.a0, self.a1, self.a2, self.d_flat
        c_mean, c_amp, inv_vs = 1.0 - 0.5 * self.loss, 0.5 * self.loss, 1.0 / self.v_switch
        force, power = self.force, self.power
        self._data = data
        for i, (qa, da) in enumerate(self._adr):
            d = qpos[qa] * 1000.0
            if d < d_flat:
                d = d_flat
            vel = qvel[da]
            f = (a0 + d * (a1 + a2 * d)) * (c_mean + c_amp * math.tanh(vel * inv_vs))
            qfrc[da] = -f  # pushes the foot back out
            force[i], power[i] = f, f * vel
        return force

    @property
    def delta(self):
        """Compression of each foot, m (from the last apply() call's data)."""
        return np.maximum(self._data.qpos[self.qadr], 0.0) if self._data is not None else np.zeros(4)


def check_curve(foot_type, f_max=300.0):
    """Quasi-static 0 -> f_max -> 0 cycle, like the paper's compression test."""

    class _Joint:  # minimal stand-in so we can test without a model
        qposadr = dofadr = [0]

    class _Model:
        def joint(self, name):
            return _Joint()

    feet = CompliantFeet(_Model(), foot_type)
    d = np.linspace(0.0, 0.015, 30001)
    f_load = feet.force_at(d, +1.0)
    d = d[f_load <= f_max]
    f_load, f_unload = feet.force_at(d, +1.0), feet.force_at(d, -1.0)
    e_load = _trapz(f_load, d) * 1000.0      # mJ
    e_a = _trapz(f_unload, d) * 1000.0       # absorbed (returned), mJ
    e_d = e_load - e_a                             # dissipated, mJ
    print(f"{foot_type:7s} d_max={d[-1] * 1000:5.2f} mm  Ea={e_a:6.1f} mJ  "
          f"Ed={e_d:5.1f} mJ  loss={100 * e_d / e_load:4.1f} %")


if __name__ == "__main__":
    print("Simulated compression test (paper Table 4 in brackets):")
    check_curve("tpms60")
    print("        [paper: d_max=6.26 mm  Ea=538.7 mJ  Ed=75.6 mJ  loss=12.3 %]")
    check_curve("solid")
    print("        [paper: d_max=4.18 mm  Ea=298.5 mJ  Ed=130.1 mJ  loss=30.4 %]")
