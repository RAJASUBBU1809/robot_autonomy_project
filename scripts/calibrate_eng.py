"""Find energy-term weights that put the same pressure on the policy for each eng_type.

    python scripts/calibrate_eng.py                       # rigid_base, levels 3e-4 1e-3 3e-3
    python scripts/calibrate_eng.py --run rigid_base --levels 1e-4 3e-4 1e-3 3e-3 1e-2

The three energy terms (phase, power, torque; see go2_env.py) have different units, so equal
weights mean nothing. This runs the starting policy of the fine-tunes at 0.2 ... 1.0 m/s
(deterministic, CPU env), averages each unweighted term over the speeds (training samples the
command uniformly in 0.2 ... 1.0) and scales the power/torque weights so their mean penalty
equals the phase term's at each phase weight ("level").
"""
import argparse

import numpy as np

from go2tpms.go2_env import ENG_TYPES
from go2tpms.runs import load_config, run_dir
from play import load, run

SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rigid_base")
    parser.add_argument("--levels", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3],
                        help="energy-phase weights to match")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    foot = load_config(run_dir(args.run)).get("foot", "tpms60")
    env, act = load(args.run, foot)
    raw = {t: [] for t in ENG_TYPES}
    print(f"Unweighted energy terms per control step, policy {args.run} ({foot} foot)")
    print("cmd [m/s]" + "".join(f"{t:>12s}" for t in ENG_TYPES))
    for speed in SPEEDS:
        r, fell_at = run(env, act, speed, seconds=args.seconds)
        if r is None:
            raise SystemExit(f"{args.run} fell at {speed} m/s after {fell_at:.1f} s")
        for t in ENG_TYPES:
            raw[t].append(-r[f"eng_{t}"])
        print(f"   {speed:.1f}  " + "".join(f"{-r[f'eng_{t}']:12.4g}" for t in ENG_TYPES))
    mean = {t: float(np.mean(v)) for t, v in raw.items()}
    print("   mean " + "".join(f"{mean[t]:12.4g}" for t in ENG_TYPES))

    print("\nWeights with the same mean penalty (penalty per step in brackets; the walking"
          " reward is ~1.45 per step):")
    print("phase weight" + "".join(f"{t + ' weight':>22s}" for t in ENG_TYPES[1:]))
    for level in args.levels:
        penalty = level * mean["phase"]
        cells = "".join(f"{float(f'{penalty / mean[t]:.2g}'):>22.2g}" for t in ENG_TYPES[1:])
        print(f"{level:12.2g}{cells}    ({penalty:.3f})")


if __name__ == "__main__":
    main()
