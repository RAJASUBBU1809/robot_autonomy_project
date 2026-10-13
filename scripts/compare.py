"""Compare the power use of several policies side by side (like the paper's Table 5).

    python scripts/compare.py rigid_base rigid_ft0@43253760 rigid_ft_eng@43253760 rigid_ft_eng_lo@43253760
    python scripts/compare.py tpms_eng_mjx_v3 solid_eng_mjx_v3 --ckpt 43253760
    python scripts/compare.py tpms_eng_mjx_v3@43253760 tpms_eng_mjx_v3:solid@43253760  # same policy, solid foot

Each run is written name[:foot][@ckpt]. The foot defaults to the one the run was trained with,
the checkpoint to --ckpt (default: the final policy). Same procedure as play.py --evaluate
(30 s per speed after 2 s settling, CPU env). Changes are relative to the FIRST run:
negative = uses less than the first run. Compare cost of transport when the actual speeds
differ: slower walking always needs less power. Electrical power is an estimate (motor work +
copper heat, no regeneration; see go2_env.heat_coef); --heat-scale 0 / 2 shows how much the
conclusion depends on the heat estimate.
"""
import argparse
import string

from go2tpms.runs import load_config, run_dir
from play import load, run

SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0)


def evaluate(spec, default_ckpt, seconds, heat_scale=1.0):
    rest, _, ckpt = spec.partition("@")
    name, _, foot = rest.partition(":")
    foot = foot or load_config(run_dir(name)).get("foot", "tpms60")
    ckpt = ckpt or default_ckpt
    env, act = load(name, foot, ckpt, heat_scale=heat_scale)
    weight = env.model.body_subtreemass[1] * 9.81
    rows = {}
    for speed in SPEEDS:
        result, fell_at = run(env, act, speed, seconds=seconds)
        if result is None:
            print(f"  {spec} at {speed} m/s: robot fell after {fell_at:.1f} s")
            continue
        v = max(result["speed"], 1e-3)
        rows[speed] = {**result, "cot": result["mech_power"] / (weight * v),
                       "cot_elec": result["elec_power"] / (weight * v)}
    return f"{name} ({foot}, {'ckpt ' + ckpt if ckpt else 'final policy'})", rows


def table(title, key, labels, results, fmt, change):
    print(f"\n{title}")
    print("cmd  " + "".join(f"{l:>18s}" for l in labels))
    for speed in SPEEDS:
        cells, ref = [], results[0].get(speed)
        for rows in results:
            if speed not in rows:
                cells.append(f"{'fell':>18s}")
                continue
            text = fmt.format(rows[speed][key])
            if change and rows is not results[0] and ref:
                text += f" ({100 * (rows[speed][key] - ref[key]) / ref[key]:+.1f}%)"
            cells.append(f"{text:>18s}")
        print(f"{speed:.1f}  " + "".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="name[:foot][@ckpt], e.g. rigid_ft_eng@43253760")
    parser.add_argument("--ckpt", help="checkpoint for runs without @ (step count or 'latest')")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--heat-scale", type=float, default=1.0,
                        help="scale of the motor heat estimate in electrical power (0 = none)")
    args = parser.parse_args()

    labels = list(string.ascii_uppercase[:len(args.runs)])
    results = []
    for label, spec in zip(labels, args.runs):
        description, rows = evaluate(spec, args.ckpt, args.seconds, args.heat_scale)
        print(f"{label} = {description}")
        results.append(rows)

    table("Actual speed [m/s]", "speed", labels, results, "{:.2f}", change=False)
    table("Speed tracking error |actual - cmd| [m/s]", "track_err", labels, results, "{:.3f}", change=False)
    table("Mechanical power, positive motor work [W]  (change vs A)", "mech_power", labels, results,
          "{:.1f}", change=True)
    table("Braking power, negative motor work [W]  (change vs A)", "brake_power", labels, results,
          "{:.1f}", change=True)
    table(f"Electrical power, estimated, heat scale {args.heat_scale:g} [W]  (change vs A)", "elec_power",
          labels, results, "{:.1f}", change=True)
    table("Cost of transport, mechanical  (change vs A)", "cot", labels, results, "{:.3f}", change=True)
    table("Cost of transport, electrical  (change vs A)", "cot_elec", labels, results, "{:.3f}", change=True)


if __name__ == "__main__":
    main()
