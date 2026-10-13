"""Energy study: evaluate all rigid-foot fine-tunes, save a CSV and plot the thesis figures.

    python scripts/sweep.py                    # every rigid run in runs/ (final policies)
    python scripts/sweep.py --ckpt 43253760    # same checkpoint step for all fine-tunes
    python scripts/sweep.py --plot-only        # re-plot from results/sweep.csv

Each run is walked 30 s at 0.2 ... 1.0 m/s in the CPU env (as play.py --evaluate), once per heat
scale (electrical power estimate; heat scale 0 is the mechanical power). Results are cached in
results/sweep.csv, so only new runs are simulated. Changes are relative to the no-energy-term
fine-tune of the same seed (eng_weight 0, init_from set: rigid_ft0 for seed 0).

"Penalty level" puts the three energy types on one axis: weight x mean unweighted term of
rigid_base (= reward lost per control step at the start of the fine-tune; see calibrate_eng.py).
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from go2tpms.go2_env import ENG_TYPES
from go2tpms.paths import RUNS_DIR
from go2tpms.runs import load_config, run_dir

SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CSV_PATH = os.path.join(RESULTS, "sweep.csv")
FIELDS = ("run", "ckpt", "heat_scale", "seed", "eng_type", "eng_weight", "is_base", "cmd", "fell",
          "speed", "track_err", "mech_power", "elec_power", "brake_power", "torque_sq",
          "eng_phase", "eng_power", "eng_torque", "cot_mech", "cot_elec")
# Validated categorical palette (dataviz skill reference, light mode), fixed order per type.
COLORS = {"phase": "#2a78d6", "power": "#eb6834", "torque": "#1baf7a"}
MARKERS = {"phase": "o", "power": "s", "torque": "^"}


def rigid_runs():
    names = []
    for name in sorted(os.listdir(RUNS_DIR)):
        cfg = load_config(run_dir(name))
        if cfg.get("foot") == "rigid" and cfg.get("trainer") == "mjx":
            names.append(name)
    return names


def evaluate(job):
    """One run at one heat scale -> list of CSV rows (one per speed)."""
    name, ckpt, heat_scale, seconds = job
    from play import load, run  # imported in the worker process

    cfg = load_config(run_dir(name))
    is_base = not cfg.get("init_from")
    env, act = load(name, "rigid", None if is_base else ckpt, heat_scale=heat_scale)
    weight = env.model.body_subtreemass[1] * 9.81
    rows = []
    for cmd in SPEEDS:
        row = {"run": name, "ckpt": "final" if is_base or not ckpt else ckpt, "heat_scale": heat_scale,
               "seed": cfg.get("seed", 0), "eng_type": cfg.get("eng_type", "phase"),
               "eng_weight": cfg.get("eng_weight", 0.0), "is_base": int(is_base), "cmd": cmd}
        r, _ = run(env, act, cmd, seconds=seconds)
        if r is None:
            rows.append({**row, "fell": 1})
            continue
        v = max(r["speed"], 1e-3)
        rows.append({**row, "fell": 0, **{k: r[k] for k in FIELDS if k in r},
                     "cot_mech": r["mech_power"] / (weight * v), "cot_elec": r["elec_power"] / (weight * v)})
    return rows


def load_csv():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            if k not in ("run", "ckpt", "eng_type") and r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def save_csv(rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarise(rows, ckpt, heat_scale):
    """Per fine-tune: mean % change vs the seed's no-energy fine-tune, averaged over speeds."""
    want = "final" if not ckpt else ckpt
    sel = [r for r in rows if r["heat_scale"] == heat_scale and (r["is_base"] or r["ckpt"] == want)]
    by_run = {}
    for r in sel:
        by_run.setdefault(r["run"], {})[r["cmd"]] = r
    base_raw = {}
    for name, rs in by_run.items():
        if rs[SPEEDS[0]]["is_base"] and rs[SPEEDS[0]]["seed"] == 0 and not any(x["fell"] for x in rs.values()):
            base_raw = {t: np.mean([rs[c][f"eng_{t}"] for c in SPEEDS]) for t in ENG_TYPES}
    refs = {rs[SPEEDS[0]]["seed"]: name for name, rs in by_run.items()
            if not rs[SPEEDS[0]]["is_base"] and rs[SPEEDS[0]]["eng_weight"] == 0}
    out = []
    for name, rs in by_run.items():
        first = rs[SPEEDS[0]]
        ref = by_run.get(refs.get(first["seed"]))
        if first["is_base"] or ref is None:
            continue
        fell = any(rs[c]["fell"] for c in SPEEDS)
        entry = {"run": name, "seed": first["seed"], "eng_type": first["eng_type"],
                 "eng_weight": first["eng_weight"], "fell": fell,
                 "level": first["eng_weight"] * -base_raw.get(first["eng_type"], np.nan)}
        if not fell:
            entry["track_err"] = np.mean([rs[c]["track_err"] for c in SPEEDS])
            for k in ("cot_mech", "cot_elec", "brake_power"):
                entry[k] = 100 * np.mean([rs[c][k] / ref[c][k] - 1 for c in SPEEDS])
        out.append(entry)
    return sorted(out, key=lambda e: (e["eng_type"], e["eng_weight"], e["seed"]))


def print_table(summary, heat_scale):
    print(f"\nMean change vs the no-energy fine-tune of the same seed, averaged over 0.2 ... 1.0 m/s"
          f" (electrical: heat scale {heat_scale:g})")
    print(f"{'run':28s}{'type':>8s}{'weight':>9s}{'level':>8s}{'track err':>11s}"
          f"{'CoT mech':>10s}{'CoT elec':>10s}{'braking':>9s}")
    for e in summary:
        head = f"{e['run']:28s}{e['eng_type']:>8s}{e['eng_weight']:>9.2g}{e['level']:>8.3f}"
        if e["fell"]:
            print(head + "   fell during evaluation")
        else:
            print(head + f"{e['track_err']:>10.3f} {e['cot_mech']:>+9.1f}%{e['cot_elec']:>+9.1f}%"
                         f"{e['brake_power']:>+8.1f}%")


def group(summary, key):
    """eng_type -> sorted list of (eng_weight, level, mean, std, n) over seeds, for one metric."""
    out = {}
    for t in ENG_TYPES:
        pts = {}
        for e in summary:
            if e["eng_type"] == t and not e["fell"] and e["eng_weight"] > 0:
                pts.setdefault(e["eng_weight"], []).append((e["level"], e[key]))
        out[t] = [(w, np.mean([p[0] for p in v]), np.mean([p[1] for p in v]),
                   np.std([p[1] for p in v]), len(v)) for w, v in sorted(pts.items())]
    return out


def plot(summaries, ckpt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    suffix = f"_{ckpt}" if ckpt else ""
    heat_scales = sorted(summaries)
    main_hs = 1.0 if 1.0 in summaries else heat_scales[0]

    # Figure 1: CoT change vs penalty level, mechanical and electrical (one panel per heat scale).
    panels = [("cot_mech", "Mechanical CoT", main_hs)] + [
        ("cot_elec", f"Electrical CoT (heat x{hs:g})", hs) for hs in heat_scales]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0), sharey=True)
    for ax, (key, title, hs) in zip(np.atleast_1d(axes), panels):
        for t, pts in group(summaries[hs], key).items():
            if not pts:
                continue
            x, y, s = [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
            ax.errorbar(x, y, yerr=s, color=COLORS[t], marker=MARKERS[t], ms=6, lw=2, capsize=3,
                        mec="white", mew=1, label=t)
        ax.axhline(0, color="#999999", lw=1)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9, color="#333333")
        ax.set_xlabel("penalty level (reward/step at start)")
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    np.atleast_1d(axes)[0].set_ylabel("change vs no energy term [%]")
    np.atleast_1d(axes)[0].legend(frameon=False, title="energy term", title_fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(RESULTS, f"sweep_cot{suffix}.png")
    fig.savefig(f1, dpi=200)

    # Figure 2: trade-off, electrical CoT change vs speed tracking error, one marker per run.
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ref_err = []
    for e in summaries[main_hs]:
        if e["fell"]:
            continue
        ax.scatter(e["track_err"], e["cot_elec"], s=40, color=COLORS[e["eng_type"]],
                   marker=MARKERS[e["eng_type"]], edgecolors="white", linewidths=1, zorder=3,
                   label=e["eng_type"])
        ax.annotate(f"{e['eng_weight']:g}", (e["track_err"], e["cot_elec"]), fontsize=7,
                    color="#555555", xytext=(4, 3), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    keep = {l: h for h, l in zip(handles, labels)}
    ax.legend(keep.values(), keep.keys(), frameon=False, title="energy term", title_fontsize=8)
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xlabel("mean speed tracking error [m/s]")
    ax.set_ylabel(f"electrical CoT change [%] (heat x{main_hs:g})")
    ax.grid(color="#e5e5e5", lw=0.8)
    fig.tight_layout()
    f2 = os.path.join(RESULTS, f"sweep_tradeoff{suffix}.png")
    fig.savefig(f2, dpi=200)
    print(f"\nWrote {f1}\n      {f2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run names (default: every rigid MJX run in runs/)")
    parser.add_argument("--ckpt", help="checkpoint step for the fine-tunes (default: final policy)")
    parser.add_argument("--heat-scales", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    rows = load_csv()
    if not args.plot_only:
        names = args.runs or rigid_runs()
        want = "final" if not args.ckpt else args.ckpt
        done = {(r["run"], r["ckpt"], r["heat_scale"]) for r in rows}
        jobs = []
        for name in names:
            is_base = not load_config(run_dir(name)).get("init_from")
            for hs in args.heat_scales:
                if (name, "final" if is_base else want, hs) not in done:
                    jobs.append((name, args.ckpt, hs, args.seconds))
        print(f"{len(names)} runs, {len(jobs)} evaluations to simulate "
              f"({len(rows) // len(SPEEDS)} cached in {CSV_PATH})")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for job, new in zip(jobs, pool.map(evaluate, jobs)):
                rows += new
                save_csv(rows)  # after every run, so an interrupted sweep keeps its results
                print(f"  done: {job[0]} (heat x{job[2]:g})", flush=True)

    summaries = {hs: summarise(rows, args.ckpt, hs) for hs in args.heat_scales}
    for hs, summary in summaries.items():
        print_table(summary, hs)
    plot(summaries, args.ckpt)


if __name__ == "__main__":
    main()
