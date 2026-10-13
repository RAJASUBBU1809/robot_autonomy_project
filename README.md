# Go2 TPMS Feet: energy-efficient quadruped walking in MuJoCo

A simulation reproduction of

> Lee et al. (2026), *"Energy-efficient Quadruped Robot Locomotion Via TPMS-Based Foot Design and
> Deep Reinforcement Learning"*, IJPEM-Green Technology, doi
> [10.1007/s40684-026-00895-5](https://doi.org/10.1007/s40684-026-00895-5)

using the **Unitree Go2** (MuJoCo Menagerie) instead of the paper's RBQ-10. Each foot gets the paper's
measured compliance (**TPMS-D 60 %** lattice vs **solid** resin hemisphere). A walking policy is
trained with **PPO**, and energy use is compared across 0.2–1.0 m/s, like the paper's Table 5.

Training runs on a laptop CPU (Stable-Baselines3) or on a free GPU (MJX + Brax PPO on Kaggle). Both
paths are evaluated in the same CPU simulation, so their results are directly comparable.

**Current focus (from 2026-10-12):** no foot design. The study asks how the training reward and its
hyperparameters make a regular quadruped (the `rigid` foot) use less energy: the paper's
energy-phase term vs standard power and torque penalties, a weight sweep, and repeated seeds. See
[Energy study](#energy-study-rigid-foot-current). The TPMS/solid foot models are kept but unused.

---

## Contents
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Setup (Windows / PowerShell)](#setup-windows--powershell)
- [Quick start: CPU](#quick-start-cpu)
- [GPU training on Kaggle](#gpu-training-on-kaggle)
- [Experiments](#experiments)
- [Differences from the paper](#differences-from-the-paper)
- [Notes and pitfalls](#notes-and-pitfalls)
- [Credits](#credits)

---

## How it works

### 1. Compliant feet (`go2tpms/compliant_foot.py`)
The paper fits each foot's loading curve as `F[N] = a0 + a1·d + a2·d²` (d in mm):

| Foot | a0 | a1 | a2 | Energy loss | Compression at 300 N |
|---|---|---|---|---|---|
| `tpms60`: TPMS-D, 60 % density (paper's best) | 4.6105 | −3.8891 | 7.9422 | 12.3 % | 6.26 mm |
| `solid`: solid hemisphere, same resin (baseline) | 9.639 | −2.472 | 15.505 | 30.4 % | 4.18 mm |

In simulation each foot is a 30 mm sphere on a short **slide joint** (0–8 mm). The curve's force is
applied to that joint at every 1 ms physics step:
- Below the parabola's minimum, the force is held flat.
- While the foot is unloading, the force is reduced by the measured loss (hysteresis), with a smooth
  switch `tanh(v / 5 mm/s)`.

`python -m go2tpms.compliant_foot` re-runs the paper's compression test. It matches the loss exactly,
and d_max and the energies to within 2–7 %.

A third type, `rigid`, has no spring: `lock_feet()` locks the slide joint with a stiff joint limit
(≈ 0.1 mm standing, ≤ ~1 mm at touchdown impacts), like a normal robot foot. Stage 1 of the study.

### 2. Robot model (`scripts/build_model.py`)
Builds `models/go2_model/` from Menagerie's `unitree_go2`:
- Replaces each foot with the sliding 30 mm sphere (mass 30 g).
- Uses a **stiff** foot–ground contact, so all foot compliance comes from the TPMS/solid spring.
  Menagerie's soft foot contact sinks about 15 mm on its own.
- Uses a 1 ms timestep and a pyramidal friction cone.
- Fixes the `home` keyframe for the new joints.

### 3. Environment (`go2tpms/go2_env.py`, Gymnasium)
- **Action** (12): joint-angle offsets from the standing pose (±0.5 rad) at **50 Hz**. With ±0.25 rad
  the thigh offsets saturated and the robot could not walk faster than ~0.8 m/s. A PD
  servo inside MuJoCo (kp 40, kd 1, clipped to the motor torque limits) tracks them every 1 ms.
- **Observation** (46): base linear and angular velocity, projected gravity, commanded forward speed,
  joint positions and velocities, last action.
- **Command:** forward speed, sampled uniformly from 0.2–1.0 m/s per episode (20 s).
- **Reward:** the sum of the terms below, × (dt·50); −2 on a fall (tilt > 60° or base below 0.12 m).

| Term | Weight / form |
|---|---|
| forward-speed tracking | `1.0 · exp(−((v_cmd−vx)² + vy²)/0.25)` |
| yaw-rate tracking (0) | `0.5 · exp(−ωz²/0.25)` |
| vertical velocity | `−2.0 · vz²` |
| roll/pitch rate | `−0.05 · ‖ωxy‖²` |
| orientation | `−1.0 · ‖g_xy‖²` |
| torques | `−1e−4 · ‖τ‖²` |
| action rate | `−0.01 · ‖aₜ − aₜ₋₁‖²` |
| feet air time | `1.0 · Σ (t_air − 0.25)` at touchdown |
| **energy** (`--eng-type`, weight `--eng-weight`, default phase 3e−3) | one of the three below |
| &nbsp;&nbsp;`phase` (paper Eq. 15) | `w · −Σ_legs ‖(ΔW_leg,t − ΔW_leg,t−1) − (ΔE_a,t − ΔE_a,t−1)‖²` |
| &nbsp;&nbsp;`power` (Fu et al. 2021) | `w · −Σ_joints |τ·ω|` |
| &nbsp;&nbsp;`torque` | `w · −Σ_joints τ²` (on top of the `torques` term) |

`ΔW_leg` is the mean mechanical power of each leg's motors and `ΔE_a` the elastic power of its foot,
both averaged over the control step. The term rewards a gait whose leg power changes in phase with
the foot storing and returning energy. With the rigid foot `ΔE_a = 0`, so it penalises changes in
leg power (a power-smoothness term). The three types have different units:
`scripts/calibrate_eng.py` finds weights with the same mean penalty.

`go2tpms/go2_mjx_env.py` is the same environment in **JAX/MJX** for GPU training, with two
differences: only the feet collide with the floor, and the solver uses 10 iterations instead of 100.
`scripts/check_mjx.py` verifies it step by step against the CPU environment.

### 4. Evaluation (`scripts/play.py --evaluate`)
For each speed 0.2, 0.4, …, 1.0 m/s: 2 s to settle, then 30 s of walking. It prints the actual speed
and, averaged over the 30 s:
- **mechanical power**: positive motor work (W), as in the earlier results;
- **braking power**: negative motor work (W), energy the motors absorb;
- **electrical power (estimate)**: Σ_joints max(τω + α·τ², 0), i.e. motor work plus copper heat,
  with no regeneration. α = 1.5·R/(K_t·G)² from a Go2 motor teardown
  ([simplexitypd.com](https://www.simplexitypd.com/blog/unitree-go2-motor-teardown/): R_line 0.44 Ω,
  K_t 0.26 Nm/A at the rotor, gear 6.22:1; the calf has a further 1.92:1 linkage), giving
  0.126 W/(Nm)² for hip/thigh and 0.034 for the calf. This value is uncertain (Unitree's published
  constant for the older Go1 motor differs), so `--heat-scale` (0 = mechanical only, 2 = double)
  shows how much a conclusion depends on it;
- **cost of transport** (mechanical and electrical) = power / (m·g·v), and the maximum foot compression.

---

## Project structure

```
mujoco_go2/
├── go2tpms/                  # the library (pip install -e .)
│   ├── compliant_foot.py     #   foot force curves + hysteresis (paper data)
│   ├── go2_env.py            #   Gymnasium env, CPU MuJoCo (SB3 training + evaluation)
│   ├── go2_mjx_env.py        #   same env in JAX/MJX (GPU training with Brax)
│   ├── numpy_policy.py       #   export / run Brax policies with plain NumPy
│   ├── runs.py               #   run folders, config.json, checkpoint lookup
│   └── paths.py              #   project paths
├── scripts/                  # command-line tools
│   ├── build_model.py        #   Menagerie Go2  ->  models/go2_model/ (compliant feet)
│   ├── stand_test.py         #   stand / drop test, prints foot compression
│   ├── train.py              #   PPO training on CPU (Stable-Baselines3)
│   ├── train_mjx.py          #   PPO training on GPU (MJX + Brax)
│   ├── play.py               #   watch a policy, or --evaluate (Table 5 style)
│   ├── compare.py            #   several policies side by side (% change vs the first)
│   ├── calibrate_eng.py      #   equal-pressure weights for the three energy terms
│   ├── sweep.py              #   energy study: evaluate all rigid runs -> results/ CSV + plots
│   ├── check_mjx.py          #   MJX env vs CPU env verification
│   └── bench_mjx.py          #   GPU speed benchmark
├── kaggle/
│   ├── go2_mjx_kaggle.ipynb  #   notebook for GPU training on Kaggle
│   └── make_bundle.ps1       #   zips the project for upload as a Kaggle dataset
├── pyproject.toml            # package + dependencies ([train] CPU, [mjx] GPU)
├── README.md
│
│   (not in git, see .gitignore)
├── mujoco_menagerie/         # git clone of MuJoCo Menagerie (input)
├── models/go2_model/         # generated by scripts/build_model.py
├── runs/<name>/              # training output: policy, checkpoints, config.json, tb/ logs
├── results/                  # sweep.csv + thesis plots from scripts/sweep.py
├── archive/                  # old exploration files
└── venv/, venv_mjx/          # virtual environments
```

---

## Setup (Windows / PowerShell)

Python 3.10+ (tested with 3.11). Commands are run from the project folder.

```powershell
cd C:\mujoco_go2

# 1. Robot model source
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# 2. Environment for CPU training (SB3)
python -m venv venv
venv\Scripts\Activate.ps1
pip install -e ".[train]"

# 3. Build the compliant-foot model and check it
python scripts\build_model.py
python -m go2tpms.compliant_foot
python scripts\stand_test.py            # viewer; --headless prints numbers only
```

Optional: a second environment for testing the GPU code locally (it runs on the CPU, slowly). Keep it
separate so the JAX packages don't interfere with the SB3 environment:

```powershell
python -m venv venv_mjx
venv_mjx\Scripts\python.exe -m pip install -e ".[mjx]"
venv_mjx\Scripts\python.exe scripts\check_mjx.py
```

---

## Quick start: CPU

```powershell
venv\Scripts\Activate.ps1

# 1. Smoke test (a few minutes): checks the training pipeline, the robot will not walk yet
python scripts\train.py --name test --steps 200000 --envs 2

# 2. Real run (~6-8 h on a 4-core laptop; plug in the charger, power mode "Best performance")
python scripts\train.py --name tpms_eng

# 3. Watch progress in a second terminal -> http://localhost:6006
tensorboard --logdir runs

# 4. Look at the gait / evaluate
python scripts\play.py --name tpms_eng --ckpt latest --speed 0.6
python scripts\play.py --name tpms_eng --evaluate
```

- **Ctrl+C** stops training and saves first. Continue later with `--resume`.
- In TensorBoard, watch `env/speed` and `rew/track_vx`. If the speed stays near 0 after about 3M
  steps, the robot has learned to stand still, and the rewards need tuning.
- `play.py --foot solid` runs the same policy on the other foot.

---

## GPU training on Kaggle

The free Kaggle tier provides a GPU (use **T4 x2**), about 30 GPU-hours a week, and 12 h sessions
that keep running with the browser closed.

1. On your PC, after `build_model.py`:
   ```powershell
   powershell -ExecutionPolicy Bypass -File kaggle\make_bundle.ps1
   ```
2. On kaggle.com, phone-verify your account. Then go to **Datasets → New Dataset**, upload
   `go2_bundle.zip`, name it `go2-bundle`, and keep it private.
3. **Notebooks → New → File → Import notebook** and choose `kaggle\go2_mjx_kaggle.ipynb`. Set
   **Accelerator: GPU T4 x2** and **Internet: On**, and add the `go2-bundle` dataset as input.
4. Run the install, check and benchmark cells. Then, for the full run, use **Save Version → Save &
   Run All**.
5. Download `runs.zip` from the version's **Output** tab, unzip it into the project folder, and
   evaluate locally:
   ```powershell
   python scripts\play.py --name tpms_eng_mjx --evaluate
   ```

`train_mjx.py` exports the policy as `policy_mjx.npz` (plain NumPy). `play.py` detects this from
the run's `config.json`, so GPU-trained policies need no JAX on your PC.

To compare runs side by side (speed, power and cost of transport per speed, % change vs the first
run), write each run as `name[:foot][@checkpoint]`:
```powershell
python scripts\compare.py rigid_base rigid_ft0@43253760 rigid_ft_eng@43253760 rigid_ft_eng_lo@43253760
python scripts\compare.py tpms_eng_mjx_v3@43253760 tpms_eng_mjx_v3:solid@43253760   # same policy, solid foot
```

---

## Experiments

### Energy study (rigid foot, current)
Question: how much energy can the reward save on a regular quadruped, which energy term works
best, and how does the saving trade against speed tracking? Every run fine-tunes the same walking
policy `rigid_base` (rigid foot, 100M steps, no energy term) for 50M steps with one energy term
(see the reward table above). The control is `rigid_ft0`: the same fine-tune with no energy term,
so any difference comes from the reward.

**Weights.** `calibrate_eng.py` runs `rigid_base` at 0.2–1.0 m/s and scales the weights so the mean
penalty is equal ("penalty level" = reward lost per control step at the start; the walking reward
is ≈ 1.45/step). Unweighted terms per step on `rigid_base`:

| cmd [m/s] | phase | power [W] | torque [N²m²] |
|---|---|---|---|
| 0.2 | 13.9 | 15.9 | 237 |
| 0.6 | 101 | 51.3 | 325 |
| 1.0 | 498 | 110 | 526 |
| mean | 195 | 57.7 | 357 |

| penalty level | phase | power | torque |
|---|---|---|---|
| 0.020 | 1e-4 | | |
| 0.059 | 3e-4 (`rigid_ft_eng_lo`) | 1e-3 | 1.6e-4 |
| 0.195 | 1e-3 | 3.4e-3 | 5.5e-4 |
| 0.585 | 3e-3 (`rigid_ft_eng`) | 1e-2 | 1.6e-3 |
| 1.95 | 1e-2 | | |

The phase term grows 36× from 0.2 to 1.0 m/s, power 7× and torque 2.2×, so at the same mean level
the phase term pushes mostly on fast walking and the torque term also on slow walking. The phase
3e-3 level (0.585/step) matches the reward drop seen at the start of `rigid_ft_eng` on Kaggle
(0.557/step).

**Runs** (Kaggle notebook section 4: set `BATCH`, one commit per batch; bundle made with
`make_bundle.ps1 -Init rigid_base`). Names: `rigid_<type>_<weight>`.

| Batch | Runs | GPU time |
|---|---|---|
| done | `rigid_ft0` (none), `rigid_ft_eng` (phase 3e-3), `rigid_ft_eng_lo` (phase 3e-4) | |
| A | phase 1e-4, 1e-3, 1e-2; power 1e-3, 3.4e-3 | ≈ 9.2 h |
| B | power 1e-2; torque 1.6e-4, 5.5e-4, 1.6e-3 | ≈ 7.3 h |
| C, D | seeds 1 and 2: new base (100M) + ft0 + phase 3e-3 + the best power/torque run | ≈ 8.7 h each |

C and D are defined after A and B have been evaluated.

**Evaluation.** After downloading a batch:
```powershell
python scripts\sweep.py                       # all rigid runs: table + results\sweep_*.png
python scripts\compare.py rigid_ft0 rigid_phase_1e-3 rigid_power_3.4e-3 rigid_torque_5.5e-4
```
`sweep.py` caches results in `results/sweep.csv`, so only new runs are simulated. Its figures:
`sweep_cot.png` (CoT change vs penalty level, mechanical and electrical; error bars over seeds) and
`sweep_tradeoff.png` (electrical CoT change vs speed tracking error, one marker per run).
Compare by cost of transport at the *actual* speed, and check the tracking error: a policy that
walks slower than commanded saves power for free.

Notes:
- **Fine-tuning.** From scratch, the jerky early gait makes the energy penalty so large that the
  policy learns to stand still, so the energy term is only added to a policy that already walks.
- **Final policy vs checkpoint.** All stage-1 fine-tunes ended stable (episode length 1000), so the
  final policies are evaluated; `--ckpt 43253760` gives the same picture (`sweep.py --ckpt` too).
  If a run collapses for good, rerun it with `--lr 1e-4`.

### Stage 1 results: rigid foot (2026-10-12)
Kaggle T4×2, 8.8 h for all four runs. `rigid_ft_eng` (3e-3) started at reward 898 (base 1455), fell a
lot at 13–16M steps (episode length 98), then recovered and walked; the other runs were stable. No
falls in the 30 s evaluations. CoT, final policies (`compare.py rigid_base rigid_ft0 rigid_ft_eng rigid_ft_eng_lo`):

| cmd [m/s] | base | ft0 (control) | ft_eng 3e-3 | vs ft0 | ft_eng_lo 3e-4 | vs ft0 |
|---|---|---|---|---|---|---|
| 0.2 | 0.397 | 0.401 | 0.283 | −29 % | 0.325 | −19 % |
| 0.4 | 0.448 | 0.459 | 0.350 | −24 % | 0.383 | −17 % |
| 0.6 | 0.501 | 0.500 | 0.424 | −15 % | 0.464 | −7 % |
| 0.8 | 0.603 | 0.610 | 0.515 | −16 % | 0.567 | −7 % |
| 1.0 | 0.694 | 0.728 | 0.615 | −16 % | 0.665 | −9 % |

Mechanical power at 1.0 m/s: base 101.8, ft0 107.1, ft_eng 84.7, ft_eng_lo 97.0 W. `ft_eng` walks
slower at 1.0 m/s (0.91 vs 0.98 m/s), but its CoT is still ~7 % below the base's CoT interpolated to
0.91 m/s. At checkpoint 43,253,760 the picture is the same (ft_eng −11 … −33 %, ft_eng_lo −7 … −26 %
vs ft0). Extra training alone (ft0 vs base) changes CoT by −0 … +7 %, a rough noise level for one
seed. → On a rigid robot the energy-phase reward (here a leg-power smoothness term) lowers the
positive mechanical power, more with the larger weight.

With braking and motor heat (`compare.py rigid_ft0 rigid_base rigid_ft_eng rigid_ft_eng_lo`, vs ft0):

| cmd [m/s] | braking ft_eng / ft_eng_lo | electrical power ft_eng / ft_eng_lo | electrical CoT ft_eng / ft_eng_lo | tracking error ft0 / ft_eng [m/s] |
|---|---|---|---|---|
| 0.2 | +3 % / +11 % | −12 % / +2 % | −9 % / −3 % | 0.008 / 0.013 |
| 0.4 | +4 % / +2 % | −19 % / −12 % | −16 % / −12 % | 0.016 / 0.026 |
| 0.6 | +4 % / +5 % | −13 % / −7 % | −9 % / −5 % | 0.019 / 0.034 |
| 0.8 | −8 % / −3 % | −15 % / −6 % | −11 % / −6 % | 0.021 / 0.033 |
| 1.0 | +2 % / −5 % | −16 % / −8 % | −10 % / −7 % | 0.027 / 0.085 |

Braking work (4–8 W) hardly changes, so the saving is not moved into braking. Motor heat is
12–26 W of the 24–132 W electrical power and the energy-phase term does not reduce it (it rises
by ~1 W), so the
electrical saving (−9 … −16 %) is about half the mechanical one. ft_eng tracks speed worse,
especially at 1.0 m/s.

### Earlier exploratory runs
The `_mjx`, `_mjx_v2` and `_mjx_v3` runs below came before the staged study (they also used the
solid foot, which the staged study leaves out). They found the
action-range limit, the weak energy-phase weight and the need for fine-tuning.

### Run log
- **`tpms_eng_mjx`** (2026-10-09, Kaggle T4×2, 100M steps, 3.4 h, action scale 0.25): walks without
  falling and tracks 0.2–0.6 m/s within 3 %, but reaches only 0.71 / 0.79 m/s at 0.8 / 1.0 m/s.
  Mechanical power 14.6 / 32.8 / 55.8 / 77.5 / 94.6 W at 0.2 … 1.0 m/s; peak foot compression ≤ 4.5 mm.
  Cause of the speed limit: thigh actions saturated 71 % of the time at 1.0 m/s (torques never at
  their limit). Action scale is now 0.5 rad; the comparison runs use the `_v2` suffix.
  (Moved to `runs_v1/`; v2 and v3 are in `runs_v2/` and `runs_v3/`.)
- **`tpms_eng_mjx_v2`** (2026-10-10, Kaggle T4×2, 100M steps, 3.3 h, action scale 0.5): tracks every
  speed (0.20 / 0.40 / 0.59 / 0.80 / 0.98 m/s). Mechanical power 10.8 / 26.2 / 45.3 / 75.1 / 112.3 W,
  CoT 0.35 / 0.44 / 0.51 / 0.63 / 0.76; peak foot compression ≤ 5.9 mm. Final reward ≈ 1440
  (v1: ≈ 1380).
  Swapping in the solid foot under this same policy gives the same motor power (±0.7 W): the TPMS
  foot loses 1.2–1.9 W less to hysteresis, but the energy is not returned at push-off. Reward
  breakdown: energy_phase was only −0.001 … −0.003 per step out of ≈ 1.45 → weight raised to 1e-2 (v3).
- **`tpms_eng_mjx_v3` from scratch, 1e-2** (2026-10-10, stopped at 16M steps): episode reward −2071 at
  5M, then ~600 with mean speed 0.00 m/s: the policy learned to stand still. → fine-tune instead.
- **Fine-tune from v2 with 1e-2** (2026-10-10): mean speed fell to 0 again. → weight 3e-3.
- **`tpms_eng_mjx_v3`** (2026-10-11, fine-tuned from v2, 3e-3, 50M steps): stable (reward ≈ 1400–1420,
  speed ≈ 0.6, no falls) up to 43.3M steps, then collapsed (episode length 1000 → 165, robot falls).
  The final policy is unusable; evaluate the checkpoint `--ckpt 43253760`, and evaluate every v3 run
  at that same checkpoint step (same STEPS → same checkpoint steps) for a fair comparison.
  At 43.25M: speed 0.21 / 0.39 / 0.58 / 0.78 / 0.92 m/s, mechanical power 10.6 / 24.1 / 43.1 / 71.5 /
  97.9 W, CoT 0.34 / 0.42 / 0.49 / 0.61 / 0.71, peak compression ≤ 6.8 mm. vs v2: CoT 2–7 % lower,
  more foot compression, slightly slower at 1.0 m/s. (Reward effect only; foot effect needs solid_eng.)

For reference, the paper reports that the TPMS foot cut battery power by 1.4–6.2 % at 0.4–1.0 m/s
compared with the solid foot.

---

## Differences from the paper

| | Paper | Here |
|---|---|---|
| Robot | RBQ-10 | Unitree Go2 (Menagerie model, 15.3 kg) |
| Simulator / RL | Isaac Lab, PPO | MuJoCo (CPU) / MJX (GPU), PPO (SB3 / Brax) |
| Policy rate | 100 Hz | 50 Hz |
| Foot | physical TPMS print + emulated spring in sim | emulated spring on a slide joint (paper's fitted curves) |
| Energy metric | measured battery power | simulated positive mechanical motor power (no motor/electrical losses) |
| Command | not specified in detail | forward speed only, 0.2–1.0 m/s |

---

## Notes and pitfalls

- **Laptop on battery:** Windows throttles the CPU, so training speed can drop by half. Train plugged
  in with power mode set to "Best performance".
- **Parallel envs:** `train.py` defaults to 6 envs. On a 4-core / 8-thread CPU more is slower.
- **Re-run `build_model.py`** after changing anything in it. `models/` is generated output.
- **MJX solver iterations:** with the 4 iterations Playground uses for other robots, the stiff foot
  contact diverges (the robot gets launched into the air). 10 is stable. Re-check with
  `check_mjx.py` if you change it.
- **JAX version:** brax 0.14.2 needs JAX < 0.10. Keep the pins in `pyproject.toml`.
- **2 ms physics option** (GPU, about 2x faster): `train_mjx.py --sim-dt 0.002 --foot-armature 0.29`.
  The larger foot armature keeps the stiff spring stable at 2 ms but makes the foot effectively
  heavier, so prefer 1 ms for the final comparison.

---

## Credits

- Paper: Lee et al. (2026), IJPEM-GT, doi 10.1007/s40684-026-00895-5. Foot curves and loss ratios
  are taken from it.
- Robot model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `unitree_go2`,
  © Unitree Robotics, BSD-3-Clause.
- Libraries: [MuJoCo / MJX](https://github.com/google-deepmind/mujoco),
  [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground),
  [Brax](https://github.com/google/brax), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
  [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
