"""Go2 quadruped with compliant TPMS feet, trained with RL in MuJoCo.

Reproduction of Lee et al. (2026), IJPEM-GT, doi 10.1007/s40684-026-00895-5.

Modules:
    compliant_foot  paper's foot force curves + hysteresis (TPMS-60 %, solid)
    go2_env         Gymnasium env (CPU MuJoCo), used by SB3 training and evaluation
    go2_mjx_env     same env in JAX/MJX for GPU training with Brax PPO
    numpy_policy    run / export Brax policies with plain NumPy
    runs            run folders, configs and checkpoint lookup
    paths           project paths
"""
__version__ = "0.1.0"
