"""Project paths, so scripts work from any working directory (and on Kaggle)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MENAGERIE_GO2 = ROOT / "mujoco_menagerie" / "unitree_go2"   # input (git clone of Menagerie)
MODEL_DIR = ROOT / "models" / "go2_model"                   # output of scripts/build_model.py
GO2_XML = MODEL_DIR / "go2.xml"
SCENE_XML = MODEL_DIR / "scene.xml"
RUNS_DIR = ROOT / "runs"                                    # training runs (one folder per run)
