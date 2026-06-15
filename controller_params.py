"""
Load controller parameters from controller_params.yaml (Hydra-style).
Falls back to built-in defaults if the file is missing or a key is absent.
Requires PyYAML for YAML loading; if unavailable, only defaults are used.
"""

import os
from copy import deepcopy
from typing import Any

# Built-in defaults (match original controller_solvers / mujoco_viewer values)
DEFAULTS: dict[str, Any] = {
    "max_joints": 6,
    "period_ms": 4,
    "max_vel_cmd_deg_s": 720.0,
    "pid": {
        "kp": 0.05,
        "kd": 0.01,
        "ki": 0.05,
        "integral_clamp": 5.0,
        "u_max_per_step": 80.0,
    },
    "invdyn": {
        "use_pd_steps": True,
        "kp": 144.0,
        "kd": 24.0,
        "qdd_max_deg": 150.0,
        "kp_pd": 0.5,
        "kd_pd": 0.1,
        "k_grav_comp": 0.02,
    },
    "pd_velff": {
        "kp": 144.0 / 90.0,
        "kd": 24.0 / 90.0,
        "u_max_per_step": 80.0,
    },
    "mpc": {
        "horizon": 20,
        "q": 10.0,
        "r": 0.1,
        "q_terminal": 50.0,
        "rho": 1.0,
        "u_max_per_step": 80.0,
    },
    "virtual_driver": {
        "kp_v": 80.0,
        "ki_v": 0.0,
        "kd_v": 8.0,
        "integral_clamp_v": 2.0,
        "tau_smooth_alpha": 0.0,
        "torque_limit_nm": 80.0,
        "vel_cmd_limit_deg_s": 720.0,
        "qacc_limit_rad_s2": 1500.0,
    },
    "motor_ctrl_nm": 80.0,
    "rest_pose_deg": [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0],
}

_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH: str | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins; base is not mutated."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def get_controller_params(config_path: str | None = None) -> dict[str, Any]:
    """Load controller_params.yaml and merge with defaults. Cached per path."""
    global _CONFIG, _CONFIG_PATH
    path = config_path
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_params.yaml")
    if _CONFIG is not None and _CONFIG_PATH == path:
        return _CONFIG
    try:
        import yaml
        if os.path.isfile(path):
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            _CONFIG = _deep_merge(DEFAULTS, loaded)
        else:
            _CONFIG = deepcopy(DEFAULTS)
    except Exception:
        _CONFIG = deepcopy(DEFAULTS)
    _CONFIG_PATH = path
    return _CONFIG


def set_controller_params(config: dict[str, Any] | None) -> None:
    """Inject config (e.g. for tests). None resets to unloaded state."""
    global _CONFIG, _CONFIG_PATH
    _CONFIG = deepcopy(config) if config is not None else None
    _CONFIG_PATH = None
