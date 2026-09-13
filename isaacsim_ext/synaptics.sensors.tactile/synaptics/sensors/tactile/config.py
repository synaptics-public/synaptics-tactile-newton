# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Sensor model profiles and asset resolution.

A model profile is a JSON file under ``data/models/``. Adding a sensor variant
is a data change, not a code change.

Assets (the baked USD and its taxel map) deliberately do **not** live in this
extension: they ship with the ``synaptics_tactile_newton`` package, so the
extension, the Isaac Lab wrapper and the standalone examples all read the same
bytes.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

#: Model profiles, relative to the extension root.
MODEL_CONFIG_DIR = Path("data") / "models"

#: Per-taxel saturation force [N] when a profile does not say.
DEFAULT_FORCE_MAX_N = 100.0

#: Override the asset directory (developer setups running from a source tree).
ASSET_DIR_ENV_VAR = "SYNAPTICS_TACTILE_ASSET_DIR"

#: Keys every model profile must define.
_REQUIRED_KEYS = ("model", "asset", "taxel_map", "sensing_shape_pattern", "force_max_n")


def resolve_extension_root(extension_id: str) -> Path:
    """Root folder of this extension, wherever Kit loaded it from."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    return Path(manager.get_extension_path(extension_id))


def package_root() -> Path:
    """Root folder of this extension, derived from this file's location.

    Works outside Kit (the diagnostics CLI), unlike
    :func:`resolve_extension_root`.
    """
    # <root>/synaptics/sensors/tactile/config.py -> <root>
    return Path(__file__).resolve().parents[3]


def available_models(extension_root: Path | None = None) -> list[str]:
    """Names of every model profile shipped in ``data/models/``."""
    root = Path(extension_root) if extension_root is not None else package_root()
    config_dir = root / MODEL_CONFIG_DIR
    if not config_dir.is_dir():
        return []
    return sorted(path.stem for path in config_dir.glob("*.json"))


def load_model_config(extension_root: Path | None, model_name: str) -> dict:
    """Load and validate ``data/models/<model_name>.json``.

    Raises:
        FileNotFoundError: no such profile.
        ValueError: the profile is malformed.
    """
    root = Path(extension_root) if extension_root is not None else package_root()
    config_path = root / MODEL_CONFIG_DIR / f"{model_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No sensor profile for model '{model_name}' at {config_path}. "
            f"Known models: {', '.join(available_models(root)) or '(none)'}."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object in {config_path}, got {type(config).__name__}."
        )

    missing = [key for key in _REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing required key(s): {', '.join(missing)}.")

    force_max = float(config["force_max_n"])
    if force_max <= 0.0:
        raise ValueError(f"Invalid force_max_n={force_max} in {config_path}; must be > 0.")
    config["force_max_n"] = force_max

    return config


def ensure_core_package_on_path() -> None:
    """Put a checkout's ``synaptics_tactile_newton`` on ``sys.path``.

    When the extension is loaded straight from a repo checkout (``isaacsim_ext/``
    beside ``synaptics_tactile_newton/``), this makes the core package
    importable without a ``--/app/python/extraPaths`` launch flag — the same
    self-contained install other Isaac Sim sensor extensions offer. No-op when
    the package already resolves (e.g. pip-installed into Kit's python, which
    therefore takes precedence). ``find_spec`` only locates the package; nothing
    is imported here.
    """
    if importlib.util.find_spec("synaptics_tactile_newton") is not None:
        return
    repo_root = Path(__file__).resolve().parents[5]
    if (repo_root / "synaptics_tactile_newton" / "__init__.py").is_file():
        sys.path.insert(0, str(repo_root))


def resolve_core_asset_dir() -> Path | None:
    """Directory holding the baked CTS assets, or None when unresolvable.

    Order: the ``SYNAPTICS_TACTILE_ASSET_DIR`` override, then the installed
    ``synaptics_tactile_newton`` package.
    """
    override = os.environ.get(ASSET_DIR_ENV_VAR, "").strip()
    if override:
        path = Path(override)
        return path if path.is_dir() else None

    try:
        import synaptics_tactile_newton
    except ImportError:
        return None

    package_dir = Path(synaptics_tactile_newton.__file__).resolve().parent
    assets = package_dir / "assets"
    return assets if assets.is_dir() else None


def resolve_model_assets(config: dict) -> tuple[Path, Path]:
    """Absolute paths to a model's USD and taxel map.

    Raises:
        FileNotFoundError: the core package (or the asset override) is missing,
            or the named files are not in it.
    """
    asset_dir = resolve_core_asset_dir()
    if asset_dir is None:
        raise FileNotFoundError(
            "Could not locate the Synaptics tactile assets. Install the core "
            "package into Isaac Sim's python "
            "(./python.sh -m pip install --no-deps /path/to/synaptics-tactile-newton) or set "
            f"{ASSET_DIR_ENV_VAR} to a directory holding the baked assets."
        )

    usd_path = asset_dir / str(config["asset"])
    taxel_map_path = asset_dir / str(config["taxel_map"])
    for path in (usd_path, taxel_map_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Asset '{path.name}' for model '{config['model']}' is not in "
                f"{asset_dir}."
            )
    return usd_path, taxel_map_path
