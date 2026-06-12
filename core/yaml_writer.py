"""Atomic YAML writer for configuration.yaml modifications via REST API."""

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_DEFAULT_CONFIG_PATH = "configuration.yaml"


def _safe_dump(data: Dict[str, Any]) -> str:
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def read_yaml(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Read and parse the configuration YAML file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(data: Dict[str, Any], config_path: str = _DEFAULT_CONFIG_PATH) -> None:
    """Atomically write configuration YAML (temp file → os.replace)."""
    p = Path(config_path)
    content = _safe_dump(data)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(p))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def update_yaml_camera(camera_cfg: Dict[str, Any], config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Add or update a camera entry by id. Returns the full updated dict."""
    data = read_yaml(config_path)
    cameras = data.setdefault("cameras", [])

    for i, cam in enumerate(cameras):
        if isinstance(cam, dict) and cam.get("id") == camera_cfg["id"]:
            cameras[i] = {**cam, **camera_cfg}
            break
    else:
        cameras.append(camera_cfg)

    write_yaml(data, config_path)
    return data


def remove_yaml_camera(camera_id: str, config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Remove a camera entry by id. Returns the full updated dict."""
    data = read_yaml(config_path)
    cameras = data.get("cameras", [])
    data["cameras"] = [c for c in cameras if not (isinstance(c, dict) and c.get("id") == camera_id)]
    write_yaml(data, config_path)
    return data


def update_yaml_space(space_cfg: Dict[str, Any], config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Add or update a space entry by id. Returns the full updated dict."""
    data = read_yaml(config_path)
    spaces = data.setdefault("spaces", [])

    for i, sp in enumerate(spaces):
        if isinstance(sp, dict) and sp.get("id") == space_cfg["id"]:
            spaces[i] = {**sp, **space_cfg}
            break
    else:
        spaces.append(space_cfg)

    write_yaml(data, config_path)
    return data


def remove_yaml_space(space_id: str, config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Remove a space entry by id. Returns the full updated dict."""
    data = read_yaml(config_path)
    spaces = data.get("spaces", [])
    data["spaces"] = [s for s in spaces if not (isinstance(s, dict) and s.get("id") == space_id)]
    write_yaml(data, config_path)
    return data


def update_yaml_config_section(section: str, values: Dict[str, Any], config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Update a top-level config section (thresholds, yolo, llm). Returns the full updated dict."""
    data = read_yaml(config_path)
    existing = data.setdefault(section, {})
    existing.update(values)
    write_yaml(data, config_path)
    return data
