import json
import os
from pathlib import Path
from typing import List
import yaml
from omegaconf import OmegaConf


def load_json_config(path: str):
    with open(path, "r") as f:
        data = json.load(f)
        config = load_yaml_config(data["config_path"])
        data.pop("config_path")

        dot_list = []
        for k, v in data.items():
            dot_list.append(f"{k}={v}")

        updated_config = OmegaConf.from_dotlist(dot_list)
        merged_config = OmegaConf.merge(config, updated_config)
        return merged_config


def load_yaml_config(
    path: str, included_paths: List[str] = None, nesting_level: int = 0, max_nesting_level: int = 20
):
    if included_paths is None:
        included_paths = []

    if nesting_level > max_nesting_level:
        raise RuntimeError(f"Exceeds maximial nesting level {max_nesting_level}!")

    # Parse yaml path
    path = parse_config_path(path)

    config = OmegaConf.load(path)
    included_configs = []
    for each in config.get("includes", []):
        if os.path.abspath(each) in included_paths:
            raise RuntimeError("Circular includes of yaml files are not supported!")

        included_configs.append(
            load_yaml_config(each, included_paths + [os.path.abspath(path)], nesting_level + 1)
        )

    merged_config = OmegaConf.merge(*included_configs, config)
    merged_config.pop("includes", None)
    return merged_config


def parse_config_path(path: str):
    """Parse config path by replacing '$/' with the library directory."""
    if path.startswith("$/"):
        library_dir = Path(os.path.realpath(__file__)).parent.parent.parent
        return str(os.path.join(library_dir, path[2:]))
    else:
        return path


def update_values(to_args: dict, from_args_list: [dict]):
    """
        update_values(template, [args1, args2])
    Parameters
    ----------
    to_args
    from_args_list

    Returns
    -------

    """
    for from_args in from_args_list:
        if not isinstance(from_args, dict):
            raise TypeError("The element in ``from_args_list`` should be dict")
        update_values_api(dict_from=from_args, dict_to=to_args)


def update_values_api(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict) and key in dict_to.keys():
            update_values_api(dict_from[key], dict_to[key])
        else:
            dict_to[key] = dict_from[key]


def get_yaml_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.safe_load(setting)
    return config
