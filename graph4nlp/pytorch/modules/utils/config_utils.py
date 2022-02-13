import json
import os
from typing import Set
import yaml
from omegaconf import OmegaConf

from .generic_utils import get_library_dir_path


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
        return OmegaConf.to_container(merged_config, resolve=True)


def load_yaml_config(
    path: str, included_paths: Set[str] = None, nesting_level: int = 0, max_nesting_level: int = 20
):
    if included_paths is None:
        included_paths = set()

    if nesting_level > max_nesting_level:
        raise RuntimeError(f"Exceeds maximial nesting level {max_nesting_level}!")

    config = OmegaConf.load(path)
    included_config_paths = [parse_config_path(path) for path in config.get("includes", [])]

    library_dir = get_library_dir_path()
    # Add default base yamls if not imported
    base_config_path = os.path.join(library_dir, "configs/defaults.yaml")
    if base_config_path not in included_config_paths and base_config_path not in included_paths:
        included_config_paths.append(base_config_path)

    # Choose which defaults to load based on user config input
    if "model_args" in config:
        config_paths_to_remove = set()
        config_paths_to_add = []
        if "graph_construction_name" in config.model_args:
            config_path = os.path.join(
                library_dir,
                f"configs/graph_construction/{config.model_args.graph_construction_name}/defaults.yaml",  # noqa
            )
            if os.path.exists(config_path):
                # Load default config file matching the provided name
                config_paths_to_add.append(config_path)

                # Remove the included config paths which are in conflict
                for each in included_config_paths:
                    if each.startswith(os.path.join(library_dir, "configs/graph_construction")):
                        config_paths_to_remove.add(each)
                        if each != config_path:
                            print(
                                f"[Warning] the imported graph construction yaml file '{each}' will be overwritten by '{config_path}' which is specified by the provided graph_construction_name '{config.model_args.graph_construction_name}'"  # noqa
                            )

        if "graph_initialization_name" in config.model_args:
            config_path = os.path.join(
                library_dir,
                f"configs/graph_initialization/{config.model_args.graph_initialization_name}.yaml",
            )
            if os.path.exists(config_path):
                # Load default config file matching the provided name
                config_paths_to_add.append(config_path)

                # Remove the included config paths which are in conflict
                for each in included_config_paths:
                    if each.startswith(os.path.join(library_dir, "configs/graph_initialization")):
                        config_paths_to_remove.add(each)
                        if each != config_path:
                            print(
                                f"[Warning] the imported graph initialization yaml file '{each}' will be overwritten by '{config_path}' which is specified by the provided graph_initialization_name '{config.model_args.graph_initialization_name}'"  # noqa
                            )

        if "graph_embedding_name" in config.model_args:
            config_path = os.path.join(
                library_dir,
                f"configs/graph_embedding/{config.model_args.graph_embedding_name}/defaults.yaml",
            )
            if os.path.exists(config_path):
                # Load default config file matching the provided name
                config_paths_to_add.append(config_path)

                # Remove the included config paths which are in conflict
                for each in included_config_paths:
                    if each.startswith(os.path.join(library_dir, "configs/graph_embedding")):
                        config_paths_to_remove.add(each)
                        if each != config_path:
                            print(
                                f"[Warning] the imported graph embedding yaml file '{each}' will be overwritten by '{config_path}' which is specified by the provided graph_embedding_name '{config.model_args.graph_embedding_name}'"  # noqa
                            )

        if "decoder_name" in config.model_args:
            config_path = os.path.join(
                library_dir, f"configs/prediction/generation/{config.model_args.decoder_name}.yaml"
            )
            if os.path.exists(config_path):
                # Load default config file matching the provided name
                config_paths_to_add.append(config_path)

                # Remove the included config paths which are in conflict
                for each in included_config_paths:
                    if each.startswith(os.path.join(library_dir, "configs/prediction/generation")):
                        config_paths_to_remove.add(each)
                        if each != config_path:
                            print(
                                f"[Warning] the imported decoder yaml file '{each}' will be overwritten by '{config_path}' which is specified by the provided decoder_name '{config.model_args.decoder_name}'"  # noqa
                            )

        included_config_paths = [
            each for each in included_config_paths if each not in config_paths_to_remove
        ]
        included_config_paths += config_paths_to_add

    included_configs = []
    for each in included_config_paths:
        if each in included_paths:
            raise RuntimeError("Circular includes of yaml files are not supported!")

        included_configs.append(
            load_yaml_config(each, included_paths.union([os.path.abspath(path)]), nesting_level + 1)
        )

    merged_config = OmegaConf.merge(*included_configs, config)
    merged_config.pop("includes", None)
    return merged_config


def parse_config_path(path: str):
    """Parse config path by replacing '$/' with the library directory if exists."""
    if path.startswith("$/"):
        library_dir = get_library_dir_path()
        return os.path.join(library_dir, path[2:])
    else:
        return os.path.abspath(path)


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
