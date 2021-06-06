import yaml


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
