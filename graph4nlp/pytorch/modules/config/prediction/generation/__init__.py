from ....utils.config_utils import get_yaml_config
import os

str2yaml = {"stdrnn": "stdrnndecoder.yaml"}


def get_decoder_args(deocder_name):
    if deocder_name in str2yaml.keys():
        yaml_name = str2yaml[deocder_name]
        path = os.path.join("graph4nlp/pytorch/modules/config/prediction/generation", yaml_name)
        config = get_yaml_config(path)
        return config
    else:
        return {}


__all__ = ["get_decoder_args"]
