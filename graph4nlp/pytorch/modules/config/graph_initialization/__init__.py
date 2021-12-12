import os

from ....modules.utils.config_utils import get_yaml_config


def get_graph_initialization_args():
    """It will build the template for ``GraphConstructionBase`` model.

    Returns
    -------
    template_dict: dict
        The template dict.
        The structure is shown as follows:
        {
            input_size: 300,
            ...,
            embedding_style: {single_token_item: True, ...}
        }
        The ``node_embedding`` contains all the parameters for node embedding.
            Specifically, it contains ``embedding_style`` which for embedding style.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, "default.yaml")
    config = get_yaml_config(path)
    return config


__all__ = ["get_graph_initialization_args"]
