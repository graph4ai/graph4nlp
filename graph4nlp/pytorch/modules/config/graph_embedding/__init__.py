import os

from ....modules.utils.config_utils import get_yaml_config

str2yaml = {
    "gat": "gat.yaml",
    "gcn": "gcn.yaml",
    "ggnn": "ggnn.yaml",
    "graphsage": "graphsage.yaml",
}

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_graph_embedding_args(graph_embedding_name):
    """
        It will build the template for ``GNNBase`` model.
    Parameters
    ----------
    graph_embedding_name: str
        The graph embedding name. Expected in ["gcn", "gat", "graphsage", "ggnn"].
        If it can't find the ``graph_embedding_name``, it will return ``{}``.
    Returns
    -------
    template_dict: dict
        The template dict.
        The structure is shown as follows:
        {
            graph_embedding_share: {num_layers: 1, input_size: 300, ...},
            graph_embedding_private: {heads: [1], attn_drop: 0.0}
        }
        The ``graph_embedding_share`` contains the parameters shared by all ``GNNBase`` models.
        The ``graph_embedding_private`` contains the parameters specifically in each \
            graph_embedding methods.
    """
    if graph_embedding_name in str2yaml.keys():
        yaml_name = str2yaml[graph_embedding_name]
        path = os.path.join(dir_path, yaml_name)
        config = get_yaml_config(path)
        return config
    else:
        return {}


__all__ = ["get_graph_embedding_args"]
