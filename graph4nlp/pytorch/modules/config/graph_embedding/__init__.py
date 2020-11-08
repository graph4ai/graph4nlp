from ....modules.utils.config_utils import get_yaml_config
import os

str2yaml = {"gat": "gat.yaml", "gcn": "gcn.yaml", "ggnn": "ggnn.yaml", "graphsage": "graphsage.yaml"}


def get_graph_embedding_args(graph_embedding_name):
    if graph_embedding_name in str2yaml.keys():
        yaml_name = str2yaml[graph_embedding_name]
        path = os.path.join("graph4nlp/pytorch/modules/config/graph_embedding", yaml_name)
        config = get_yaml_config(path)
        return config
    else:
        return {}


__all__ = ["get_graph_embedding_args"]
