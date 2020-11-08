from ....modules.utils.config_utils import get_yaml_config
import os

str2yaml = {"dependency": "dependency.yaml", "constituency": "constituency.yaml", "ie": "ie_graph_construction.yaml",
            "node_emb": "node_emb.yaml", "node_emb_refined": "node_emb_refine.yaml"}


def get_graph_construction_args(graph_construction_name):
    if graph_construction_name in str2yaml.keys():
        yaml_name = str2yaml[graph_construction_name]
        path = os.path.join("graph4nlp/pytorch/modules/config/graph_construction", yaml_name)
        config = get_yaml_config(path)
        return config
    else:
        return {}


__all__ = ["get_graph_construction_args"]
