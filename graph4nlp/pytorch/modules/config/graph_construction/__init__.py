import os

from ....modules.utils.config_utils import get_yaml_config

str2yaml = {
    "dependency": "dependency.yaml",
    "constituency": "constituency.yaml",
    "ie": "ie_graph_construction.yaml",
    "node_emb": "node_emb.yaml",
    "node_emb_refined": "node_emb_refine.yaml",
}


dir_path = os.path.dirname(os.path.realpath(__file__))


def get_graph_construction_args(graph_construction_name):
    """
        It will build the template for ``GraphConstructionBase`` model.
    Parameters
    ----------
    graph_construction_name: str
        The graph construction method name. Expected in ["dependency", "constituency", "ie", \
            "node_emb", "node_emb_refined"].
        If it can't find the ``graph_construction_name``, it will return ``{}``.
    Returns
    -------
    template_dict: dict
        The template dict.
        The structure is shown as follows:
        {
            graph_construction_share: {graph_type: 'constituency', root_dir: null, ...},
            graph_construction_private: {merge_strategy: 'tailhead', sequential_link: True},
            node_embedding: {
                                input_size: 300,
                                ...,
                                embedding_style: {single_token_item: True, ...}
                            }
        }
        The ``graph_construction_share`` contains the parameters shared by all \
            ``GraphConstructionBase`` models.
        The ``graph_construction_private`` contains the parameters specifically \
            in each graph_construction methods.
        The ``node_embedding`` contains all the parameters for node embedding.
            Specifically, it contains ``embedding_style`` which for embedding style.
    """
    if graph_construction_name in str2yaml.keys():
        yaml_name = str2yaml[graph_construction_name]
        path = os.path.join(dir_path, yaml_name)
        config = get_yaml_config(path)
        return config
    else:
        return {}


__all__ = ["get_graph_construction_args"]
