import os

from ....utils.config_utils import get_yaml_config

str2yaml = {"stdrnn": "stdrnndecoder.yaml", "stdtree": "stdtreedecoder.yaml"}
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_decoder_args(deocder_name):
    """
        It will build the template for ``DecoderBase`` model.
    Parameters
    ----------
    decoder_name: str
        The decoder name. Expected in ["stdrnn", "stdtree"].

    Returns
    -------
    template_dict: dict
        The template dict.
        The structure is shown as follows:
        {
            rnn_decoder_share: {rnn_type: "lstm", input_size: 300, ...},
            rnn_decoder_private: {max_decoder_step: 50}
        }
        The ``rnn_decoder_share`` contains the parameters shared by all ``DecoderBase`` models.
        The ``rnn_decoder_private`` contains the parameters specifically in each decoder methods.
    """
    if deocder_name in str2yaml.keys():
        yaml_name = str2yaml[deocder_name]
        path = os.path.join(dir_path, yaml_name)
        config = get_yaml_config(path)
        return config
    else:
        return {}


__all__ = ["get_decoder_args"]
