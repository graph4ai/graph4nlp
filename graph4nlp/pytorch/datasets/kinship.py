from graph4nlp.pytorch.data.dataset import KGCompletionDataset


class KinshipDataset(KGCompletionDataset):
    @property
    def raw_file_names(self) -> dict:
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'e1rel_to_e2_ranking_train.json',
                'val': 'e1rel_to_e2_ranking_dev.json',
                'test': 'e1rel_to_e2_ranking_test.json'}

    @property
    def processed_file_names(self) -> dict:
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'KG_graph': 'KG_graph.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self,
                 root_dir,
                 topology_builder=None,
                 topology_subdir=None,
                 edge_strategy=None,
                 pretrained_word_emb_name=None,
                 pretrained_word_emb_url=None,
                 target_pretrained_word_emb_name=None,
                 target_pretrained_word_emb_url=None,
                 pretrained_word_emb_cache_dir=".vector_cache/",
                 word_emb_size=300,
                 share_vocab=True,  # share vocab between entity and relation
                 **kwargs):
        self.split_token = ' '
        super(KinshipDataset, self).__init__(root_dir=root_dir,
                                             word_emb_size=word_emb_size,
                                             share_vocab=share_vocab,
                                             topology_builder=topology_builder,
                                             topology_subdir=topology_subdir,
                                             edge_strategy=edge_strategy,
                                             pretrained_word_emb_name=pretrained_word_emb_name,
                                             pretrained_word_emb_url=pretrained_word_emb_url,
                                             target_pretrained_word_emb_name=target_pretrained_word_emb_name,
                                             target_pretrained_word_emb_url=target_pretrained_word_emb_url,
                                             pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
                                             **kwargs)


