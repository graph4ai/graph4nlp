from graph4nlp.pytorch.data.dataset import KGCompletionDataset


class KinshipDataset(KGCompletionDataset):
    def __init__(self, root_dir=None, topology_subdir="kgc", word_emb_size=300, **kwargs):
        super(KinshipDataset, self).__init__(
            root_dir, topology_subdir=topology_subdir, word_emb_size=word_emb_size, **kwargs
        )

    @property
    def raw_file_names(self):
        """
        3 reserved keys: 'train', 'val' (optional), 'test'.
        Represent the split of dataset.
        """
        return {
            "train": "e1rel_to_e2_train.json",
            "val": "e1rel_to_e2_ranking_dev.json",
            "test": "e1rel_to_e2_ranking_test.json",
        }

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab' and 'data'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    @property
    def raw_dir(self) -> str:
        """The directory where the raw data is stored."""
        return self.root

    def download(self):
        return


if __name__ == "__main__":
    dataset = KinshipDataset(
        root_dir="examples/pytorch/kg_completion/data/kinship", topology_subdir="kgc"
    )
