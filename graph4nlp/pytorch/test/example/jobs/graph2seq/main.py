from graph4nlp.pytorch.datasets.jobs import JobsDataset


class Jobs:
    def __init__(self):
        super(Jobs, self).__init__()
        self._build_dataloader()


    def _build_dataloader(self):
        self.train_loader = JobsDataset(root_dir="../../../dataset/jobs")

    def train(self):
        pass


if __name__ == "__main__":
    runner = Jobs()
    runner.train()
