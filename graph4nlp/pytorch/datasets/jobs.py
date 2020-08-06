from ..data.dataset import Dataset

dataset_root = '../test/dataset/jobs'


class JobsDataset(Dataset):
    def raw_file_names(self) -> list:
        return ['test.txt', 'train.txt', 'vocab.f.txt', 'vocab.q.txt']

    def download(self):
        raise NotImplementedError(
            'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir='../test/dataset/jobs'):
        super(JobsDataset, self).__init__(root_dir)

