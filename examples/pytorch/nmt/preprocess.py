import json, re, string
from unicodedata import normalize
import argparse
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_raw_dataset_src', default='/home/shiina/shiina/lib/dataset/europarl-v7.fr-en.fr', type=str, help='path to the config file')
    parser.add_argument('--train_raw_dataset_tgt', type=str, default='/home/shiina/shiina/lib/dataset/europarl-v7.fr-en.en', help='rank')
    parser.add_argument('--train_output_dataset', type=str, default='/home/shiina/shiina/lib/dataset/raw/train.json', help="raw data path")

    parser.add_argument('--val_raw_dataset_src', default='/home/shiina/shiina/lib/dataset/test2011/newstest2011.fr',
                        type=str, help='path to the config file')
    parser.add_argument('--val_raw_dataset_tgt', type=str,
                        default='/home/shiina/shiina/lib/dataset/test2011/newstest2011.en', help='rank')
    parser.add_argument('--val_output_dataset', type=str, default='/home/shiina/shiina/lib/dataset/raw/test.json',
                        help="raw data path")

    cfg = parser.parse_args()
    return cfg



class TextProcessor:
    def __init__(self, raw_dataset_src, raw_dataset_tgt, output_dataset):
        self.raw_dataset_src = raw_dataset_src
        self.raw_dataset_tgt = raw_dataset_tgt
        self.output_dataset = output_dataset

    def _preprocess(self, lines):

        # clean a list of lines
        cleaned = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for line in lines:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lower case
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            cleaned.append(' '.join(line))
        return cleaned

    def run(self):
        src_lines = self._read(self.raw_dataset_src)
        tgt_lines = self._read(self.raw_dataset_tgt)
        print(len(src_lines), len(tgt_lines))
        assert len(src_lines) == len(tgt_lines)
        output_content = []
        for src, tgt in zip(src_lines, tgt_lines):
            if src.strip() == "":
                continue
            processed = self._preprocess([src, tgt])
            output_content.append((processed[0], processed[1]))
        with open(self.output_dataset, "w") as f:
            json.dump(output_content, f)
        print("done")

    def _read(self, path):
        with open(path, "r") as f:
            content = f.read()
        sentences = content.strip().split("\n")
        print(sentences[0])
        return sentences

if __name__ == "__main__":
    opt = get_args()
    # process training set
    # processor = TextProcessor(raw_dataset_src=opt.train_raw_dataset_src, raw_dataset_tgt=opt.train_raw_dataset_tgt,
    #                           output_dataset=opt.train_output_dataset)
    # processor.run()
    # process validation set
    processor = TextProcessor(raw_dataset_src=opt.val_raw_dataset_src, raw_dataset_tgt=opt.val_raw_dataset_tgt,
                              output_dataset=opt.val_output_dataset)
    processor.run()




