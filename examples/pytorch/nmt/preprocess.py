import json, re, string
from unicodedata import normalize
import argparse
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_raw_dataset_src', default='/home/shiina/nmt_data/wmt16_de_en/data/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.de', type=str, help='path to the config file')
    parser.add_argument('--train_raw_dataset_tgt', type=str, default='/home/shiina/nmt_data/wmt16_de_en/data/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en', help='rank')
    parser.add_argument('--train_raw_dataset_xliff', default='/home/shiina/nmt_data/wmt16_de_en/data/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.xliff', type=str, help='path to the config file')

    parser.add_argument('--train_output_dataset', type=str, default='/home/shiina/shiina/lib/dataset/news-commentary-v11/de-en/raw/train.json', help="raw data path")


    parser.add_argument('--val_raw_dataset_src', default='/home/shiina/nmt_data/wmt16_de_en/data/dev/dev/newstest2015.en',
                        type=str, help='path to the config file')
    parser.add_argument('--val_raw_dataset_tgt', type=str,
                        default='/home/shiina/nmt_data/wmt16_de_en/data/dev/dev/newstest2015.de', help='rank')
    parser.add_argument('--val_output_dataset', type=str, default='/home/shiina/shiina/lib/dataset/news-commentary-v11/de-en/raw/val.json',
                        help="raw data path")

    parser.add_argument('--test_raw_dataset_src',
                        default='/home/shiina/nmt_data/wmt16_de_en/data/test/test/newstest2016.en',
                        type=str, help='path to the config file')
    parser.add_argument('--test_raw_dataset_tgt', type=str,
                        default='/home/shiina/nmt_data/wmt16_de_en/data/test/test/newstest2016.de', help='rank')
    parser.add_argument('--test_output_dataset', type=str,
                        default='/home/shiina/shiina/lib/dataset/news-commentary-v11/de-en/raw/test.json',
                        help="raw data path")

    cfg = parser.parse_args()
    return cfg



class TextProcessor:
    def __init__(self, output_dataset, raw_dataset_xliff=None, raw_dataset_src=None, raw_dataset_tgt=None):
        self.raw_dataset_src = raw_dataset_src
        self.raw_dataset_tgt = raw_dataset_tgt
        self.raw_dataset_xliff = raw_dataset_xliff
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
        if self.raw_dataset_xliff is not None:
            src_lines, tgt_lines = self._read_xliff(self.raw_dataset_xliff)
        else:
            src_lines = self._read(self.raw_dataset_src)
            tgt_lines = self._read(self.raw_dataset_tgt)
            print("-----", src_lines[0], "ooooo", tgt_lines[0])
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

    def _read_xliff(self, path):
        with open(path, "r") as f:
            content = f.read()
        items = re.findall(r'''<trans-unit id=.*?>(.*?)</trans-unit>''', content, flags=re.S|re.M)
        src_list = []
        tgt_list = []
        for item in items:
            src = re.findall("<source>(.*?)</source>", item, flags=re.S | re.M)[0]
            tgt = re.findall("<target>(.*?)</target>", item, flags=re.S | re.M)[0]
            src = src.strip()
            tgt = tgt.strip()
            src_list.append(tgt)
            tgt_list.append(src)
        print("--------", src_list[0], "oooo", tgt_list[0])
        return src_list, tgt_list

    def _read(self, path):
        with open(path, "r") as f:
            content = f.read()
        sentences = content.strip().split("\n")
        print(sentences[0])
        return sentences

if __name__ == "__main__":
    opt = get_args()
    # process training set
    processor = TextProcessor(raw_dataset_xliff=opt.train_raw_dataset_xliff,
                              output_dataset=opt.train_output_dataset)
    processor.run()
    # process validation set
    processor = TextProcessor(raw_dataset_src=opt.val_raw_dataset_src, raw_dataset_tgt=opt.val_raw_dataset_tgt,
                              output_dataset=opt.val_output_dataset)
    processor.run()

    processor = TextProcessor(raw_dataset_src=opt.test_raw_dataset_src, raw_dataset_tgt=opt.test_raw_dataset_tgt,
                              output_dataset=opt.test_output_dataset)
    processor.run()


