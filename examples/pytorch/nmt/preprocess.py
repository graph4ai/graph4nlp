import argparse
import os
import pickle as pkl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_root",
        default="/home/shiina/data/nmt/iwslt14/iwslt14.tokenized.de-en",
        type=str,
        help="path to the config file",
    )
    parser.add_argument(
        "--train_raw_dataset_tgt",
        type=str,
        default="/home/shiina/shiina/lib/dataset/europarl-v7.fr-en.en",
        help="rank",
    )
    parser.add_argument(
        "--output_dir_root", default="examples/pytorch/nmt/data/raw", type=str, help=""
    )

    cfg = parser.parse_args()
    return cfg


def process(source_file_path, target_file_path):
    with open(source_file_path, "r") as f:
        source = f.read().split("\n")
    with open(target_file_path, "r") as f:
        target = f.read().split("\n")
    assert len(source) == len(target)
    output = []
    for s, t in zip(source, target):
        output.append((s, t))
    return output


if __name__ == "__main__":
    opt = get_args()
    train_source_file_path_raw = os.path.join(opt.raw_data_root, "train.de")
    train_target_file_path_raw = os.path.join(opt.raw_data_root, "train.en")
    os.makedirs(opt.output_dir_root, exist_ok=True)
    train_split_path_processed = os.path.join(opt.output_dir_root, "train.pkl")
    train_split = process(train_source_file_path_raw, train_target_file_path_raw)
    with open(train_split_path_processed, "wb") as f:
        pkl.dump(train_split, f)

    val_source_file_path_raw = os.path.join(opt.raw_data_root, "valid.de")
    val_target_file_path_raw = os.path.join(opt.raw_data_root, "valid.en")
    val_split_path_processed = os.path.join(opt.output_dir_root, "val.pkl")
    val_split = process(val_source_file_path_raw, val_target_file_path_raw)
    with open(val_split_path_processed, "wb") as f:
        pkl.dump(val_split, f)

    test_source_file_path_raw = os.path.join(opt.raw_data_root, "test.de")
    test_target_file_path_raw = os.path.join(opt.raw_data_root, "test.en")
    test_split_path_processed = os.path.join(opt.output_dir_root, "test.pkl")
    test_split = process(test_source_file_path_raw, test_target_file_path_raw)
    with open(test_split_path_processed, "wb") as f:
        pkl.dump(test_split, f)
