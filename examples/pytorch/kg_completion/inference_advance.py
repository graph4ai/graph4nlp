import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.kinship import KinshipDataset
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

import argparse
from main import KGC


def ranking_and_hits_this(cfg, model, dev_rank_batcher, vocab, name, kg_graph=None):
    print("")
    print("-" * 50)
    print(name)
    print("-" * 50)
    print("")
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for _ in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var["e1_tensor"].to(cfg["env_args"]["device"])
        e2 = str2var["e2_tensor"].to(cfg["env_args"]["device"])
        rel = str2var["rel_tensor"].to(cfg["env_args"]["device"])
        rel_reverse = str2var["rel_eval_tensor"].to(cfg["env_args"]["device"])
        e2_multi1 = str2var["e2_multi1"].float().to(cfg["env_args"]["device"])
        e2_multi2 = str2var["e2_multi2"].float().to(cfg["env_args"]["device"])

        pred1 = model(e1, rel, kg_graph)
        pred2 = model(e2, rel_reverse, kg_graph)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(e1.shape[0]):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            # save the prediction that is relevant
            target_value1 = pred1[i, e2[i, 0].item()].item()
            target_value2 = pred2[i, e1[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(e1.shape[0]):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i] == e2[i, 0].item())[0][0]
            rank2 = np.where(argsort2[i] == e1[i, 0].item())[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        # dev_rank_batcher.state.loss = [0]

    for i in range(10):
        print("Hits left @{0}: {1}".format(i + 1, np.mean(hits_left[i])))
        print("Hits right @{0}: {1}".format(i + 1, np.mean(hits_right[i])))
        print("Hits @{0}: {1}".format(i + 1, np.mean(hits[i])))
    print("Mean rank left: {0}".format(np.mean(ranks_left)))
    print("Mean rank right: {0}".format(np.mean(ranks_right)))
    print("Mean rank: {0}".format(np.mean(ranks)))
    print("Mean reciprocal rank left: {0}".format(np.mean(1.0 / np.array(ranks_left))))
    print("Mean reciprocal rank right: {0}".format(np.mean(1.0 / np.array(ranks_right))))
    print("Mean reciprocal rank: {0}".format(np.mean(1.0 / np.array(ranks))))

    return np.mean(1.0 / np.array(ranks))


def main(cfg, model_path):
    np.random.seed(cfg["env_args"]["seed"])
    torch.manual_seed(cfg["env_args"]["seed"])

    if not cfg["env_args"]["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        cfg["env_args"]["device"] = torch.device(
            "cuda" if cfg["env_args"]["gpu"] < 0 else "cuda:%d" % cfg["env_args"]["gpu"]
        )
        cudnn.benchmark = True
        torch.cuda.manual_seed(cfg["env_args"]["seed"])
    else:
        cfg["env_args"]["device"] = torch.device("cpu")

    print("\n" + cfg["checkpoint_args"]["out_dir"])

    dataset = KinshipDataset(
        root_dir="examples/pytorch/kg_completion/data/{}".format(
            cfg["preprocessing_args"]["dataset"]
        ),
        topology_subdir="kgc",
    )

    test_dataloader = DataLoader(
        dataset.test,
        batch_size=cfg["training_args"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    num_entities = len(dataset.vocab_model.in_word_vocab)
    num_relations = len(dataset.vocab_model.out_word_vocab)

    graph_path = "examples/pytorch/kg_completion/data/{}/processed/kgc/" "KG_graph.pt".format(
        cfg["preprocessing_args"]["dataset"]
    )
    KG_graph = torch.load(graph_path).to(cfg["env_args"]["device"])
    model = KGC(cfg, num_entities, num_relations).to(cfg["env_args"]["device"])

    model_params = torch.load(model_path)
    # total_param_size = []
    # params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
    # for key, size, count in params:
    #     total_param_size.append(count)
    #     print(key, size, count)
    # print(np.array(total_param_size).sum())
    model.load_state_dict(model_params)
    model.eval()
    ranking_and_hits_this(
        cfg, model, test_dataloader, dataset.vocab_model, "test_evaluation", kg_graph=KG_graph
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json_config",
        "--json_config",
        required=True,
        type=str,
        help="path to the json config file",
    )
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->  {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)

    model_name = "{0}_{1}_{2}_{3}".format(
        config["model_args"]["model"],
        config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["direction_option"],
        config["model_args"]["input_drop"],
        config["model_args"]["hidden_drop"],
    )
    model_path = "examples/pytorch/kg_completion/saved_models/{0}_{1}.model".format(
        config["preprocessing_args"]["dataset"], model_name
    )

    main(config, model_path)
