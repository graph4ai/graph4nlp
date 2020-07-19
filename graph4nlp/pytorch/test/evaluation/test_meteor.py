from graph4nlp.pytorch.modules.evaluation.meteor import METEOR

if __name__ == "__main__":
    import json

    scorer = METEOR()
    pred_file_path = "/home/shiina/shiina/question/iq/pred.json"
    gt_file_path = "/home/shiina/shiina/question/iq/gt.json"
    with open(gt_file_path, "r") as f:
        gt = json.load(f)
        print(gt[0])
        gts = []
        for i in gt:
            for j in i:
                gts.append(str(j))
    with open(pred_file_path, "r") as f:
        pred = json.load(f)
        print(pred[1])
        preds = []
        for i in pred:
            for j in i:
                preds.append(str(j))
    print(len(gts), len(preds))
    score, scores = scorer.calculate_scores(gts, preds)
    print(score)
    print(len(scores))
