import json
#
# train_data_file = 'raw/train_3w.json'
# val_data_file = 'raw/val.json'
# test_data_file = 'raw/test.json'
#
# with open(train_data_file, 'r') as f:
#     data = json.load(f)
#
# part = 'train'

for part in ['train', 'val', 'test']:
    if part == 'train':
        with open('raw/{}_3w.json'.format(part), 'r') as f:
            data = json.load(f)
    else:
        with open('raw/{}.json'.format(part), 'r') as f:
            data = json.load(f)

    inputs = []
    outputs = []
    for data_item in data:
        input = ' '.join(' '.join(data_item['article']).split()[:400]).lower()
        output = ' '.join(
            ' '.join(['<t> ' + sent[0] + ' . </t>' for sent in data_item['highlight']]).split()[:99]).lower()
        inputs.append(input)
        outputs.append(output)

    with open('fairseq_cnn/{}.input'.format(part), 'w+') as f:
        for line in inputs:
            f.write(line+'\n')

    with open('fairseq_cnn/{}.output'.format(part), 'w+') as f:
        for line in outputs:
            f.write(line+'\n')

a = 0