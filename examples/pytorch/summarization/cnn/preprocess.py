import os
import json
import random

def process_stories(root_dir):
    # root_dir = 'raw/stories/'
    stories = os.listdir(root_dir)
    examples = []
    cnt_words = []
    for f_name in stories:
        with open(root_dir+f_name, 'r') as f:
            text = f.readlines()
            # print(len(' '.join(text).split(' ')))
            cnt_words.append(len(' '.join(text).split(' ')))
            example_dict = {'article':[],
                            'highlight':[]}
            is_hightlight = False
            for line in text:
                if line == '\n':
                    continue
                if line == '@highlight\n':
                    is_hightlight = True
                    continue
                if is_hightlight:
                    example_dict['highlight'].append([line.strip()])
                    is_hightlight = False
                else:
                    example_dict['article'].append(line.strip())

            examples.append(example_dict)

    print(sum(cnt_words)/len(cnt_words))
    return examples

example1 = process_stories('/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/raw/cnn/stories/')
example2 = process_stories('/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/raw/dm_stories_tokenized/')

examples = example1 + example2
random.shuffle(examples)

train_3w = examples[:30000]
train_9w = examples[:90000]

with open('raw/train_3w.json', 'w+') as f:
    json.dump(train_3w, f, indent=1)

with open('raw/train_9w.json', 'w+') as f:
    json.dump(train_9w, f, indent=1)

with open('raw/val.json', 'w+') as f:
    json.dump(examples[-10000:-5000], f, indent=1)

with open('raw/test.json', 'w+') as f:
    json.dump(examples[-5000:], f, indent=1)

# print('num_skip_long_input='+str(num_skip_long_input))
a = 0