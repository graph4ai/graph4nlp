import json
import os
import random

random.seed(10)


def process_stories(root_dir):
    stories = os.listdir(root_dir)
    examples = []
    cnt_words = []
    for i, f_name in enumerate(stories):
        if i % 100 == 0:
            print(i)
        with open(root_dir + f_name, "r") as f:
            text = f.readlines()
            # examples.append(text)
            cnt_words.append(len(" ".join(text).split(" ")))
            example_dict = {"article": [], "highlight": []}
            is_hightlight = False
            for line in text:
                if line == "\n":
                    continue
                if line == "@highlight\n":
                    is_hightlight = True
                    continue
                if is_hightlight:
                    example_dict["highlight"].append([line.strip()])
                    is_hightlight = False
                else:
                    example_dict["article"].append(line.strip())

            examples.append(example_dict)

    # print(sum(cnt_words)/len(cnt_words))
    return examples


example1 = process_stories("raw/cnn_stories_tokenized/")
# example2 = process_stories('raw/dm_stories_tokenized/')

# examples = example1 + example2
examples = example1
random.shuffle(examples)

# train_30 = examples[:30]
# train_3k = examples[:3000]
train_1w = examples[:10000]
train_3w = examples[:30000]
# train_9w = examples[:90000]
val = examples[-6000:-3000]
test = examples[-3000:]

# with open('raw/train_30.json', 'w+') as f:
#     json.dump(train_30, f, indent=1)

# with open('raw/train_3k.json', 'w+') as f:
#     json.dump(train_3k, f, indent=1)

with open("raw/train_1w.json", "w+") as f:
    json.dump(train_1w, f, indent=1)

with open("raw/train_3w.json", "w+") as f:
    json.dump(train_3w, f, indent=1)

# with open('raw/train_9w.json', 'w+') as f:
#     json.dump(train_9w, f, indent=1)

with open("raw/val.json", "w+") as f:
    json.dump(val, f, indent=1)

with open("raw/test.json", "w+") as f:
    json.dump(test, f, indent=1)

# print('num_skip_long_input='+str(num_skip_long_input))
a = 0
