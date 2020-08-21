import os
import json

root_dir = 'raw/stories/'
stories = os.listdir('raw/stories/')
examples = []

num_skip_long_input = 0

for f_name in stories:
    with open(root_dir+f_name, 'r') as f:
        text = f.readlines()
        print(len(' '.join(text).split(' ')))
        if len(' '.join(text).split(' ')) > 900:
            num_skip_long_input = num_skip_long_input + 1
            continue
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

with open('raw/train.json', 'w+') as f:
    json.dump(examples[:-1], f, indent=1)

with open('raw/test.json', 'w+') as f:
    json.dump(examples[-1:], f, indent=1)

print('num_skip_long_input='+str(num_skip_long_input))
a = 0