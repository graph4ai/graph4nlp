import sys
import chardet

# fin = open('raw/train.tsv', 'r')
# fout = open('raw/train.txt', 'w')

fin = open('raw/test.tsv', 'r')
fout = open('raw/test.txt', 'w')

for line in fin:
    data = line.strip().split()
    text = ' '.join(data[:-1])
    label = data[-1]
    fout.write('{}\t{}\n'.format(text, label))
