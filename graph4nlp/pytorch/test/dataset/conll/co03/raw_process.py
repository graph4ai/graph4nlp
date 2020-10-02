'''
transfer the raw conll03 data to the required format
'''

import os
import string
punc=string.punctuation

def preprocess(input_file, output_file):    
    input_sentence=[]
    output_tags=[]
    with open(input_file) as f:
        sent=''
        tag=''    
        for line in f:
            line=line.strip()
            if len(line)>0:
                if line[0]=='.':
                    sent=sent+line.split(' ')[0]
                    tag=tag+' '+line.split(' ')[-1]            
                    input_sentence.append(sent)
                    output_tags.append(tag)
                    sent=''
                    tag=''
                else:
                   if line[0]!='-':
                        if line[0] in punc and line[0]!='(':
                            sent=sent+line.split(' ')[0]
                            tag=tag+' '+line.split(' ')[-1] 
                        else:
                            sent=sent+' '+line.split(' ')[0]
                            tag=tag+' '+line.split(' ')[-1]
    #if os.path.exist(output_file):
      #     os.remove(output_file)
    file_write=open(output_file,'w')
    for sent_idx in range(len(input_sentence)):
               file_write.writelines(input_sentence[sent_idx]+'\t'+output_tags[sent_idx])
               file_write.writelines('\n')
 
preprocess('eng.testa','sentence_tags_dev.txt')    
preprocess('eng.testb','sentence_tags_test.txt')
preprocess('eng.train','sentence_tags_train.txt')
#raw_path='F:/xiaojie/graph4nlp/graph4nlp/pytorch/test/dataset/conll/raw/'       
#tags=[]
#files=['sentence_tags_dev.txt','sentence_tags_test.txt','sentence_tags_train.txt']
#for file in files:
#    with open(raw_path+file) as f:
#        for line in f:
#           tag=line.strip().split('--->')[1].split(' ')
#           tags.extend(tag)
#           tags=list(set(tags))
#tags=['I-PER', 'O', 'B-ORG', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'B-MISC']

