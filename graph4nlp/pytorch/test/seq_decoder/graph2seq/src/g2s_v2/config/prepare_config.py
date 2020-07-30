import os
import re

path = 'ATIS_uni'

all_files = os.listdir(path)

print(all_files)

for file in all_files:
    if file.startswith('ATIS_graph2seq_wo_bilstm'):
        with open(path+'/'+file,'r') as f:
            data = f.read()

        data = data.replace('bignn: True','bignn: False')
        data = data.replace("out_dir: '../../out/ATIS/graph2seq_word300_h300",
                            "out_dir: '../../out/ATIS/uni_graph2seq_word300_h300")

        with open(path+'/'+file,'w') as f:
            f.write(data)
    a = 0

# path = 'GEO_uni'
#
# all_files = os.listdir(path)
#
# print(all_files)
#
# for file in all_files:
#     if file.startswith('GEO_graph2seq_wo_bilstm'):
#         with open(path+'/'+file,'r') as f:
#             data = f.read()
#
#         data = data.replace('bignn: True','bignn: False')
#         data = data.replace("out_dir: '../../out/GEO/graph2seq_word300_h300",
#                             "out_dir: '../../out/GEO/uni_graph2seq_word300_h300")
#
#         with open(path+'/'+file,'w') as f:
#             f.write(data)
#     a = 0

# path = 'JOBS_uni'
#
# all_files = os.listdir(path)
#
# print(all_files)
#
# for file in all_files:
#     if file.startswith('JOBS_graph2seq_wo_bilstm'):
#         with open(path+'/'+file,'r') as f:
#             data = f.read()
#
#         data = data.replace('bignn: True','bignn: False')
#         data = data.replace("out_dir: '../../out/job_dataset/graph2seq_word300_h300",
#                             "out_dir: '../../out/job_dataset/uni_graph2seq_word300_h300")
#
#         with open(path+'/'+file,'w') as f:
#             f.write(data)
#     a = 0

# for sp

# path = 'shortest-path_uni'
#
# all_files = os.listdir(path)
#
# print(all_files)
#
# for file in all_files:
#     if file.startswith('graph2seq_123'):
#         continue
#     with open(path+'/'+file,'r') as f:
#         data = f.read()
#
#     data = data.replace('bignn: True','bignn: False')
#     data = data.replace("out_dir: '../../out/shortest-path/graph2seq_word100_h300",
#                         "out_dir: '../../out/shortest-path/uni_graph2seq_word100_h300")
#
#     with open(path+'/'+file,'w') as f:
#         f.write(data)
#     a = 0

# for babi-19

# path = 'babi-19_uni'
#
# all_files = os.listdir(path)
#
# print(all_files)
#
# for file in all_files:
#     if file.startswith('graph2seq_123'):
#         continue
#     with open(path+'/'+file,'r') as f:
#         data = f.read()
#
#     data = data.replace('bignn: True','bignn: False')
#     data = data.replace("out_dir: '../../out/babi-19/graph2seq_word100_h300",
#                         "out_dir: '../../out/babi-19/uni_graph2seq_word100_h300")
#
#     with open(path+'/'+file,'w') as f:
#         f.write(data)
#     a = 0