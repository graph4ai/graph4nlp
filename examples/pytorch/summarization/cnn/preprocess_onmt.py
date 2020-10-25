import json
import nltk

def process_data(split='train'):
    with open('raw/'+split+'_3.json', 'r') as f:
        data = json.load(f)

    src_seqs = []
    tgt_seqs = []
    cnt=0
    for idx, data_item in enumerate(data):
        if idx%500==0:
            print(idx)
        src_seq = ' '.join(data_item['article'])
        src_seq = ' '.join(nltk.word_tokenize(src_seq)).lower()
        tgt_seq = ' '.join([x[0]+' . ' for x in data_item['highlight']])
        tgt_seq = ' '.join(nltk.word_tokenize(tgt_seq)).lower()
        if src_seq.strip()=='' or tgt_seq.strip()=='':
            continue

        if len(src_seq.split())<500:
            cnt+=1

        src_seqs.append(src_seq)
        tgt_seqs.append(tgt_seq)

    print(cnt)

    with open('raw/src-'+split+'.txt', 'w+') as f:
        for line in src_seqs:
            f.write(line+'\n')

    with open('raw/tgt-'+split+'.txt', 'w+') as f:
        for line in tgt_seqs:
            f.write(line+'\n')
    return

process_data('train')
process_data('test')
process_data('val')

"""
>>> from rouge import FilesRouge
>>> 
>>> files_rouge = FilesRouge()
>>> hyp_path = '/data/ghn/graph4nlp/examples/pytorch/summarization/cnn/raw/pred.txt'
>>> ref_path = '/data/ghn/graph4nlp/examples/pytorch/summarization/cnn/raw/tgt-test.txt'
>>> scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
>>> scores
{'rouge-1': {'f': 0.33420923089388826, 'p': 0.3423968264680328, 'r': 0.33961800680705834}, 'rouge-2': {'f': 0.1285818954177987, 'p': 0.13299888658125958, 'r': 0.12963773043518154}, 'rouge-l': {'f': 0.3206686928426199, 'p': 0.3274243335779314, 'r': 0.32403407661185024}}
>>> scores
{'rouge-1': {'f': 0.349101996343003, 'p': 0.3625366633665354, 'r': 0.3496437263226343}, 'rouge-2': {'f': 0.137736319617489, 'p': 0.1440401628983199, 'r': 0.1371908157349791}, 'rouge-l': {'f': 0.33717586192746857, 'p': 0.3487242357024281, 'r': 0.33639927704772404}}
"""

"""
bm=5, step=10000, wocopy
{'rouge-1': {'f': 0.24609845288572946, 'p': 0.24995781547605841, 'r': 0.2550254593940543}, 'rouge-2': {'f': 0.07913915878666514, 'p': 0.08093749170888129, 'r': 0.08113200497408996}, 'rouge-l': {'f': 0.262401752694348, 'p': 0.34459940592334054, 'r': 0.22382076103634743}}

bm=5, step=20000, wocopy
{'rouge-1': {'f': 0.2892025270562292, 'p': 0.29688728335467834, 'r': 0.29401705461132155}, 'rouge-2': {'f': 0.10829242165770399, 'p': 0.1116103244812709, 'r': 0.1095164620514086}, 'rouge-l': {'f': 0.29990236484608745, 'p': 0.36684135649462535, 'r': 0.2672166048773799}}
"""

# onmt_preprocess -train_src raw/src-train.txt -train_tgt raw/tgt-train.txt -valid_src raw/src-val.txt -valid_tgt raw/tgt-val.txt -save_data raw/onmt_data --share_vocab -dynamic_dict -overwrite -src_seq_length 500 -src_seq_length_trunc 500 -tgt_seq_length 100 -tgt_seq_length_trunc 100
# onmt_preprocess -train_src raw/src-val.txt -train_tgt raw/tgt-val.txt -valid_src raw/src-val.txt -valid_tgt raw/tgt-val.txt -save_data raw/demo --num_threads 20 --share_vocab --report_every 500
# CUDA_VISIBLE_DEVICES=1
# screen -S ghn_code1 onmt_train -data raw/onmt_data -save_model raw/onmt_data_model --word_vec_size 128 --share_embeddings --encoder_type brnn --rnn_size 512 --global_attention general --copy_attn --coverage_attn --learning_rate 0.15 --adagrad_accumulator_init 0.1 --seed 777 --optim adagrad --batch_size 32 --valid_batch_size 32 --gpu_ranks 0 --log_file raw/log_onmt1.txt
# screen -S ghn_code2 onmt_translate -batch_size 20 -beam_size 1 -min_length 35 -stepwise_penalty -beta 5 -length_penalty wu -alpha 0.9 -ignore_when_blocking "." -model raw/onmt_data_model_wocopy_step_10000.pt -src raw/src-test.txt -output raw/pred.txt -replace_unk -verbose
# screen -S ghn_code2 onmt_translate -batch_size 20 -beam_size 5 -min_length 35 -stepwise_penalty -beta 5 -length_penalty wu -alpha 0.9 -ignore_when_blocking "." -model raw/onmt_data_model_wocopy_step_20000.pt -src raw/src-test.txt -output raw/pred_b5.txt -replace_unk -verbose