import pickle as pkl
import re
from unidecode import unidecode

rplc_list = pkl.load(open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev_rplc_list.pkl', 'rb'))
lexs = open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev-webnlg-all-delex.lex','r').readlines()
triples = open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev-webnlg-all-delex.triple','r').readlines()
triple2lex = {}

new_triples = []
for triple_str in triples:
    triple_list = triple_str.split(' < TSP > ')
    new_triple_list = []
    for triple in triple_list:
        s, p, o = triple.strip().split(' | ')
        if s.startswith('ENTITIES'):
            s = s.split()[0]
        if o.startswith('ENTITIES'):
            o = o.split()[0]
        new_triple_list.append([s, p, o])
    new_triples.append(new_triple_list)

triple2lexs = {}
for triples, lex, rplc_dict in zip(new_triples, lexs, rplc_list):
    triple_str = '<TSP>'.join(['|'.join(t) for t in triples])
    triple_str_ent = triple_str
    if 'designer' in triple_str_ent:
        a = 0
    lex_ent = lex
    for k, v in rplc_dict.items():
        k = unidecode(k)
        rplc_dict[k] = unidecode(v)
        rplc_dict[k] = re.sub('\s+', ' ', ' '.join(re.split('(\W)', rplc_dict[k])).strip())
        triple_str = triple_str.replace(k, v)
        lex = lex.replace(k, v)
    triple_str = re.sub('\s+', ' ', ' '.join(re.split('(\W)', triple_str)).strip()).lower()
    triple_str_ent = re.sub('\s+', ' ', ' '.join(re.split('(\W)', triple_str_ent)).strip())
    lex = re.sub('\s+', ' ', ' '.join(re.split('(\W)', lex)).strip()).lower()
    lex_ent = re.sub('\s+', ' ', ' '.join(re.split('(\W)', lex_ent)).strip())
    if unidecode(triple_str) not in triple2lexs.keys():
        triple2lexs[unidecode(triple_str)] = []
    triple2lexs[unidecode(triple_str)].append([lex, triple_str_ent, lex_ent, rplc_dict])

    # if triple_str not in triple2lexs.keys():
    #     triple2lexs[triple_str] = []
    # triple2lexs[triple_str].append([lex, triple_str_ent, lex_ent, rplc_dict])

notdelex_triples = open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev-webnlg-all-notdelex.triple', 'r').readlines()
notdelex_lexs = open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev-webnlg-all-notdelex.lex', 'r').readlines()

from collections import Counter
notdelex_triples = Counter(notdelex_triples)

new_triple_file = []
new_lex_file = []
new_rplc_file = []
for k, v in notdelex_triples.items():
    k = k.strip().lower().replace('_',' ')
    k = re.sub('\s+', ' ', k)
    try:
        if len(triple2lexs[k])!=v:
            a = 0
    except:
        k = k.replace('u . s .', 'united states')
        if k == 'aip advances | editor | a . t . charlie johnson < tsp > a . t . charlie johnson | residence | united states < tsp > aip advances | publisher | american institute of physics':
            k = 'american institute of physics advances | editor | a . t . charlie johnson < tsp > a . t . charlie johnson | residence | united states < tsp > american institute of physics advances | publisher | american institute of physics'
        elif k == 'a . f . c . blackpool | manager | stuart parker ( footballer ) < tsp > stuart parker ( footballer ) | club | kv mechelen < tsp > stuart parker ( footballer ) | club | irlam town f . c .':
            k = 'a . f . c . blackpool | manager | stuart parker ( footballer ) < tsp > stuart parker ( footballer ) | club | kv mechelen < tsp > stuart parker ( footballer ) | club | irlam town football club'
        elif k == 'a . s . roma | fullname | associazione sportiva roma s . p . a . < tsp > a . s . roma | ground | rome , italy < tsp > a . s . roma | numberofmembers | 70634':
            k = 'associazione sportiva roma | fullname | associazione sportiva roma s . p . a . < tsp > associazione sportiva roma | ground | rome , italy < tsp > associazione sportiva roma | numberofmembers | 70634'
        elif k == 'a . s . gubbio 1910 | fullname | associazione sportiva gubbio 1910 srl < tsp > a . s . gubbio 1910 | season | 2014 < tsp > a . s . gubbio 1910 | ground | italy < tsp > a . s . gubbio 1910 | numberofmembers | 5300':
            k = 'associazione sportiva gubbio 1910 | fullname | associazione sportiva gubbio 1910 srl < tsp > associazione sportiva gubbio 1910 | season | 2014 < tsp > associazione sportiva gubbio 1910 | ground | italy < tsp > associazione sportiva gubbio 1910 | numberofmembers | 5300'
        elif k== 'a . s . gubbio 1910 | fullname | associazione sportiva gubbio 1910 srl < tsp > a . s . gubbio 1910 | season | 2014 < tsp > a . s . gubbio 1910 | ground | stadio pietro barbetti < tsp > a . s . gubbio 1910 | numberofmembers | 5300':
            k='associazione sportiva gubbio 1910 | fullname | associazione sportiva gubbio 1910 srl < tsp > associazione sportiva gubbio 1910 | season | 2014 < tsp > associazione sportiva gubbio 1910 | ground | stadio pietro barbetti < tsp > associazione sportiva gubbio 1910 | numberofmembers | 5300'
        elif k=='a . s . roma | numberofmembers | 70634 < tsp > a . s . roma | fullname | associazione sportiva roma s . p . a . < tsp > a . s . roma | ground | rome < tsp > a . s . roma | season | 2014 - 15 serie a':
            k='associazione sportiva roma | numberofmembers | 70634 < tsp > associazione sportiva roma | fullname | associazione sportiva roma s . p . a . < tsp > associazione sportiva roma | ground | rome < tsp > associazione sportiva roma | season | 2014 - 15 serie a'
        elif k=='serie a | champions | juventus f . c . < tsp > a . s . roma | fullname | associazione sportiva roma s . p . a . < tsp > a . s . roma | ground | stadio olimpico < tsp > a . s . roma | league | serie a':
            k='serie a | champions | juventus f . c . < tsp > associazione sportiva roma | fullname | associazione sportiva roma s . p . a . < tsp > associazione sportiva roma | ground | stadio olimpico < tsp > associazione sportiva roma | league | serie a'
        try:
            if len(triple2lexs[k])!=v:
                a = 0
        except:
            a = 0
            continue

    for item in triple2lexs[k]:
        new_triple_file.append(item[1])
        new_lex_file.append(item[2])
        new_rplc_file.append(item[3])

with open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev-webnlg-all-delex.lex','w') as f:
    for it in new_lex_file:
        f.write(it+'\n')

with open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev-webnlg-all-delex.triple','w') as f:
    for it in new_triple_file:
        f.write(it+'\n')

with open('/raid/user8/graph4nlp-rdf2text/examples/pytorch/rdf2text/data/webnlg1/raw/dev_rplc_list.pkl','wb') as f:
    pkl.dump(new_rplc_file, f)

a = 0

# import json
# from collections import Counter
#
# ent2type_file = open('webnlg1_ent2type.json', 'r')
# ent2type = json.load(ent2type_file)
#
# type2cnt = {}
# for k,vs in ent2type.items():
#     for v in vs:
#         if v in type2cnt.keys():
#             type2cnt[v] += 1
#         else:
#             type2cnt[v] = 1
#
# type2cnt = Counter(type2cnt)
# type2cnt_sort = type2cnt.most_common()
#
# supl_type = [t_k for t_k, t_v in type2cnt_sort if t_v>10]
# supl_type.remove('comics')
# supl_type.remove('Character')
# supl_type.remove('IN')
# supl_type.remove('OF')
# supl_type.remove('HOLDER')
# supl_type.remove('THE')
# supl_type.remove('UNITED')
# supl_type.remove('STATES')
# supl_type.remove('SETTLEMENT')
# supl_type.remove('ORGANISATION')
# supl_type.remove('ARTIST')
# supl_type.remove('OFFICE')
# supl_type.remove('ADMINISTRATIVE')
# supl_type.remove('BUILDING')
# supl_type.remove('TOWN')
# supl_type.remove('LIVING')
# supl_type.remove('POLITICIAN')
# supl_type.remove('CORPORATION')
#
# type2ent = {}
# for t in supl_type:
#     type2ent[t] = []
#
# add_k = []
# for new_type in supl_type:
#     for k,v in ent2type.items():
#         v = [x.upper() for x in v]
#         if new_type in v and k not in add_k:
#             type2ent[new_type].append(k)
#             add_k.append(k)
#
# with open('raw/delex_dict.json', 'r') as f:
#     delex_dict = json.load(f)
#
# type2ent = {**delex_dict, **type2ent}
#
# ent2type_new = {v.lower().replace('_', ' '): k.upper() for k, vs in type2ent.items() for v in vs}
#
# with open('raw/webnlg1_ent2type_new.json', 'w') as f:
#     f.write(json.dumps(ent2type_new, indent=4, ensure_ascii=False))
#
# with open('raw/webnlg1_delex_dict.json', 'w') as f:
#     f.write(json.dumps(type2ent, indent=4, ensure_ascii=False))
#
# a = 0

# 'azerbaijan | capital | baku < TSP > baku turkish martyrs ' memorial | material | red granite and white marble < TSP > azerbaijan | leaderTitle | prime minister of azerbaijan < TSP > baku turkish martyrs ' memorial | dedicatedTo | ottoman army soldiers killed in the battle of baku < TSP > baku turkish martyrs ' memorial | location | azerbaijan < TSP > baku turkish martyrs ' memorial | designer | huseyin butuner and hilmi guner < TSP > azerbaijan | legislature | national assembly ( azerbaijan )'
# 'azerbaijan | capital | baku < TSP > baku turkish martyrs ' memorial | material | red granite and white marble < TSP > azerbaijan | leaderTitle | prime minister of azerbaijan < TSP > baku turkish martyrs ' memorial | dedicatedTo | ottoman army soldiers killed in the battle of baku < TSP > baku turkish martyrs ' memorial | location | azerbaijan < TSP > baku turkish martyrs ' memorial | designer | huseyin butuner and hilmi guner < TSP > azerbaijan | legislature | national assembly ( azerbaijan )'