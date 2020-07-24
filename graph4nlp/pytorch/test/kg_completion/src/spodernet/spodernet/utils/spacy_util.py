import spacy

subjects = set(['nsubj'])
objects = set(['dobj', 'pobj'])

def merge_noun_phrases(sent_doc):
    for np in sent_doc.noun_chunks:
       np.merge(np.root.tag_, np.text, np.root.ent_type_)

def merge_entities(sent_doc):
    for ent in sent_doc.ents:
        ent.merge(ent.root.dep_, ent.text, ent.label_)

def merge_verbs(sent_doc):
    has_double_verb = False
    for span_length in [3, 2]:
        i = 1
        while i < len(sent_doc)-1:
            token = sent_doc[i]
            if token.pos_ == 'VERB':
                full_match = True
                for j in range(1, span_length):
                    full_match &= sent_doc[i-j].pos_ == 'VERB'
                if full_match:
                    span = sent_doc[i-1:i+span_length-1]
                    span.merge()
                    i += span_length-1
                    has_double_verb = True
            i += 1

def merge_with_set(sent_doc, to_match, write_key='pobj'):
    for span_length in [5, 4, 3, 2]:
        i = span_length-1
        while i < len(sent_doc)-1:
            token = sent_doc[i]
            if token.dep_ in write_key:
                pos, dep = token.pos_, token.dep_
                full_match = True
                for j in range(1, span_length):
                    full_match &= sent_doc[i-j].dep_ in to_match
                    full_match &= sent_doc[i-j].pos_ != 'VERB'
                    idx = sent_doc[i-j].idx
                if full_match:
                    span = sent_doc[i-1:i+span_length-1]
                    span.merge()
                    sent_doc[i-1].dep_ = dep
                    i += span_length-1
            i += 1

def merge_tokens(sent_doc):
    merge_noun_phrases(sent_doc)
    merge_entities(sent_doc)
    merge_verbs(sent_doc)
    merge_with_set(sent_doc, set(['pobj', 'prep']))
    merge_with_set(sent_doc, set(['pobj', 'prep']))
    merge_with_set(sent_doc, set(['pobj', 'prep']))
    merge_with_set(sent_doc, set(['attr', 'punct', 'cc', 'conj', 'pobj']), 'pobj')
    merge_with_set(sent_doc, set(['attr', 'punct', 'cc', 'conj', 'pobj']), 'pobj')
    merge_with_set(sent_doc, set(['dobj', 'pobj']), 'dobj')
    merge_with_set(sent_doc, set(['dobj', 'pobj']), 'dobj')
    merge_with_set(sent_doc, set(['attr', 'punct', 'cc', 'conj', 'pobj', 'dobj']), 'dobj')
    merge_with_set(sent_doc, set(['attr', 'punct', 'cc', 'conj', 'pobj', 'dobj']), 'dobj')

def extract_triples(sent_doc):
    triples = []
    triple = []
    for token in sent_doc:
        if token.pos_ == 'VERB':
            if len(triple) == 0: continue
            if triple[-1].dep_ in subjects:
                triple.append(token)
            else:
                triple = []
        if token.dep_ in subjects:
            if len(triple) == 0:
                triple.append(token)
            else:
                triple = [token]
        if token.dep_ in objects:
            if len(triple) == 0: continue
            if triple[-1].pos_ == 'VERB':
                triple.append(token)
                triples.append(triple)
                triple = []
            else:
                triple = []
    return triples
