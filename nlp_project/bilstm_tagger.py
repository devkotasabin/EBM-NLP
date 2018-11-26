import dynet as dy
from collections import Counter
import random
import numpy as np
from operator import add

import util

# format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.
# train_file="/Users/yogo/Vork/Research/corpora/pos/WSJ.TRAIN"
# test_file="/Users/yogo/Vork/Research/corpora/pos/WSJ.TEST"

train_file="dataset/train.txt"
test_file="dataset/test.txt"


MLP=True

def read(fname):
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            # w,p = line
            w, pos, p1, p2 = line
            sent.append((w,p1,p2))

train=list(read(train_file))
test=list(read(test_file))
words=[]
tags=[]
wc=Counter()

for s in train:
    # for w, p in s:
    for w,p1,p2 in s:
        words.append(w)
        # tags.append(p)
        tags.append(p1)
        tags.append(p2)
        wc[w]+=1

words.append("_UNK_")
#words=[w if wc[w] > 1 else "_UNK_" for w in words]
tags.append("_START_")

for s in test:
    # for w,p in s:
    for w, p1, p2 in s:
        words.append(w)

vw = util.Vocab.from_corpus([words])
vt = util.Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

E = model.add_lookup_parameters((nwords, 128))
p_t1  = model.add_lookup_parameters((ntags, 30))
if MLP:
    pH = model.add_parameters((32, 50*2))
    pO = model.add_parameters((ntags, 32))
else:
    pO = model.add_parameters((ntags, 50*2))

builders=[
        dy.LSTMBuilder(1, 128, 50, model),
        dy.LSTMBuilder(1, 128, 50, model),
        ]

builders2 = [
        dy.LSTMBuilder(1, 2*50, 50, model),
        dy.LSTMBuilder(1, 2*50, 50, model),
    ]

def build_tagging_graph_lvl1(words, tags, builders):
    dy.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    wembs = [E[w] for w in words]
    wembs = [dy.noise(we,0.1) for we in wembs]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    # fw_rnn_hidden_outs = [x.value() for x in fw]
    # bw_rnn_hidden_outs = [x.value() for x in bw]

    # print ("Transducing")
    fw_rnn_hidden_outs = f_init.transduce(wembs)
    bw_rnn_hidden_outs = b_init.transduce(reversed(wembs))



    if MLP:
        H = dy.parameter(pH)
        O = dy.parameter(pO)
    else:
        O = dy.parameter(pO)
    errs = []
    for f,b,t in zip(fw, reversed(bw), tags):
        f_b = dy.concatenate([f,b])
        if MLP:
            r_t = O*(dy.tanh(H * f_b))
        else:
            r_t = O * f_b
        err = dy.pickneglogsoftmax(r_t, t)
        errs.append(err)

    return {'err': dy.esum(errs), 'fw': fw_rnn_hidden_outs, 'bw':bw_rnn_hidden_outs}

def build_tagging_graph_lvl2(embeds, words, tags, builders):
    dy.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    wembs = [E[w] for w in words]
    wembs = [dy.noise(we,0.1) for we in wembs]

    # fw = [x.output() for x in f_init.add_inputs(embeds)]
    # bw = [x.output() for x in b_init.add_inputs(reversed(embeds))]

    # fw = [x.output() for x in f_init.add_inputs(wembs)]
    # bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    fw = f_init.transduce(embeds)
    bw = b_init.transduce(reversed(embeds))

    if MLP:
        H = dy.parameter(pH)
        O = dy.parameter(pO)
    else:
        O = dy.parameter(pO)
    errs = []
    for f,b,t in zip(fw, reversed(bw), tags):
        f_b = dy.concatenate([f,b])
        if MLP:
            r_t = O*(dy.tanh(H * f_b))
        else:
            r_t = O * f_b
        err = dy.pickneglogsoftmax(r_t, t)
        errs.append(err)

    return dy.esum(errs)

# This function prepares the input for level 2
# Uses the gold tags in level 1 for training
# tags is the gold tags in level 1
# fw is the output from forward LSTM from level 1
# bw is the output from backward LSTM from level 1
# isNew: if true, learns from the outer tag to inner tag
#       if false, learns from the inner tag to outer tag

def prepare_for_lvl2(tags, fw, bw, isNew=True):
    prev_t = 'O'
    start_index = 0
    started_seq = False
    interval_list = []

    # print tags

    for i,t in enumerate(tags):
        if t != 'O':
            if not started_seq:
                started_seq = True
                start_index = i
            elif prev_t != t:
                # if prev_t != -1:
                end_index = i-1
                interval_list.append((start_index, end_index))
                start_index = i

        else:
            if started_seq:
                started_seq = False
                end_index = i-1
                interval_list.append((start_index, end_index))
        prev_t = t

    if started_seq:
        started_seq = False
        end_index = i
        interval_list.append((start_index, end_index))

    f_b = [dy.concatenate([f,b]) for f,b in zip(fw, reversed(bw))]
    # f_b = [f+b for f,b in zip(fw, reversed(bw))]
    
    # print interval_list
    
    for (s,e) in interval_list:
        avg = f_b[s]
        for i in range(s+1, e+1):
            # avg += f_b[i]
            # avg = list(map(add, avg, f_b[i]))
            avg = [a+b for a,b in zip(avg, f_b[i])]
        # avg /= (float(e-s+1)*10)
        divisor = (e-s+1)*10.0
        # print divisor
        avg = [a/divisor for a in avg]
        
        for i in range(s, e+1):
            # f_b[i] = f_b[i] + avg
            f_b[i] = [a+b for a,b in zip(avg, f_b[i])]

    return f_b


def tag_sent(sent, builders):
    dy.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    wembs = [E[vw.w2i.get(w, UNK)] for w,t,t2 in sent]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    # fw_rnn_hidden_outs = [x.value() for x in fw]
    # bw_rnn_hidden_outs = [x.value() for x in bw]

    fw_rnn_hidden_outs = f_init.transduce(wembs)
    bw_rnn_hidden_outs = b_init.transduce(reversed(wembs))

    if MLP:
        H = dy.parameter(pH)
        O = dy.parameter(pO)
    else:
        O = dy.parameter(pO)
    tags=[]
    for f,b,(w,t, t2) in zip(fw,reversed(bw),sent):
        if MLP:
            r_t = O*(dy.tanh(H * dy.concatenate([f,b])))
        else:
            r_t = O*dy.concatenate([f,b])
        out = dy.softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(vt.i2w[chosen])
    return {'tags':tags, 'fw':fw_rnn_hidden_outs, 'bw':bw_rnn_hidden_outs}

def tag_sent_lvl2(sent, input_embeds ,builders):
    dy.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    # wembs = [E[vw.w2i.get(w, UNK)] for w,t,t2 in sent]

    fw = [x.output() for x in f_init.add_inputs(input_embeds)]
    bw = [x.output() for x in b_init.add_inputs(reversed(input_embeds))]

    if MLP:
        H = dy.parameter(pH)
        O = dy.parameter(pO)
    else:
        O = dy.parameter(pO)
    tags=[]
    for f,b,(w,t, t2) in zip(fw,reversed(bw),sent):
        if MLP:
            r_t = O*(dy.tanh(H * dy.concatenate([f,b])))
        else:
            r_t = O*dy.concatenate([f,b])
        out = dy.softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(vt.i2w[chosen])
    return tags


tagged = loss = loss2 = 0

NUM_ITERS = 50

for ITER in range(NUM_ITERS):
    random.shuffle(train)
    for i,s in enumerate(train,1):
        # if i:
        if i % 5000 == 0:
            print ("Training status: Iter %d, Sample %d" %(ITER, i))
            trainer.status()
            if (tagged != 0):
                print('Level 1 avg loss %f' %(loss / tagged))
                print('Level 2 avg loss %f' %(loss2/tagged))
            loss = 0
            loss2 = 0
            tagged = 0
        # if i:
        if i % 10000 == 0:
            # Test level 1
            with open("results/%d.txt" %(ITER), 'w') as f:
                print ("Test status Level 1: Iter %d" %(ITER))
                good = bad = 0.0
                good2 = bad2 = 0.0
                for sent in test:
                    result = tag_sent(sent, builders)
                    tags = result['tags']
                    # Implement testing for level 2
                    fw = result['fw']
                    bw = result['bw']

                    level2_inp = prepare_for_lvl2(tags, fw, bw)
                    tags_lvl2 = tag_sent_lvl2(sent, level2_inp, builders2)

                    words = [w for w,t,t2 in sent]
                    golds = [t for w,t,t2 in sent]
                    for w,go,gu in zip(words,golds,tags):
                        f.write('%s\t%s\t%s\t' %(w, go, gu))
                        if go == gu: good +=1 
                        else: bad+=1
                    
                    golds2 = [t2 for w,t,t2 in sent]
                    for go,gu in zip(golds2,tags_lvl2):
                        f.write('%s\t%s\n' %(go, gu))
                        if go == gu: good2 +=1 
                        else: bad2 += 1
                    f.write('\n')

                print("Accuracy Level 1: %f" %(good/(good+bad)))
                print("Accuracy Level 2: %f" %(good2/(good2+bad2)))
                


        #Train Level 1
        ws = [vw.w2i.get(w, UNK) for w,p,p2 in s]
        ps = [vt.w2i[p] for w,p,p2 in s]
        result = build_tagging_graph_lvl1(ws,ps,builders)
        sum_errs1 = result['err']
        fw = result['fw']
        bw = result['bw']

        # squared = -sum_errs# * sum_errs
        
        loss += sum_errs1.scalar_value()
        tagged += len(ps)
        
        sum_errs1.backward()
        trainer.update()

        p_tags = [p for w,p,p2 in s]
        print(fw)

        level2_inp = prepare_for_lvl2(p_tags, fw, bw)
        p2s = [vt.w2i[p2] for w,p,p2 in s]
        sum_errs2 = build_tagging_graph_lvl2(level2_inp, ws, p2s, builders2)

        loss2 += sum_errs2.scalar_value()
        
        sum_errs2.backward()
        trainer.update()

