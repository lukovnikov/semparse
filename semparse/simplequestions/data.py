import numpy as np
import re
import qelos as q
from torch.utils.data import TensorDataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer
from tabulate import tabulate
from unidecode import unidecode
from collections import OrderedDict
from tqdm import tqdm
import pickle as pkl


def get_all_rels(p="../../data/buboqa/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt",
                 outp="../../data/buboqa/data/rels.txt"):
    rels = set()
    relre = re.compile("www\.freebase\.com/(.+)")
    for line in tqdm(open(p).readlines()):
        m = relre.match(line.split("\t")[1])
        if m:
            relparts = m.group(1).split("/")
            rel = "fb:" + ".".join(relparts)
            rels.add(rel)
        else:
            raise Exception("not matching regex")
    print(len(rels))
    rels = sorted(rels)
    rels = [rel+"\n" for rel in rels]
    print(rels)
    with open(outp, "w") as f:
        f.writelines(rels)


def get_all_types(p="../../data/buboqa/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt",
                  outp="../../data/buboqa/data/ent2type.pkl"):
    types = set()
    typere = re.compile("www\.freebase\.com/m/(.+)")
    typerel = ["www.freebase.com/common/topic/notable_types"]
    ent2types = {}
    for line in tqdm(open(p).readlines()):
        if line.split("\t")[1] in typerel:
            splits = line.strip().split("\t")
            subj = "fb:m."+typere.match(splits[0]).group(1)
            obj = "fb:m."+typere.match(splits[2]).group(1)
            if subj not in ent2types:
                ent2types[subj] = set()
            ent2types[subj].add(obj)
            types.add(obj)
    types= sorted(types)
    types = [rel+"\n" for rel in types]
    print("".join(types))
    print(len(types))
    maxtypesperent = max([len(v) for k, v in ent2types.items()])
    print("max types per ent: {}".format(maxtypesperent))
    ent2type = {k: list(v)[0] if len(v) != 0 else "none" for k, v in ent2types.items()}
    print(len(ent2type))
    pkl.dump(ent2type, open(outp, "wb"))
    # with open(outp, "w") as f:
    #     f.writelines(rels)


def get_names_for_entities(namespath="../../data/buboqa/data/freebase_names/names.trimmed.2M.txt",
                           outpath="../../data/buboqa/indexes/names_2M_withtypes.pkl",
                           withtype=True):
    print("getting names map...")
    names = {}
    with open(namespath, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))
                continue
            entity = items[0]
            nametype = items[1]
            literal = items[2].strip()
            if literal != "":
                if entity not in names:
                    names[entity] = []
                if withtype:
                    names[entity].append((literal, nametype))
                else:
                    names[entity].append(literal)
    if outpath is not None:
        pkl.dump(names, open(outpath, "wb"))
    return names


def load_data(p="../../data/buboqa/data/processed_simplequestions_dataset/",
              relp="../../data/buboqa/data/rels.txt",
              typep="../../data/buboqa/data/ent2type.pkl",
              outp="../../data/buboqa/data/bertified_dataset_v2",
              ):
    tt = q.ticktock("dataloader")
    tt.tick("loading files")
    trainlines = open(p+"train.txt", encoding="utf8").readlines()
    devlines = open(p+"valid.txt", encoding="utf8").readlines()
    testlines = open(p+"test.txt", encoding="utf8").readlines()
    allrels = [x.strip() for x in open(relp).readlines()]
    ent2type = pkl.load(open(typep, "rb"))
    tt.tock("files loaded")
    tt.tick("splitting")
    trainlines = [line.strip().split("\t") for line in trainlines]
    devlines = [line.strip().split("\t") for line in devlines]
    testlines = [line.strip().split("\t") for line in testlines]
    tt.tock("splitted")

    tt.tick("doing some stats")
    stt = q.ticktock("datastats")
    trainrels = set([line[3] for line in trainlines])
    devrels = set([line[3] for line in devlines])
    testrels = set([line[3] for line in testlines])
    unkrels = set()
    for line in testlines:
        if line[3] not in trainrels:
            unkrels.add(line[0])

    stt.msg("{}/{} unique rels in test not in train ({})"
            .format(len(testrels-trainrels), len(testrels), len(trainrels)))
    stt.msg("{}/{} unique rels in devnot in train ({})"
            .format(len(devrels-trainrels), len(devrels), len(trainrels)))
    stt.msg("{} unique rels".format(len(trainrels | devrels | testrels)))
    stt.msg("{}/{} unkrel cases in test".format(len(unkrels), len(testlines)))

    # print(trainlines[3])

    tt.tick("creating word matrix")
    sm = q.StringMatrix(specialtoks=["<ENT>"], indicate_end=True)
    sm.tokenize = lambda x: x.split()
    wordborders = np.zeros((len(trainlines) + len(devlines) + len(testlines), 2), dtype="int64")

    def do_line(line_, i_):
        try:
            sm.add(line_[5])
            previo = "O"
            ioline = line_[6]
            if "[" in ioline or not "I" in ioline:
                print(ioline)
            ioline = ioline.replace("'", "").replace("[", "").replace("]", "").replace(",", "")
            io = ioline.split() + ["O"]
            k = 0
            for j in range(len(io)):
                if io[j] != previo:
                    if k > 1:
                        print(line_)
                    wordborders[i_, k] = j
                    previo = io[j]
                    k += 1
        except Exception as e:
            print(e)
            print(line_)

    i = 0
    for line in tqdm(trainlines):
        do_line(line, i)
        i += 1
    word_devstart = i
    for line in tqdm(devlines):
        do_line(line, i)
        i += 1
    word_teststart = i
    for line in tqdm(testlines):
        do_line(line, i)
        i += 1

    sm.finalize()
    print(len(sm.D))
    print(sm[0])
    tt.tock("created word matrix")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def bertify(line):
        try:
            subj = line[1]
            subjtype = ent2type[subj] if subj in ent2type else "none"
            rel = line[3]
            sent = "[CLS] {} [SEP]".format(line[5].lower())
            span = "O {} O".format(line[6]).split()
            bertsent = []       #tokenizer.basic_tokenizer.tokenize(sent)
            unberter = []
            sent = sent.split()
            bertspan = []
            for i, (token, io) in enumerate(zip(sent, span)):
                berttokens = tokenizer.tokenize(token)
                bertsent += berttokens
                bertspan += [io] * len(berttokens)
                unberter += [i] * len(berttokens)
        except Exception as e:
            print(e)
            print(line)
            # raise e
        return bertsent, bertspan, rel, unberter, subjtype

    k = 1331
    ret = bertify(trainlines[k])
    print(tabulate(ret[0:2]))
    print(ret[2])
    print(tabulate([trainlines[k][5].split(), trainlines[k][6].split()]))

    tt.tick("bertifying")
    bert_tokens_train, bert_io_train, bert_rel_train, unberter_train, bert_type_train = zip(*[bertify(line) for line in trainlines])
    bert_tokens_dev,   bert_io_dev,   bert_rel_dev,   unberter_dev, bert_type_dev   = zip(*[bertify(line) for line in devlines])
    bert_tokens_test,  bert_io_test,  bert_rel_test,  unberter_test, bert_type_test  = zip(*[bertify(line) for line in testlines])
    tt.tock("bertified")

    print(tabulate([bert_tokens_train[3], bert_io_train[3], unberter_train[3]]))
    print(bert_rel_train[3])

    print("{} entities in train have 'none' type".format())

    # construct numpy matrix with ids in bert vocabulary
    # and also, numpy matrix with spans
    # and also, numpy vector of relations and dictionary
    tt.tick("creating token matrix")
    assert(tokenizer.convert_tokens_to_ids(["[PAD]"]) == [0])
    maxlen = max([max([len(x) for x in bert_toks]) for bert_toks
                  in [bert_tokens_train, bert_tokens_dev, bert_tokens_test]])
    print(maxlen)
    tokmat = np.zeros((len(bert_tokens_train) + len(bert_tokens_dev) + len(bert_tokens_test),
                       maxlen), dtype="int32")
    i = 0
    for bert_toks in [bert_tokens_train, bert_tokens_dev, bert_tokens_test]:
        for x in bert_toks:
            xids = tokenizer.convert_tokens_to_ids(x)
            tokmat[i, :len(xids)] = xids
            i += 1
    devstart = len(bert_tokens_train)
    teststart = len(bert_tokens_train) + len(bert_tokens_dev)

    assert(word_devstart == devstart, word_teststart == teststart)

    print(tokmat.shape)
    tt.tock("token matrix created")

    tt.tick("creating io matrix")
    iomat = np.zeros_like(tokmat)
    iobordersmat = np.zeros((tokmat.shape[0], 2), dtype="int32")
    i = 0
    for bert_io in [bert_io_train, bert_io_dev, bert_io_test]:
        for x in bert_io:
            xids = [1 if xe == "O" else 2 for xe in x]
            iomat[i, :len(xids)] = xids
            ioborders = []
            for j in range(1, len(xids)):
                if xids[j] != xids[j-1]:
                    ioborders.append(j)
            iobordersmat[i, :len(ioborders)] = ioborders
            i += 1
    tt.tock("io matrix created")

    # unbert mat
    unbertmat = np.zeros_like(tokmat)
    i = 0
    for unberter in [unberter_train, unberter_dev, unberter_test]:
        for unbert_i in unberter:
            unbertmat[i, :len(unbert_i)] = [xe+1 for xe in unbert_i]
            i += 1

    tt.tick("testing")
    test_i = 1331
    test_tokids = [xe for xe in tokmat[test_i] if xe != 0]
    test_ios = iomat[test_i, :len(test_tokids)]
    test_tokens = tokenizer.convert_ids_to_tokens(test_tokids)
    print(tabulate([test_tokens, test_ios]))
    print(iobordersmat[test_i])
    tt.tock("tested")

    tt.tick("doing relations")
    bert_rel_all = bert_rel_train + bert_rel_dev + bert_rel_test
    allrelwcounts = dict(zip(allrels, [0]*len(allrels)))
    for rel in bert_rel_train:
        allrelwcounts[rel] += 1
    allrelwcounts = sorted(allrelwcounts.items(), key=lambda x: x[1], reverse=True)
    print(allrelwcounts[0])
    tt.msg("{} total unique rels".format(len(allrelwcounts)))
    relD = dict(zip([rel for rel in allrels],
                           range(len(allrels))))
    rels = [relD[xe] for xe in bert_rel_all]
    rels = np.array(rels).astype("int32")
    relcounts = [rel[1] for rel in allrelwcounts]
    relcounts = np.array(relcounts).astype("int32")
    tt.tock("done relations")

    np.savez(outp, wordmat=sm.matrix, worddic=sm.D, wordborders=wordborders,
             tokmat=tokmat, iomat=iomat, tokborders=iobordersmat,
             rels=rels, relD=relD, relcounts=relcounts, unbertmat=unbertmat,
             devstart=devstart, teststart=teststart)

    threshold = 2
    stt.msg("{} unique rels at least {} time(s) in train data".format(
        len([xe for xe in allrelwcounts if xe[1] > threshold]), threshold))
    rarerels = set([xe[0] for xe in allrelwcounts if xe[1] <= threshold])
    testrarecount = 0
    for rel in bert_rel_test:
        if rel in rarerels:
            testrarecount += 1
    stt.msg("{}/{} test examples affected by rare rel".format(
        testrarecount, len(bert_rel_test)
    ))

    tt.tick("reload")
    reloaded = np.load(open(outp+".npz", "rb"))
    _relD = reloaded["relD"].item()
    _tokmat = reloaded["tokmat"]
    print(reloaded["devstart"])
    tt.tock("reloaded")


def run(lr=0):
    load_data()



if __name__ == '__main__':
    # q.argprun(run)
    # q.argprun(get_all_types)
    q.argprun(get_names_for_entities)