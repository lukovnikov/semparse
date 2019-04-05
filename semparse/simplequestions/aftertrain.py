import json
import pickle as pkl
import numpy as np
import re
from unidecode import unidecode
from tabulate import tabulate
import qelos as q
import os
import fuzzywuzzy
import torch
from pytorch_pretrained_bert import BertTokenizer
from tabulate import tabulate
from fuzzyset import FuzzySet
from fuzzywuzzy import fuzz
from tqdm import tqdm
from bloom_filter import BloomFilter
from functools import partial
from pprint import PrettyPrinter
from IPython import embed


class MiniBaseIndex(object):
    def __init__(self, field=None, tokenizer=None, similarity=None, base=None, idf_limit=0.05, **kw):
        super(MiniBaseIndex, self).__init__(**kw)
        self.content = {}
        self.field = field
        self.tokenizer = tokenizer
        self.similarity = similarity
        self.base = base
        self.counts = {}
        self.fuzzwords = FuzzySet(rel_sim_cutoff=0.7, use_levenshtein=False)
        self.blacklist = set()
        self.idf_limit = idf_limit

    def add(self, tok, i):
        if tok not in self.content:
            if tok not in self.blacklist:
                self.content[tok] = set()
            self.counts[tok] = 0
        self.content[tok].add(i)
        self.counts[tok] += 1
        # if self.counts[tok]/len(self.base.entries) > self.idf_limit:
        #     self.blacklist.add(tok)
        #     del self.counts[tok]
        #     del self.content[tok]
        self.fuzzwords.add(tok)

    def finalize(self):
        for tok in self.content:
            pass
            # self.fuzzwords.add(tok)

    def search(self, x, expl=5000, top=25, maxtok=250, debug=False):
        tokenizer = self.tokenizer
        xtoks = tokenizer(x)
        # maxtok = maxtok * len(xtoks)
        results = {}
        # collect all toks
        alltoks = []
        alltoks_set = set()
        for xtok in xtoks:
            for xtok_fuzz_score, xtok_fuzz_tok \
                    in self.fuzzwords.get(xtok):
                xtok_fuzz_sim = self.similarity(xtok, xtok_fuzz_tok)
                if xtok_fuzz_tok not in alltoks_set:
                    alltoks.append((xtok_fuzz_score, xtok_fuzz_tok, xtok_fuzz_sim))
                    alltoks_set.add(xtok_fuzz_tok)
        # alltoks = list(alltoks)
        # sort together by fuzziness
        alltoks = sorted(alltoks, key=lambda x: x[2]*100 + 1/self.counts[x[1]], reverse=True)
        # take maxtok only
        if debug:
            print(len(alltoks), maxtok)
            for tok in alltoks:
                print(tok, self.counts[tok[1]])
        alltoks = alltoks[:maxtok]
        # sort by inverse frequency
        # alltoks = sorted(alltoks, key=lambda x: self.counts[x[1]])
        # alltoksset = set(alltoks)
        for xtok_fuzz_score, xtok_fuzz_tok, xtok_fuzz_sim in alltoks:
            for _id in self.content[xtok_fuzz_tok]:
                if _id not in results:
                    results[_id] = 0
                results[_id] += xtok_fuzz_score
                if len(results) > expl:
                    break
            if len(results) > expl:
                break
        if debug:
            print(len(results))
        results = [(res[0], res[1], self.similarity(x, self.base.entries[res[0]][self.field]))
                   for res in results.items()]

        def sortkey(x):
            entid = x[0]
            pop = self.base.entries[entid]["pop"]
            sim = x[2]
            return sim * 1e2 + pop * 1e-3

        results = sorted(results, key=sortkey, reverse=True)
        results = results[:top]
        return results


class MiniBase(object):
    def __init__(self, **kw):
        super(MiniBase, self).__init__(**kw)
        self.entries = []   # list of dicts
        self.indexes = {}

    @classmethod
    def load(cls, p):
        import dill
        ret = dill.load(open(p, "rb"))
        return ret

    def save(self, p):
        import dill
        dill.dump(self, open(p, "wb"))

    def add(self, **kw):
        self.entries.append(kw)

    def build_index(self, field=None, tokenizer=None, similarity=None):
        if tokenizer == None:
            tokenizer = lambda x: x.split()
        if similarity == None:
            similarity = lambda q, k: fuzz.ratio(q, k)
        index = MiniBaseIndex(field=field, tokenizer=tokenizer, similarity=similarity, base=self)
        i = 0
        for entry in tqdm(self.entries):
            if field in entry:
                fieldval = entry[field]
                fieldval_tokens = tokenizer(fieldval)
                for tok in fieldval_tokens:
                    index.add(tok, i)
            i += 1
        assert(field not in self.indexes)
        index.finalize()
        self.indexes[field] = index

    def search(self, x=None, field=None, fuzziness=0, top=20, expl=5000, debug=False):
        if field is None:
            if len(self.indexes) == 1:
                field = list(self.indexes.keys())[0]
        index = self.indexes[field]
        ret_ids = index.search(x, top=top, expl=expl, debug=debug)
        results = [
            {
             "entry_number": res[0],
             "similarity": res[2],
             # "number_query_words_matched": res[1],
             "entry": self.entries[res[0]]} for res in ret_ids]
        return results


class FuzzyBaseIndex(object):
    def __init__(self, field=None, similarity=None, base=None, **kw):
        super(FuzzyBaseIndex, self).__init__(**kw)
        self.fuzz = FuzzySet(rel_sim_cutoff=1., use_levenshtein=False)
        self.content = {}
        self.field = field
        self.similarity = similarity
        self.base = base

    def add(self, x, i):
        self.fuzz.add(x)
        if x not in self.content:
            self.content[x] = set()
        self.content[x].add(i)

    def finalize(self):
        pass

    def search(self, x, top=25, debug=True):
        results = self.fuzz.get(x)
        ret = []
        for r in results:
            for i in self.content[r[1]]:
                sim = self.similarity(x, r[1])
                ret.append((i, r[0], sim))
        ret = sorted(ret, key=lambda x: x[2], reverse=True)
        ret = ret[:top]
        return ret


class FuzzyBase(object):
    def __init__(self, **kw):
        super(FuzzyBase, self).__init__(**kw)
        self.entries = []   # list of dicts
        self.indexes = {}

    @classmethod
    def load(cls, p):
        import dill
        ret = dill.load(open(p, "rb"))
        return ret

    def save(self, p):
        import dill
        dill.dump(self, open(p, "wb"))

    def add(self, **kw):
        self.entries.append(kw)

    def build_index(self, field=None, similarity=None):
        if similarity == None:
            similarity = lambda q, k: fuzz.ratio(q, k)
        index = FuzzyBaseIndex(field=field, similarity=similarity, base=self)
        i = 0
        for entry in tqdm(self.entries):
            if field in entry:
                fieldval = entry[field]
                index.add(fieldval, i)
            i += 1
        assert(field not in self.indexes)
        index.finalize()
        self.indexes[field] = index

    def search(self, x=None, field=None, fuzziness=0):
        if field is None:
            if len(self.indexes) == 1:
                field = list(self.indexes.keys())[0]
        index = self.indexes[field]
        ret_ids = index.search(x)
        results = [
            {"entry_number": res[0],
             "similarity": res[2],
             "fuzz": res[1],
             "entry": self.entries[res[0]]} for res in ret_ids]
        return results



def build_entity_index(p="../../data/buboqa/indexes/names_2M.pkl",
                       ncp="../../data/buboqa/indexes/numconnections_2M.pkl",
                       outp="../../data/buboqa/data/names_2M.labels.index",
                       testsearch=False):
    tt = q.ticktock("index-entities")
    tt.tick("loading names pkl")
    entnames = pkl.load(open(p, "rb"))
    numrels = pkl.load(open(ncp, "rb"))
    tt.tock("loaded names pkl")
    tt.tick("building minibase")
    fset = MiniBase()
    for uri, names in tqdm(entnames.items()):
        for name in names:
            fset.add(sf=name, uri=uri, pop=numrels[uri])
    tt.tock("built")
    tt.tick("building index")
    fset.build_index("sf")
    tt.tock("index built")
    # tt.tick("saving minibase")
    # pkl.dump(fset.indexes["sf"].content, open(outp, "wb"))
    # fset.save(outp)
    # tt.tock("saved")
    tt.tick("test search")
    query = "cassiesteele"
    query = "the science museum ( london )"
    query = "w '' is for wasted"
    queyr = "home of the brave"
    # print("query: ", query)
    if testsearch:

        embed()
    # ret = fset.search(query)
    tt.tock("test search done")
    # pp = PrettyPrinter()
    # pp.pprint(ret)
    return fset

def test_index():
    base = MiniBase()
    base.add(sf="bla bla", uri="fb:blabla")
    base.add(sf="bla blam")
    base.add(sf="blam bla bl")
    base.add(sf="bla kablam")
    base.add(sf="bla blame")
    base.add(sf="blabla bla")
    base.add(sf="blablabla")
    base.add(sf="moo", uri="fb:moo")
    base.build_index(field="sf")
    ret = base.search("blam bla bla bla")
    pp = PrettyPrinter()
    pp.pprint(ret)





def build_entity_bloom(p="../../data/buboqa/indexes/names_2M.pkl",
                       outp="../../data/buboqa/data/names_2M.labels.bloom"):
    tt = q.ticktock("bloom-entities")
    tt.tick("loading names pkl")
    entnames = pkl.load(open(p, "rb"))
    tt.tock("loaded names pkl")
    tt.tick("building bloom filter")
    fset = BloomFilter(2e6, 1e-3)
    for uri, names in tqdm(entnames.items()):
        for name in names:
            fset.add(name)
    tt.tock("built")
    tt.tick("saving bloom filter")
    with open(outp, "wb") as f:
        pkl.dump(fset, f)
    tt.tock("saved")
    return fset


def run(indexp="../../data/buboqa/indexes/",
        datap="../../data/buboqa/data/"):
    names = pkl.load(open(indexp + "names_2M.pkl", "rb"))
    # entities = pkl.load(open(indexp + "entity_2M.pkl", "rb"))
    print(len(names))


def get_dsF1(p="exp_bilstm_span_borders_12",     #p="exp_bert_both_23",
             dp="../../data/buboqa/data/bertified_dataset_new.npz"):
    """ Assumes wordpiece-level predictions for spans"""
    print(p)
    data = np.load(dp)
    wordbordersgold, devstart, teststart = data["wordborders"], data["devstart"], data["teststart"]
    tokmat, unbertmat = data["tokmat"], data["unbertmat"]
    borderpreds = torch.tensor(np.load(os.path.join(p, "borderpreds.dev.npy")))
    #bordergold = torch.tensor(ioborders[devstart:teststart]).long()
    unbertmat = torch.tensor(unbertmat[devstart:teststart]).long()
    tokmat = torch.tensor(tokmat[devstart:teststart]).long()
    mask_test = (tokmat != 0).float()
    borderprobs = torch.nn.Softmax(-1)(borderpreds + torch.log(mask_test.unsqueeze(1)))
    word_border_probs = torch.zeros(len(unbertmat), 2, borderpreds.size(2), dtype=torch.float)
    word_border_probs.scatter_add_(2, unbertmat.unsqueeze(1).repeat(1, 2, 1), borderprobs)
    word_border_golds = torch.tensor(wordbordersgold)[devstart:teststart] #unbertmat.gather(1, bordergold)

    pred_start, pred_end = torch.argmax(word_border_probs, 2).split(1, dim=1)
    pred_start, pred_end = pred_start - 2, pred_end - 2
    gold_start, gold_end = word_border_golds.split(1, dim=1)
    overlap_start = torch.max(pred_start, gold_start)
    overlap_end = torch.min(pred_end, gold_end)
    overlap = (overlap_end - overlap_start).float().clamp_min(0).sum()
    expected = (gold_end - gold_start).float().clamp_min(1e-6).sum()
    predicted = (pred_end - pred_start).float().clamp_min(1e-6).sum()
    recall = overlap / expected
    precision = overlap / predicted
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-6)
    print("Dataset-wide F1, precision and recall:")
    print(f1.item(), precision.item(), recall.item())

    pred_start, pred_end = torch.argmax(word_border_probs, 2).split(1, dim=1)
    pred_start, pred_end = pred_start - 2, pred_end - 2
    gold_start, gold_end = word_border_golds.split(1, dim=1)
    overlap_start = torch.max(pred_start, gold_start)
    overlap_end = torch.min(pred_end, gold_end)
    overlap = (overlap_end - overlap_start).float().clamp_min(0)
    recall = overlap / (gold_end - gold_start).float().clamp_min(1e-6)
    precision = overlap / (pred_end - pred_start).float().clamp_min(1e-6)
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-6)
    print((recall > 1).nonzero(), (precision > 1).nonzero())
    recall = recall.mean()
    precision = precision.mean()
    acc = (f1 == 1).float().mean()
    f1 = f1.mean()
    print("Averaged F1, precision and recall:")
    print(f1.item(), precision.item(), recall.item())
    print("Span accuracy")
    print(acc)


def get_dsF1_wordlevel(p="exp_bilstm_span_borders_12",     #p="exp_bert_both_23",
             dp="../../data/buboqa/data/bertified_dataset_new.npz"):
    """ Assumes wordpiece-level predictions for spans"""
    print(p)
    data = np.load(dp)
    wordbordersgold, devstart, teststart = data["wordborders"], data["devstart"], data["teststart"]
    tokmat = data["wordmat"]
    borderpreds = torch.tensor(np.load(os.path.join(p, "borderpreds.dev.npy")))
    #bordergold = torch.tensor(ioborders[devstart:teststart]).long()
    tokmat = torch.tensor(tokmat[devstart:teststart]).long()
    mask_test = (tokmat != 0).float()
    word_border_probs = torch.nn.Softmax(-1)(borderpreds + torch.log(mask_test.unsqueeze(1)))
    word_border_golds = torch.tensor(wordbordersgold)[devstart:teststart] #unbertmat.gather(1, bordergold)

    pred_start, pred_end = torch.argmax(word_border_probs, 2).split(1, dim=1)
    gold_start, gold_end = word_border_golds.split(1, dim=1)
    overlap_start = torch.max(pred_start, gold_start)
    overlap_end = torch.min(pred_end, gold_end)
    overlap = (overlap_end - overlap_start).float().clamp_min(0).sum()
    expected = (gold_end - gold_start).float().clamp_min(1e-6).sum()
    predicted = (pred_end - pred_start).float().clamp_min(1e-6).sum()
    recall = overlap / expected
    precision = overlap / predicted
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-6)
    print("Dataset-wide F1, precision and recall:")
    print(f1.item(), precision.item(), recall.item())

    pred_start, pred_end = torch.argmax(word_border_probs, 2).split(1, dim=1)
    gold_start, gold_end = word_border_golds.split(1, dim=1)
    overlap_start = torch.max(pred_start, gold_start)
    overlap_end = torch.min(pred_end, gold_end)
    overlap = (overlap_end - overlap_start).float().clamp_min(0)
    recall = overlap / (gold_end - gold_start).float().clamp_min(1e-6)
    precision = overlap / (pred_end - pred_start).float().clamp_min(1e-6)
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-6)
    print((recall > 1).nonzero(), (precision > 1).nonzero())
    recall = recall.mean()
    precision = precision.mean()
    acc = (f1 == 1).float().mean()
    f1 = f1.mean()
    print("Averaged F1, precision and recall:")
    print(f1.item(), precision.item(), recall.item())
    print("Span accuracy")
    print(acc)


def run_borders(p="exp_bert_both_23",
                which="dev",
                qp="../../data/buboqa/data/processed_simplequestions_dataset/all.txt",
                dp="../../data/buboqa/data/bertified_dataset_new.npz",
                ):
    """ Convert wordpiece level borders to wordlevel borders, check with available names """
    # region load data
    berttok = BertTokenizer.from_pretrained("bert-base-uncased")
    borderpreds = torch.tensor(np.load(os.path.join(p, "borderpreds.{}.npy".format(which))))
    print(borderpreds.shape)
    data = np.load(dp)
    print(data.keys())
    devstart, teststart = data["devstart"], data["teststart"]
    print(teststart)
    tokmat = data["tokmat"]
    unbertmat = data["unbertmat"]
    questions = open(qp, encoding="utf8").readlines()
    # endregion

    if which == "dev":
        slicer = slice(devstart, teststart)
    else:
        slicer = slice(teststart, None)
    tokmat_test = torch.tensor(tokmat[slicer]).long()
    unbert_test = torch.tensor(unbertmat[slicer]).long()
    _questions_test = questions[slicer]

    questions_test = [["[CLS]"] + qe.split("\t")[5].split() + ["[SEP]"] for qe in _questions_test]
    borders_test_gold = [["O"] + qe.split("\t")[6].split() + ["O"] for qe in _questions_test]
    uris_gold = [qe.split("\t")[1] for qe in _questions_test]
    maxlen = max([len(qte) for qte in questions_test])
    print("maxlen", maxlen)

    mask_test = (tokmat_test != 0).float()
    borderprobs = torch.nn.Softmax(-1)(borderpreds + torch.log(mask_test.unsqueeze(1)))

    word_border_probs = torch.zeros(len(questions_test), 2, maxlen+1, dtype=torch.float)
    word_border_probs.scatter_add_(2, unbert_test.unsqueeze(1).repeat(1, 2, 1), borderprobs)
    word_border_probs = word_border_probs[:, :, 1:]

    def debug_print(k = 9):
        tbtoks = berttok.convert_ids_to_tokens([xe for xe in tokmat_test[k].cpu().numpy() if xe != 0])
        tbunbert = list(unbert_test[k])[:len(tbtoks)]
        tbstartprobs = list(borderprobs[k, 0].numpy())[:len(tbtoks)]
        tbendprobs = list(borderprobs[k, 1].numpy())[:len(tbtoks)]
        tbstartbars = [q.percentagebar(float(xe)) for xe in tbstartprobs]
        tbendbars = [q.percentagebar(float(xe)) for xe in tbendprobs]
        print(tabulate([tbtoks, tbunbert, tbstartbars, tbendbars]))
        # print(questions_test[k])
        # print(word_border_probs[0])
        twtoks = questions_test[k]
        twstartprobs = list(word_border_probs[k, 0].numpy())[:len(twtoks)]
        twendprobs = list(word_border_probs[k, 1].numpy())[:len(twtoks)]
        twstartbars = [q.percentagebar(float(xe)) for xe in twstartprobs]
        twendbars = [q.percentagebar(float(xe)) for xe in twendprobs]
        print(tabulate([twtoks, twstartbars, twendbars, borders_test_gold[k]]))
    # print(borderprobs[1])
    debug_print(9)

    index = build_entity_index(testsearch=False)

    def valid_startend(x, q):
        if x is None:       return False
        if x[0] >= x[1]:    return False
        # entname = " ".join(q[x[0]:x[1]])
        # if not (entname in entnameset): return False
        return True

    errors = set()
    nooverlap = set()
    someoverlap = set()
    at1 = set()
    at5 = set()
    at20 = set()
    at50 = set()
    at150 = set()
    notfound = set()
    invalid_startends = []
    # empty_startends = []
    allcands = []
    for i in tqdm(range(len(questions_test))):
        question = questions_test[i]
        startend = torch.einsum("a,b->ab", word_border_probs[i, 0], word_border_probs[i, 1])
        startend_xy = startend.numpy().nonzero()
        startend = startend.numpy()
        startend_probs = startend[startend_xy]
        startend = zip(*(startend_xy + (startend_probs,)))
        startend = sorted(startend, key=lambda x: x[2], reverse=True)

        best_startend = None
        j = 0
        while j < len(startend) and \
                not valid_startend(best_startend, question):
            best_startend = startend[j]
            j += 1
        if j > 1:
            invalid_startends.append(i)
        start, end, prob = best_startend
        # print(i, start, end)
        if start > end:
            debug_print(i)
        # assert(start < end)

        # compare with gold
        gold_io = borders_test_gold[i]
        gold_io = [0 if xe == "O" else 1 for xe in gold_io]
        pred_io = [0 for _ in gold_io]
        pred_io[start:end] = [1] * (end-start)
        if not (pred_io == gold_io):
            errors.add(i)
        if not any([ae * be for ae, be in zip(gold_io, pred_io)]):
            nooverlap.add(i)
        if i not in nooverlap and i in errors:
            someoverlap.add(i)
            # raise q.SumTingWongException()

        # recalls
        mention = " ".join(question[start:end])
        searchres = index.search(mention, top=200)
        at1ent, at5ents, at20ents, at50ents, at150ents = set(), set(), set(), set(), set()
        cands = []

        for searchres_i in searchres:
            uri = searchres_i["entry"]["uri"]
            if len(at1ent) < 1:
                at1ent.add(uri)
            if len(at5ents) < 5:
                at5ents.add(uri)
            if len(at20ents) < 20:
                at20ents.add(uri)
            if len(at50ents) < 50:
                at50ents.add(uri)
            if len(at150ents) < 150:
                if uri not in at150ents:
                    cands.append(searchres_i)
                at150ents.add(uri)
        allcands.append(cands)
        if not uris_gold[i] in at1ent:
            at1.add(i)
        if not uris_gold[i] in at5ents:
            at5.add(i)
        if not uris_gold[i] in at20ents:
            at20.add(i)
        if not uris_gold[i] in at50ents:
            at50.add(i)
        if not uris_gold[i] in at150ents:
            at150.add(i)
            # debug_print(i)
            # print(_questions_test[i])

    print("{:.4} % corrected (end before start)".format(100 * len(invalid_startends) / len(questions_test)))
    print("{:.4} % accuracy after postprocessing".format(100*(1 - (len(errors)/ len(questions_test)))))
    print("{:.4} % overlap after postprocessing".format(100*(1 - (len(nooverlap)/ len(questions_test)))))
    print("{:.4} % R@1".format(100*(1 - (len(at1)/ len(questions_test)))))
    print("{:.4} % R@5".format(100*(1 - (len(at5)/ len(questions_test)))))
    print("{:.4} % R@20".format(100*(1 - (len(at20)/ len(questions_test)))))
    print("{:.4} % R@50".format(100*(1 - (len(at50)/ len(questions_test)))))
    print("{:.4} % R@150".format(100*(1 - (len(at150)/ len(questions_test)))))
    # for k in range(50):
    #     debug_print(list(someoverlap)[k])
    #     print(_questions_test[list(someoverlap)[k]])

    with open(os.path.join(p, "entcands.{}.pkl".format(which)), "wb") as f:
        pkl.dump(allcands, f)


if __name__ == '__main__':
    # build_entity_bloom()
    # q.argprun(get_dsF1_wordlevel)
    q.argprun(run_borders)
    # test_index()
    # build_entity_index(testsearch=True)