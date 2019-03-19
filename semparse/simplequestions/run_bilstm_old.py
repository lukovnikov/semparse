import qelos as q
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert.optimization import *
import torch
from tabulate import tabulate
from torch.utils.data import TensorDataset, DataLoader
from functools import partial
import os
import json
from tqdm import tqdm
import pickle as pkl


DEFAULT_LR=0.0001
DEFAULT_BATSIZE=10
DEFAULT_EPOCHS=6
DEFAULT_INITWREG=0.
DEFAULT_WREG=0.01
DEFAULT_SMOOTHING=0.


def load_word_mat(origp="../../data/buboqa/data/processed_simplequestions_dataset/"):
    outp = os.path.join(origp, "all.pkl")
    generate = True
    if generate:
        trainp = os.path.join(origp, "train.txt")
        validp = os.path.join(origp, "valid.txt")
        testp = os.path.join(origp, "test.txt")
        trainlines = open(trainp, encoding="utf8").readlines()
        validlines = open(validp, encoding="utf8").readlines()
        testlines = open(testp, encoding="utf8").readlines()
        sm = q.StringMatrix()
        sm.tokenize = lambda x: x.split()

        i = 0
        for line in tqdm(trainlines):
            sm.add(line.split("\t")[5])
            i += 1
        devstart = i
        for line in tqdm(validlines):
            sm.add(line.split("\t")[5])
            i += 1
        teststart = i
        for line in tqdm(testlines):
            sm.add(line.split("\t")[5])

        sm.finalize()
        print(len(sm.D))
        print(sm[0])
        pkl.dump((sm.matrix, sm.D, (devstart, teststart)), open(outp, "wb"))
    else:
        wordmat, wordD, (devstart, teststart) = pkl.load(open(outp, "rb"))
        return wordmat, wordD, (devstart, teststart)



def load_data(p="../../data/buboqa/data/bertified_dataset.npz",
              which="span/io",
              retrelD = False,
              retrelcounts = False,
              rettokD = False,
              datafrac=1.,
              wordlevel=False):
    """
    :param p:       where the stored matrices are
    :param which:   which data to include in output datasets
                        "span/io": O/I annotated spans,
                        "span/borders": begin and end positions of span
                        "rel+io": what relation (also gives "spanio" outputs to give info where entity is supposed to be (to ignore it))
                        "rel+borders": same, but gives "spanborders" instead
                        "all": everything
    :return:
    """
    tt = q.ticktock("dataloader")
    if wordlevel:
        tt.tick("loading original data word-level stringmatrix")
        wordmat, wordD, (word_devstart, word_teststart) = load_word_mat()
        twordmat, vwordmat, xwordmat = wordmat[:word_devstart], wordmat[word_devstart:word_teststart], wordmat[word_teststart:]
        tt.tock("loaded stringmatrix")
    tt.tick("loading saved np mats")
    data = np.load(p)
    print(data.keys())
    relD = data["relD"].item()
    revrelD = {v: k for k, v in relD.items()}
    devstart = data["devstart"]
    teststart = data["teststart"]
    if wordlevel:
        assert(devstart == word_devstart)
        assert(teststart == word_teststart)
    tt.tock("mats loaded")
    tt.tick("loading BERT tokenizer")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tt.tock("done")

    if wordlevel:
        tokD = wordD
    else:
        tokD = tokenizer.vocab

    def pp(i):
        tokrow = data["tokmat"][i]
        iorow = [xe - 1 for xe in data["iomat"][i] if xe != 0]
        ioborderrow = data["ioborders"][i]
        rel_i = data["rels"][i]
        tokrow = tokenizer.convert_ids_to_tokens([tok for tok in tokrow if tok != 0])
        # print(" ".join(tokrow))
        print(tabulate([range(len(tokrow)), tokrow, iorow]))
        print(ioborderrow)
        print(revrelD[rel_i])
    # tt.tick("printing some examples")
    # for k in range(10):
    #     print("\nExample {}".format(k))
    #     pp(k)
    # tt.tock("printed some examples")

    # datasets
    tt.tick("making datasets")
    if which == "span/io":
        selection = ["tokmat", "iomat"]
    elif which == "span/borders":
        selection = ["tokmat", "ioborders"]
    elif which == "rel+io":
        selection = ["tokmat", "iomat", "rels"]
    elif which == "rel+borders":
        selection = ["tokmat", "ioborders", "rels"]
    elif which == "all":
        selection = ["tokmat", "iomat", "ioborders", "rels"]
    else:
        raise q.SumTingWongException("unknown which mode: {}".format(which))

    if wordlevel:
        tokmat = wordmat
    else:
        tokmat = data["tokmat"]

    selected = [torch.tensor(data[sel] if sel != "tokmat" else tokmat).long() for sel in selection]
    tselected = [sel[:devstart] for sel in selected]
    vselected = [sel[devstart:teststart] for sel in selected]
    xselected = [sel[teststart:] for sel in selected]

    if datafrac <= 1.:
        # restrict data such that least relations are unseen
        # get relation counts
        trainrels = data["rels"][:devstart]
        uniquerels, relcounts = np.unique(data["rels"][:devstart], return_counts=True)
        relcountsD = dict(zip(uniquerels, relcounts))
        relcounter = dict(zip(uniquerels, [0]*len(uniquerels)))
        totalcap = int(datafrac * len(trainrels))
        capperrel = max(relcountsD.values())

        def numberexamplesincluded(capperrel_):
            numberexamplesforcap = np.clip(relcounts, 0, capperrel_).sum()
            return numberexamplesforcap

        while capperrel > 0:        # TODO do binary search
            numexcapped = numberexamplesincluded(capperrel)
            if numexcapped <= totalcap:
                break
            capperrel -= 1

        print("rel count cap is {}".format(capperrel))

        remainids = []
        for i in range(len(trainrels)):
            if len(remainids) >= totalcap:
                break
            if relcounter[trainrels[i]] > capperrel:
                pass
            else:
                relcounter[trainrels[i]] += 1
                remainids.append(i)
        print("{}/{} examples retained".format(len(remainids), len(trainrels)))
        tselected_new = [sel[remainids] for sel in tselected]
        if datafrac == 1.:
            for a, b in zip(tselected_new, tselected):
                assert(np.all(a == b))
        tselected = tselected_new

    traindata = TensorDataset(*tselected)
    devdata = TensorDataset(*vselected)
    testdata = TensorDataset(*xselected)

    ret = (traindata, devdata, testdata)
    if retrelD:
        ret += (relD,)
    if rettokD:
        ret += (tokD,)
    if retrelcounts:
        ret += data["relcounts"]
    tt.tock("made datasets")
    return ret


class AutomaskedBCELoss(torch.nn.Module):
    def __init__(self, mode="logits", weight=None, reduction="mean",
                 pos_weight=None, maskid=0, trueid=2,
                 **kw):
        """

        :param mode:        "logits" or "probs". If "probs", pos_weight must be None
        :param weight:
        :param reduction:
        :param pos_weight:
        :param maskid:
        :param trueid:
        :param kw:
        """
        super(AutomaskedBCELoss, self).__init__(**kw)
        self.mode = mode
        if mode == "logits":
            self.loss = torch.nn.BCEWithLogitsLoss(weight=weight, reduction="none", pos_weight=pos_weight)
        elif mode == "probs":
            assert(pos_weight is None)
            self.loss = torch.nn.BCELoss(weight=weight, reduction="none")
        else:
            raise q.SumTingWongException("unknown mode: {}".format(mode))
        self.reduction = reduction
        self.maskid, self.trueid = maskid, trueid

    def forward(self, pred, gold):
        # map gold
        mask = (gold != self.maskid).float()
        realgold = (gold == self.trueid).float()
        l = self.loss(pred, realgold)
        l = l * mask

        if self.reduction == "sum":
            ret = l.sum()
        elif self.reduction == "mean":
            ret = l.mean() * (np.prod(gold.size()) / mask.sum())
        else:
            ret = l
        return ret


class AutomaskedBinarySeqAccuracy(torch.nn.Module):
    def __init__(self, mode="logits", threshold=0.5, reduction="mean",
                 maskid=0, trueid=2, **kw):
        super(AutomaskedBinarySeqAccuracy, self).__init__(**kw)
        self.reduction = reduction
        self.maskid = maskid
        self.trueid = trueid
        self.mode = mode
        self.threshold = threshold
        self.act = torch.nn.Sigmoid() if mode == "logits" else None

    def forward(self, pred, gold):
        if self.act is not None:
            pred = self.act(pred)
        pred = (pred > self.threshold)

        mask = (gold != self.maskid)
        realgold = (gold == self.trueid)

        same = (pred == realgold)
        same = (same | ~mask).long()

        same = same.sum(1)      # sum along seq dimension
        same = (same == gold.size(1)).float()

        if self.reduction == "sum":
            ret = same.sum()
        elif self.reduction == "mean":
            ret = same.mean()
        else:
            ret = same
        return ret


class SpanF1Borders(torch.nn.Module):
    def __init__(self, reduction="mean", **kw):
        super(SpanF1Borders, self).__init__(**kw)
        self.reduction = reduction

    def forward(self, pred, gold):      # pred: (batsize, 2, seqlen) probs, gold: (batsize, 2)
        pred_start, pred_end = torch.argmax(pred, 2).split(1, dim=1)
        gold_start, gold_end = gold.split(1, dim=1)
        overlap_start = torch.max(pred_start, gold_start)
        overlap_end = torch.min(pred_end, gold_end)
        overlap = (overlap_end - overlap_start).float().clamp_min(0)
        recall = overlap / (gold_end - gold_start).float().clamp_min(1e-6)
        precision = overlap / (pred_end - pred_start).float().clamp_min(1e-6)
        f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-6)

        if self.reduction == "sum":
            ret = f1.sum()
        elif self.reduction == "mean":
            ret = f1.mean()
        else:
            ret = f1
        return ret

class InitL2Penalty(q.PenaltyGetter):
    def __init__(self, model, factor=1., reduction="mean"):
        super(InitL2Penalty, self).__init__(model, factor=factor, reduction=reduction)
        initweight_dict = dict(model.named_parameters())
        self.weight_names = sorted(initweight_dict.keys())
        with torch.no_grad():
            self.initweights = torch.cat([initweight_dict[x].detach().flatten()
                                     for x in self.weight_names], 0)

    def forward(self, *_, **__):
        if q.v(self.factor) > 0:
            weight_dict = dict(self.model.named_parameters())
            weights = torch.cat([weight_dict[x].flatten() for x in self.weight_names], 0)
            penalty = torch.norm(weights - self.initweights, p=2)
            ret = penalty * q.v(self.factor)
        else:
            ret = 0
        return ret



class IOSpanDetector(torch.nn.Module):
    def __init__(self, bert, dropout=0., extra=False, **kw):
        super(IOSpanDetector, self).__init__(**kw)
        self.bert = bert
        dim = self.bert.config.hidden_size
        if extra:
            self.lin = torch.nn.Linear(dim, dim)
            self.act = torch.nn.Tanh()
        self.extra = extra
        self.linout = torch.nn.Linear(dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.actout = torch.nn.Sigmoid()

    def forward(self, x):       # x: (batsize, seqlen) ints
        mask = (x != 0).long()
        a, _ = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)
        a = self.dropout(a)
        if self.extra:
            a = self.act(self.lin(a))
        logits = self.linout(a).squeeze(-1)
        return logits


def test_io_span_detector():
    x = torch.randint(1, 800, (5, 10))
    x[0, 6:] = 0
    bert = BertModel.from_pretrained("bert-base-uncased")
    m = IOSpanDetector(bert)
    y = m(x)
    print(y)


class BorderSpanDetector(torch.nn.Module):
    def __init__(self, emb, bilstm, dim, dropout=0., extra=False, **kw):
        super(BorderSpanDetector, self).__init__(**kw)
        self.emb = emb
        self.bilstm = bilstm
        if extra:
            self.lin = torch.nn.Linear(dim, dim)
            self.act = torch.nn.Tanh()
        self.extra = extra
        self.linstart = torch.nn.Linear(dim, 1)
        self.linend = torch.nn.Linear(dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.actout = torch.nn.Sigmoid()
        self.outlen = None

    def forward(self, x):       # x: (batsize, seqlen) ints
        xemb = self.emb(x)
        mask = (x != 0).float()
        a = self.bilstm(xemb, mask=mask)
        a = self.dropout(a)
        if self.extra:
            a = self.act(self.lin(a))
        logits_start = self.linstart(a)
        logits_end = self.linend(a)
        logits = torch.cat([logits_start.transpose(1, 2), logits_end.transpose(1, 2)], 1)
        if self.outlen is not None:
            logits = torch.cat([logits,
                                torch.zeros(logits.size(0), 2, self.outlen - logits.size(2),
                                    device=logits.device,
                                    dtype=logits.dtype)],
                               2)
        return logits


schedmap = {
    "ang": "warmup_linear",
    "lin": "warmup_constant",
    "cos": "warmup_cosine"
}


def run_span_io(lr=DEFAULT_LR,
                dropout=.5,
                wreg=DEFAULT_WREG,
                batsize=DEFAULT_BATSIZE,
                epochs=DEFAULT_EPOCHS,
                cuda=False,
                gpu=0,
                balanced=False,
                warmup=-1.,
                sched="ang",  # "lin", "cos"
                ):
    settings = locals().copy()
    print(locals())
    if cuda:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    # region data
    tt = q.ticktock("script")
    tt.msg("running span io with BERT")
    tt.tick("loading data")
    data = load_data(which="span/io")
    trainds, devds, testds = data
    tt.tock("data loaded")
    tt.msg("Train/Dev/Test sizes: {} {} {}".format(len(trainds), len(devds), len(testds)))
    trainloader = DataLoader(trainds, batch_size=batsize, shuffle=True)
    devloader = DataLoader(devds, batch_size=batsize, shuffle=False)
    testloader = DataLoader(testds, batch_size=batsize, shuffle=False)
    # compute balancing hyperparam for BCELoss
    trainios = trainds.tensors[1]
    numberpos = (trainios == 2).float().sum()
    numberneg = (trainios == 1).float().sum()
    if balanced:
        pos_weight = (numberneg/numberpos)
    else:
        pos_weight = None
    # endregion

    # region model
    tt.tick("loading BERT")
    bert = BertModel.from_pretrained("bert-base-uncased")
    spandet = IOSpanDetector(bert, dropout=dropout)
    spandet.to(device)
    tt.tock("loaded BERT")
    # endregion

    # region training
    totalsteps = len(trainloader) * epochs
    optim = BertAdam(spandet.parameters(), lr=lr, weight_decay=wreg, warmup=warmup, t_total=totalsteps, schedule=schedmap[sched])
    losses = [AutomaskedBCELoss(pos_weight=pos_weight), AutomaskedBinarySeqAccuracy()]
    trainlosses = [q.LossWrapper(l) for l in losses]
    devlosses = [q.LossWrapper(l) for l in losses]
    testlosses = [q.LossWrapper(l) for l in losses]
    trainloop = partial(q.train_epoch, model=spandet, dataloader=trainloader, optim=optim, losses=trainlosses, device=device)
    devloop = partial(q.test_epoch, model=spandet, dataloader=devloader, losses=devlosses, device=device)
    testloop = partial(q.test_epoch, model=spandet, dataloader=testloader, losses=testlosses, device=device)

    tt.tick("training")
    q.run_training(trainloop, devloop, max_epochs=epochs)
    tt.tock("done training")

    tt.tick("testing")
    testres = testloop()
    print(testres)
    tt.tock("tested")
    # endregion


def get_schedule(sched=None, warmup=-1, t_total=-1, cycles=None):
    if sched == "none" or sched is None:
        schedule = LRSchedule(warmup=warmup, t_total=t_total)
    elif sched == "lin":
        schedule = WarmupConstantSchedule(warmup=warmup, t_total=t_total)
    elif sched == "ang":
        schedule = WarmupLinearSchedule(warmup=warmup, t_total=t_total)
    elif sched == "cos":
        schedule = WarmupCosineSchedule(warmup=warmup, t_total=t_total, cycles=cycles)
    elif sched == "cosrestart":
        schedule = WarmupCosineWithRestartsSchedule(warmup=warmup, t_total=t_total, cycles=cycles)
    else:
        raise Exception("unknown schedule '{}'".format(sched))
    return schedule


def replace_entity_span(*dss):
    print("replacing entity span")
    berttok = BertTokenizer.from_pretrained("bert-base-uncased")
    maskid = berttok.vocab["[MASK]"]
    padid = berttok.vocab["[PAD]"]
    outdss = []
    for ds in dss:
        tokmat, borders, rels = ds.tensors
        outtokmat = torch.ones_like(tokmat) * padid
        for i in range(len(tokmat)):
            k = 0
            for j in range(tokmat.size(1)):
                if borders[i][0] == j:
                    outtokmat[i, k] = maskid
                    k += 1
                elif borders[i][0] < j < borders[i][1]:
                    pass
                elif tokmat[i, j] == padid:
                    break
                else:
                    outtokmat[i, k] = tokmat[i, j]
                    k += 1
        outds = torch.utils.data.TensorDataset(outtokmat, rels)
        outdss.append(outds)
    print("replaced entity span")
    return outdss


def run_span_borders(lr=DEFAULT_LR,
                dropout=.3,
                wreg=DEFAULT_WREG,
                initwreg=DEFAULT_INITWREG,
                batsize=DEFAULT_BATSIZE,
                evalbatsize=-1,
                epochs=DEFAULT_EPOCHS,
                smoothing=DEFAULT_SMOOTHING,
                dim=200,
                numlayers=1,
                cuda=False,
                gpu=0,
                savep="exp_bilstm_span_borders_",
                datafrac=1.,
                vanillaemb=False,
                embdim=300,
                sched="cos",
                warmup=0.1,
                cycles=0.5,
                ):
    settings = locals().copy()
    print(locals())
    if evalbatsize < 0:
        evalbatsize = batsize
    if cuda:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    # region data
    tt = q.ticktock("script")
    tt.msg("running span border with BiLSTM")
    tt.tick("loading data")
    data = load_data(which="span/borders", datafrac=datafrac)
    trainds, devds, testds = data
    tt.tock("data loaded")
    tt.msg("Train/Dev/Test sizes: {} {} {}".format(len(trainds), len(devds), len(testds)))
    trainloader = DataLoader(trainds, batch_size=batsize, shuffle=True)
    devloader = DataLoader(devds, batch_size=evalbatsize, shuffle=False)
    testloader = DataLoader(testds, batch_size=evalbatsize, shuffle=False)
    evalds = TensorDataset(*testloader.dataset.tensors[:-1])
    evalloader = DataLoader(evalds, batch_size=evalbatsize, shuffle=False)
    evalds_dev = TensorDataset(*devloader.dataset.tensors[:-1])
    evalloader_dev = DataLoader(evalds_dev, batch_size=evalbatsize, shuffle=False)
    # endregion

    # region model
    tt.tick("creating model")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    emb = bert.embeddings.word_embeddings
    if vanillaemb:
        tt.msg("using vanilla emb of size {}".format(embdim))
        emb = torch.nn.Embedding(emb.weight.size(0), embdim)
    else:
        embdim = bert.config.hidden_size
    # inpD = tokenizer.vocab
    # q.WordEmb.masktoken = "[PAD]"
    # emb = q.WordEmb(embdim, worddic=inpD)
    bilstm = q.rnn.LSTMEncoder(embdim, *([dim] * numlayers), bidir=True, dropout_in_shared=dropout)
    spandet = BorderSpanDetector(emb, bilstm, dim*2, dropout=dropout)
    spandet.to(device)
    tt.tock("model created")
    # endregion

    # region training
    totalsteps = len(trainloader) * epochs
    params = spandet.parameters()
    sched = get_schedule(sched, warmup=warmup, t_total=totalsteps, cycles=cycles)
    optim = BertAdam(params, lr=lr, weight_decay=wreg, schedule=sched)
    # optim = torch.optim.Adam(spandet.parameters(), lr=lr, weight_decay=wreg)
    losses = [q.SmoothedCELoss(smoothing=smoothing), SpanF1Borders(), q.SeqAccuracy()]
    xlosses = [q.SmoothedCELoss(smoothing=smoothing), SpanF1Borders(), q.SeqAccuracy()]
    trainlosses = [q.LossWrapper(l) for l in losses]
    devlosses = [q.LossWrapper(l) for l in xlosses]
    testlosses = [q.LossWrapper(l) for l in xlosses]
    trainloop = partial(q.train_epoch, model=spandet, dataloader=trainloader, optim=optim, losses=trainlosses, device=device)
    devloop = partial(q.test_epoch, model=spandet, dataloader=devloader, losses=devlosses, device=device)
    testloop = partial(q.test_epoch, model=spandet, dataloader=testloader, losses=testlosses, device=device)

    tt.tick("training")
    q.run_training(trainloop, devloop, max_epochs=epochs)
    tt.tock("done training")

    tt.tick("testing")
    testres = testloop()
    print(testres)
    tt.tock("tested")

    if len(savep) > 0:
        tt.tick("making predictions and saving")
        i = 0
        while os.path.exists(savep+str(i)):
            i += 1
        os.mkdir(savep + str(i))
        savedir = savep + str(i)
        # save model
        # torch.save(spandet, open(os.path.join(savedir, "model.pt"), "wb"))
        # save settings
        json.dump(settings, open(os.path.join(savedir, "settings.json"), "w"))

        outlen = trainloader.dataset.tensors[0].size(1)
        spandet.outlen = outlen

        # save test predictions
        testpreds = q.eval_loop(spandet, evalloader, device=device)
        testpreds = testpreds[0].cpu().detach().numpy()
        np.save(os.path.join(savedir, "borderpreds.test.npy"), testpreds)
        # save dev predictions
        testpreds = q.eval_loop(spandet, evalloader_dev, device=device)
        testpreds = testpreds[0].cpu().detach().numpy()
        np.save(os.path.join(savedir, "borderpreds.dev.npy"), testpreds)
        tt.msg("saved in {}".format(savedir))
        tt.tock("done")
    # endregion


class RelationClassifier(torch.nn.Module):
    def __init__(self, emb=None, bilstm=None, dim=-1, relD=None, dropout=0., extra=True, **kw):
        super(RelationClassifier, self).__init__(**kw)
        self.bilstm = bilstm
        self.emb = emb
        self.dim = dim
        if extra:
            self.lin = torch.nn.Linear(dim, dim)
            self.act = torch.nn.ReLU()
        self.extra = extra
        self.relD = relD
        numrels = max(relD.values()) + 1
        self.linout = torch.nn.Linear(dim, numrels)
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.actout = torch.nn.Sigmoid()

    def forward(self, x):       # x: (batsize, seqlen) ints
        xemb = self.emb(x)
        mask = (x != 0)
        # xemb_, unsorter = q.seq_pack(xemb, mask, ret_sorter=False)
        _, a = self.bilstm(xemb, mask=mask, ret_states=True)
        # _, (hn, cn) = self.bilstm(xemb_)      # (numlayers x numdirs, batsize, dim)
        # hn = hn.view(self.bilstm.num_layers, 2, x.size(0), -1)
        # hn = hn[-1, :, :, :].transpose(0, 1).contiguous().view(x.size(0), -1)
        # a = hn[unsorter]
        a = a.view(a.size(0), -1)
        if self.extra:
            a = self.act(self.lin(a))
        a = self.dropout(a)
        logits = self.linout(a)
        return logits


def run_relations(lr=DEFAULT_LR,
                dropout=.3,
                wreg=DEFAULT_WREG,
                initwreg=DEFAULT_INITWREG,
                batsize=DEFAULT_BATSIZE,
                epochs=10,
                smoothing=DEFAULT_SMOOTHING,
                cuda=False,
                gpu=0,
                balanced=False,
                maskentity=False,
                savep="exp_bilstm_rels_",
                test=False,
                datafrac=1.,
                vanillaemb=False,
                gloveemb=True,
                embdim=300,
                dim=300,
                numlayers=2,
                warmup=0.01,
                cycles=0.5,
                sched="cos",
                evalbatsize=-1,
                classweighted=False,
                ):
    print(locals())
    settings = locals().copy()
    if evalbatsize < 0:
        evalbatsize = batsize
    if test:
        epochs=0
    if cuda:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    # region data
    assert(not gloveemb or not vanillaemb)
    tt = q.ticktock("script")
    tt.msg("running relation classifier with BiLSTM")
    tt.tick("loading data")
    data = load_data(which="rel+borders", retrelD=True, datafrac=datafrac, wordlevel=gloveemb, rettokD=True)
    trainds, devds, testds, relD, tokD = data
    if maskentity:
        trainds, devds, testds = replace_entity_span(trainds, devds, testds)
    else:
        trainds, devds, testds = [TensorDataset(ds.tensors[0], ds.tensors[2]) for ds in [trainds, devds, testds]]
    relcounts = torch.zeros(max(relD.values()) + 1)
    trainrelcounts = torch.bincount(trainds.tensors[1])
    relcounts[:len(trainrelcounts)] += trainrelcounts.float()
    tt.tock("data loaded")
    tt.msg("Train/Dev/Test sizes: {} {} {}".format(len(trainds), len(devds), len(testds)))
    trainloader = DataLoader(trainds, batch_size=batsize, shuffle=True)
    devloader = DataLoader(devds, batch_size=evalbatsize, shuffle=False)
    testloader = DataLoader(testds, batch_size=evalbatsize, shuffle=False)
    evalds = TensorDataset(*testloader.dataset.tensors[:1])
    evalloader = DataLoader(evalds, batch_size=evalbatsize, shuffle=False)
    evalds_dev = TensorDataset(*devloader.dataset.tensors[:1])
    evalloader_dev = DataLoader(evalds_dev, batch_size=evalbatsize, shuffle=False)

    if test:
        evalloader = DataLoader(TensorDataset(*evalloader.dataset[:10]),
                                batch_size=batsize, shuffle=False)
        testloader = DataLoader(TensorDataset(*testloader.dataset[:10]),
                                batch_size=batsize, shuffle=False)
    # endregion

    # region model
    tt.tick("making model")
    if vanillaemb:
        bert = BertModel.from_pretrained("bert-base-uncased")
        emb = bert.embeddings.word_embeddings
        tt.msg("using vanilla emb of size {}".format(embdim))
        emb = torch.nn.Embedding(emb.weight.size(0), embdim)
    elif gloveemb:
        emb = q.WordEmb.load_glove("glove.50d", selectD=tokD)
    else:
        bert = BertModel.from_pretrained("bert-base-uncased")
        emb = bert.embeddings.word_embeddings
        embdim = bert.config.hidden_size
    bilstm = q.rnn.LSTMEncoder(embdim, *([dim] * numlayers), bidir=True, dropout_in=dropout)
    # bilstm = torch.nn.LSTM(embdim, dim, batch_first=True, num_layers=numlayers, bidirectional=True, dropout=dropout)
    m = RelationClassifier(emb=emb, bilstm=bilstm, dim=dim*2, relD=relD, dropout=dropout)
    m.to(device)
    tt.tock("made model")
    # endregion

    # region training
    totalsteps = len(trainloader) * epochs
    params = m.parameters()
    sched = get_schedule(sched, warmup=warmup, t_total=totalsteps, cycles=cycles)
    # optim = BertAdam(params, lr=lr, weight_decay=wreg, warmup=warmup, t_total=totalsteps, schedule=schedmap[sched])
    optim = BertAdam(params, lr=lr, weight_decay=wreg, schedule=sched)
    losses = [q.SmoothedCELoss(smoothing=smoothing,
                               weight=1/relcounts.clamp_min(1e-6) if classweighted else None),
              q.Accuracy()]
    xlosses = [q.SmoothedCELoss(smoothing=smoothing), q.Accuracy()]
    trainlosses = [q.LossWrapper(l) for l in losses]
    devlosses = [q.LossWrapper(l) for l in xlosses]
    testlosses = [q.LossWrapper(l) for l in xlosses]
    trainloop = partial(q.train_epoch, model=m, dataloader=trainloader, optim=optim, losses=trainlosses, device=device)
    devloop = partial(q.test_epoch, model=m, dataloader=devloader, losses=devlosses, device=device)
    testloop = partial(q.test_epoch, model=m, dataloader=testloader, losses=testlosses, device=device)

    tt.tick("training")
    q.run_training(trainloop, devloop, max_epochs=epochs)
    tt.tock("done training")

    tt.tick("testing")
    testres = testloop()
    print(testres)
    tt.tock("tested")

    if len(savep) > 0:
        tt.tick("making predictions and saving")
        i = 0
        while os.path.exists(savep+str(i)):
            i += 1
        os.mkdir(savep + str(i))
        savedir = savep + str(i)
        # save model
        # torch.save(m, open(os.path.join(savedir, "model.pt"), "wb"))
        # save settings
        json.dump(settings, open(os.path.join(savedir, "settings.json"), "w"))
        # save relation dictionary
        # json.dump(relD, open(os.path.join(savedir, "relD.json"), "w"))
        # save test predictions
        testpreds = q.eval_loop(m, evalloader, device=device)
        testpreds = testpreds[0].cpu().detach().numpy()
        np.save(os.path.join(savedir, "relpreds.test.npy"), testpreds)
        testpreds = q.eval_loop(m, evalloader_dev, device=device)
        testpreds = testpreds[0].cpu().detach().numpy()
        np.save(os.path.join(savedir, "relpreds.dev.npy"), testpreds)
        tt.msg("saved in {}".format(savedir))
        # save bert-tokenized questions
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # with open(os.path.join(savedir, "testquestions.txt"), "w") as f:
        #     for batch in evalloader:
        #         ques, io = batch
        #         ques = ques.numpy()
        #         for question in ques:
        #             qstr = " ".join([x for x in tokenizer.convert_ids_to_tokens(question) if x != "[PAD]"])
        #             f.write(qstr + "\n")

        tt.tock("done")
    # endregion


if __name__ == '__main__':
    # test_io_span_detector()
    # q.argprun(run_span_io)
    q.argprun(run_span_borders)
    # q.argprun(run_relations)