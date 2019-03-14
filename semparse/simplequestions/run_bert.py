import qelos as q
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam, \
    WarmupCosineSchedule, WarmupConstantSchedule, WarmupLinearSchedule, LRSchedule, \
    WarmupCosineWithRestartsSchedule
import torch
from tabulate import tabulate
from torch.utils.data import TensorDataset, DataLoader
from functools import partial
import os
import json


DEFAULT_LR=0.0001
DEFAULT_BATSIZE=10
DEFAULT_EPOCHS=6
DEFAULT_INITWREG=0.
DEFAULT_WREG=0.01
DEFAULT_SMOOTHING=0.


def load_data(p="../../data/buboqa/data/bertified_dataset.npz",
              which="span/io",
              retrelD = False,
              retrelcounts = False,
              datafrac=1.):
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
    tt.tick("loading saved np mats")
    data = np.load(p)
    print(data.keys())
    relD = data["relD"].item()
    revrelD = {v: k for k, v in relD.items()}
    devstart = data["devstart"]
    teststart = data["teststart"]
    tt.tock("mats loaded")
    tt.tick("loading BERT tokenizer")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tt.tock("done")

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

    tokmat = data["tokmat"]

    selected = [torch.tensor(data[sel]).long() for sel in selection]
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
    def __init__(self, bert, dropout=0., extra=False, **kw):
        super(BorderSpanDetector, self).__init__(**kw)
        self.bert = bert
        dim = self.bert.config.hidden_size
        if extra:
            self.lin = torch.nn.Linear(dim, dim)
            self.act = torch.nn.Tanh()
        self.extra = extra
        self.linstart = torch.nn.Linear(dim, 1)
        self.linend = torch.nn.Linear(dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.actout = torch.nn.Sigmoid()

    def forward(self, x):       # x: (batsize, seqlen) ints
        mask = (x != 0).long()
        maxlen = mask.sum(1).max().item()
        maxlen = min(x.size(1), maxlen + 1)
        mask = mask[:, :maxlen]
        x = x[:, :maxlen]
        a, _ = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)
        a = self.dropout(a)
        if self.extra:
            a = self.act(self.lin(a))
        logits_start = self.linstart(a)
        logits_end = self.linend(a)
        logits = torch.cat([logits_start.transpose(1, 2), logits_end.transpose(1, 2)], 1)
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


def run_span_borders(lr=DEFAULT_LR,
                dropout=.5,
                wreg=DEFAULT_WREG,
                initwreg=DEFAULT_INITWREG,
                batsize=DEFAULT_BATSIZE,
                epochs=DEFAULT_EPOCHS,
                smoothing=DEFAULT_SMOOTHING,
                cuda=False,
                gpu=0,
                balanced=False,
                warmup=-1.,
                sched="ang",
                savep="exp_bert_span_borders_",
                freezeemb=False,
                ):
    settings = locals().copy()
    print(locals())
    if cuda:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    # region data
    tt = q.ticktock("script")
    tt.msg("running span border with BERT")
    tt.tick("loading data")
    data = load_data(which="span/borders")
    trainds, devds, testds = data
    tt.tock("data loaded")
    tt.msg("Train/Dev/Test sizes: {} {} {}".format(len(trainds), len(devds), len(testds)))
    trainloader = DataLoader(trainds, batch_size=batsize, shuffle=True)
    devloader = DataLoader(devds, batch_size=batsize, shuffle=False)
    testloader = DataLoader(testds, batch_size=batsize, shuffle=False)
    evalds = TensorDataset(*testloader.dataset.tensors[:-1])
    evalloader = DataLoader(evalds, batch_size=batsize, shuffle=False)
    evalds_dev = TensorDataset(*devloader.dataset.tensors[:-1])
    evalloader_dev = DataLoader(evalds_dev, batch_size=batsize, shuffle=False)
    # endregion

    # region model
    tt.tick("loading BERT")
    bert = BertModel.from_pretrained("bert-base-uncased")
    spandet = BorderSpanDetector(bert, dropout=dropout)
    spandet.to(device)
    tt.tock("loaded BERT")
    # endregion

    # region training
    totalsteps = len(trainloader) * epochs
    params = []
    for paramname, param in spandet.named_parameters():
        if paramname.startswith("bert.embeddings.word_embeddings"):
            if not freezeemb:
                params.append(param)
        else:
            params.append(param)
    optim = BertAdam(params, lr=lr, weight_decay=wreg, warmup=warmup, t_total=totalsteps,
                     schedule=schedmap[sched])
    losses = [q.SmoothedCELoss(smoothing=smoothing), SpanF1Borders(reduction="none"), q.SeqAccuracy()]
    xlosses = [q.SmoothedCELoss(smoothing=smoothing), SpanF1Borders(reduction="none"), q.SeqAccuracy()]
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
        # save test predictions
        testpreds = q.eval_loop(spandet, evalloader, device=device)
        testpreds = testpreds[0].cpu().detach().numpy()
        np.save(os.path.join(savedir, "borderpreds.test.npy"), testpreds)
        # save dev predictions
        testpreds = q.eval_loop(spandet, evalloader_dev, device=device)
        testpreds = testpreds[0].cpu().detach().numpy()
        np.save(os.path.join(savedir, "borderpreds.dev.npy"), testpreds)
        tt.tock("done")
    # endregion


class RelationClassifier(torch.nn.Module):
    def __init__(self, bert, relD, dropout=0., extra=False, **kw):
        super(RelationClassifier, self).__init__(**kw)
        self.bert = bert
        dim = self.bert.config.hidden_size
        if extra:
            self.lin = torch.nn.Linear(dim, dim)
            self.act = torch.nn.Tanh()
        self.extra = extra
        self.relD = relD
        numrels = max(relD.values()) + 1
        self.linout = torch.nn.Linear(dim, numrels)
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.actout = torch.nn.Sigmoid()

    def forward(self, x):       # x: (batsize, seqlen) ints
        mask = (x != 0)
        maxlen = mask.long().sum(1).max().item()
        maxlen = min(x.size(1), maxlen + 1)
        mask = mask[:, :maxlen]
        x = x[:, :maxlen]
        mask = mask.long()
        _, a = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)
        a = self.dropout(a)
        if self.extra:
            a = self.act(self.lin(a))
        logits = self.linout(a)
        return logits


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


def run_relations(lr=DEFAULT_LR,
                dropout=.5,
                wreg=DEFAULT_WREG,
                initwreg=DEFAULT_INITWREG,
                batsize=DEFAULT_BATSIZE,
                epochs=10,
                smoothing=DEFAULT_SMOOTHING,
                cuda=False,
                gpu=0,
                balanced=False,
                maskentity=False,
                warmup=-1.,
                sched="ang",
                savep="exp_bert_rels_",
                test=False,
                freezeemb=False,
                ):
    settings = locals().copy()
    if test:
        epochs=0
    print(locals())
    if cuda:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    # region data
    tt = q.ticktock("script")
    tt.msg("running relation classifier with BERT")
    tt.tick("loading data")
    data = load_data(which="rel+borders", retrelD=True)
    trainds, devds, testds, relD = data
    if maskentity:
        trainds, devds, testds = replace_entity_span(trainds, devds, testds)
    else:
        trainds, devds, testds = [TensorDataset(ds.tensors[0], ds.tensors[2]) for ds in [trainds, devds, testds]]
    tt.tock("data loaded")
    tt.msg("Train/Dev/Test sizes: {} {} {}".format(len(trainds), len(devds), len(testds)))
    trainloader = DataLoader(trainds, batch_size=batsize, shuffle=True)
    devloader = DataLoader(devds, batch_size=batsize, shuffle=False)
    testloader = DataLoader(testds, batch_size=batsize, shuffle=False)
    evalds = TensorDataset(*testloader.dataset.tensors[:1])
    evalloader = DataLoader(evalds, batch_size=batsize, shuffle=False)
    evalds_dev = TensorDataset(*devloader.dataset.tensors[:1])
    evalloader_dev = DataLoader(evalds_dev, batch_size=batsize, shuffle=False)
    if test:
        evalloader = DataLoader(TensorDataset(*evalloader.dataset[:10]),
                                batch_size=batsize, shuffle=False)
        testloader = DataLoader(TensorDataset(*testloader.dataset[:10]),
                                batch_size=batsize, shuffle=False)
    # endregion

    # region model
    tt.tick("loading BERT")
    bert = BertModel.from_pretrained("bert-base-uncased")
    m = RelationClassifier(bert, relD, dropout=dropout)
    m.to(device)
    tt.tock("loaded BERT")
    # endregion

    # region training
    totalsteps = len(trainloader) * epochs

    params = []
    for paramname, param in m.named_parameters():
        if paramname.startswith("bert.embeddings.word_embeddings"):
            if not freezeemb:
                params.append(param)
        else:
            params.append(param)
    optim = BertAdam(params, lr=lr, weight_decay=wreg, warmup=warmup, t_total=totalsteps,
                     schedule=schedmap[sched], init_weight_decay=initwreg)
    losses = [q.SmoothedCELoss(smoothing=smoothing), q.Accuracy()]
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


class BordersAndRelationClassifier(torch.nn.Module):
    def __init__(self, bert, relD, dropout=0., mask_entity_mention=False, extra=False, **kw):
        assert(mask_entity_mention is False)
        super(BordersAndRelationClassifier, self).__init__(**kw)
        self.bert = bert
        dim = self.bert.config.hidden_size
        self.extra = extra
        self.dropout = torch.nn.Dropout(p=dropout)
        # relation classification
        self.mask_entity_mention = mask_entity_mention
        if extra:
            self.rlin = torch.nn.Linear(dim, dim)
            self.ract = torch.nn.Tanh()
        self.relD = relD
        numrels = max(relD.values()) + 1
        self.rlinout = torch.nn.Linear(dim, numrels)
        self.clip_len = True
        # endregion
        # region border classification
        if extra:
            self.blin = torch.nn.Linear(dim, dim)
            self.bact = torch.nn.Tanh()
        self.blinstart = torch.nn.Linear(dim, 1)
        self.blinend = torch.nn.Linear(dim, 1)
        # endregion
        # self.actout = torch.nn.Sigmoid()
        # self.relation_model = RelationClassifier(bert, relD, dropout=dropout, mask_entity_mention=False)
        # self.border_model = BorderSpanDetector(bert, dropout=dropout, extra=False)

    def forward(self, x, spanio=None):
        # region shared
        mask = (x != 0).long()
        if self.clip_len:
            maxlen = mask.sum(1).max().item()
            maxlen = min(x.size(1), maxlen + 1)
            mask = mask[:, :maxlen]
            x = x[:, :maxlen]
        all, pool = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)

        # endregion
        # region relation prediction
        assert(self.mask_entity_mention is False)
        pool = self.dropout(pool)
        if self.extra:
            pool = self.ract(self.rlin(pool))
        rlogits = self.rlinout(pool)
        # endregion

        # region border prediction
        all = self.dropout(all)
        if self.extra:
            all = self.bact(self.blin(all))
        blogits_start = self.blinstart(all)
        blogits_end = self.blinend(all)
        blogits = torch.cat([blogits_start.transpose(1, 2),
                             blogits_end.transpose(1, 2)], 1)
        # endregion
        return blogits, rlogits


class BordersAndRelationLosses(torch.nn.Module):
    def __init__(self, m, cesmoothing=0., **kw):
        super(BordersAndRelationLosses, self).__init__(**kw)
        self.m = m
        self.blosses = [q.SmoothedCELoss(smoothing=cesmoothing, reduction="none"),
                        q.SeqAccuracy(reduction="none"),
                        SpanF1Borders(reduction="none")]
        self.rlosses = [q.SmoothedCELoss(smoothing=cesmoothing, reduction="none"),
                        q.Accuracy(reduction="none")]

    def forward(self, toks, io, borders, rels):
        borderpreds, relpreds = self.m(toks)
        borderces = self.blosses[0](borderpreds, borders)   # (batsize, 2)
        borderces = borderces.mean(1)
        borderaccs = self.blosses[1](borderpreds, borders)   # (batsize, 2)
        borderf1 = self.blosses[2](borderpreds, borders)
        relces = self.rlosses[0](relpreds, rels)
        relaccs = self.rlosses[1](relpreds, rels)
        bothacc = (relaccs.long() & borderaccs.long()).float()
        allces = borderces + relces
        return [allces, borderces, relces, borderf1, borderaccs, relaccs, bothacc]


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


def run_both(lr=DEFAULT_LR,
                dropout=.5,
                wreg=DEFAULT_WREG,
                initwreg=DEFAULT_INITWREG,
                batsize=DEFAULT_BATSIZE,
                evalbatsize=None,
                epochs=10,
                smoothing=DEFAULT_SMOOTHING,
                cuda=False,
                gpu=0,
                balanced=False,
                maskmention=False,
                warmup=-1.,
                sched="ang",
                cycles=-1.,
                savep="exp_bert_both_",
                test=False,
                freezeemb=False,
                large=False,
                datafrac=1.,
                ):
    settings = locals().copy()
    print(locals())
    tt = q.ticktock("script")
    if evalbatsize is None:
        evalbatsize = batsize
    tt.msg("running borders and rel classifier with BERT")
    if test:
        epochs=0
    if cuda:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")
    if cycles == -1:
        if sched == "cos":
            cycles = 0.5
        elif sched == "cosrestart":
            cycles = 1.0

    # region data
    tt.tick("loading data")
    data = load_data(which="all", retrelD=True, datafrac=datafrac)
    trainds, devds, testds, relD = data
    tt.tock("data loaded")
    tt.msg("Train/Dev/Test sizes: {} {} {}".format(len(trainds), len(devds), len(testds)))
    trainloader = DataLoader(trainds, batch_size=batsize, shuffle=True)
    devloader = DataLoader(devds, batch_size=evalbatsize, shuffle=False)
    testloader = DataLoader(testds, batch_size=evalbatsize, shuffle=False)
    evalds = TensorDataset(*testloader.dataset.tensors[:1])
    evalds_dev = TensorDataset(*devloader.dataset.tensors[:1])
    evalloader = DataLoader(evalds, batch_size=evalbatsize, shuffle=False)
    evalloader_dev = DataLoader(evalds_dev, batch_size=evalbatsize, shuffle=False)
    if test:
        evalloader = DataLoader(TensorDataset(*evalloader.dataset[:10]),
                                batch_size=batsize, shuffle=False)
        testloader = DataLoader(TensorDataset(*testloader.dataset[:10]),
                                batch_size=batsize, shuffle=False)
    # endregion

    # region model
    tt.tick("loading BERT")
    whichbert = "bert-base-uncased"
    if large:
        whichbert = "bert-large-uncased"
    bert = BertModel.from_pretrained(whichbert)
    m = BordersAndRelationClassifier(bert, relD, dropout=dropout, mask_entity_mention=maskmention)
    m.to(device)
    tt.tock("loaded BERT")
    # endregion

    # region training
    totalsteps = len(trainloader) * epochs
    assert(initwreg == 0.)
    initl2penalty = InitL2Penalty(bert, factor=q.hyperparam(initwreg))

    params = []
    for paramname, param in m.named_parameters():
        if paramname.startswith("bert.embeddings.word_embeddings"):
            if not freezeemb:
                params.append(param)
        else:
            params.append(param)
    sched = get_schedule(sched, warmup=warmup, t_total=totalsteps, cycles=cycles)
    optim = BertAdam(params, lr=lr, weight_decay=wreg, schedule=sched)
    tmodel = BordersAndRelationLosses(m, cesmoothing=smoothing)
    # xmodel = BordersAndRelationLosses(m, cesmoothing=smoothing)
    # losses = [q.SmoothedCELoss(smoothing=smoothing), q.Accuracy()]
    # xlosses = [q.SmoothedCELoss(smoothing=smoothing), q.Accuracy()]
    tlosses = [q.SelectedLinearLoss(i) for i in range(7)]
    xlosses = [q.SelectedLinearLoss(i) for i in range(7)]
    trainlosses = [q.LossWrapper(l) for l in tlosses]
    devlosses = [q.LossWrapper(l) for l in xlosses]
    testlosses = [q.LossWrapper(l) for l in xlosses]
    trainloop = partial(q.train_epoch, model=tmodel, dataloader=trainloader, optim=optim, losses=trainlosses, device=device)
    devloop = partial(q.test_epoch, model=tmodel, dataloader=devloader, losses=devlosses, device=device)
    testloop = partial(q.test_epoch, model=tmodel, dataloader=testloader, losses=testlosses, device=device)

    tt.tick("training")
    m.clip_len = True
    q.run_training(trainloop, devloop, max_epochs=epochs)
    tt.tock("done training")

    tt.tick("testing")
    testres = testloop()
    print(testres)
    settings["testres"] = testres
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
        m.clip_len = False
        # TEST data
        testpreds = q.eval_loop(m, evalloader, device=device)
        borderpreds = testpreds[0].cpu().detach().numpy()
        relpreds = testpreds[1].cpu().detach().numpy()
        np.save(os.path.join(savedir, "borderpreds.test.npy"), borderpreds)
        np.save(os.path.join(savedir, "relpreds.test.npy"), relpreds)
        # DEV data
        testpreds = q.eval_loop(m, evalloader_dev, device=device)
        borderpreds = testpreds[0].cpu().detach().numpy()
        relpreds = testpreds[1].cpu().detach().numpy()
        np.save(os.path.join(savedir, "borderpreds.dev.npy"), borderpreds)
        np.save(os.path.join(savedir, "relpreds.dev.npy"), relpreds)
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
    # q.argprun(run_span_borders)
    q.argprun(run_both)
    # q.argprun(run_relations)