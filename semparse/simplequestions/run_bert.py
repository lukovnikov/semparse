import qelos as q
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam
import torch
from tabulate import tabulate
from torch.utils.data import TensorDataset, DataLoader
from functools import partial


def load_data(p="../../data/buboqa/data/bertified_dataset.npz",
              which="span/io",
              retrelD = False,
              retrelcounts = False):
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
    tt.tick("printing some examples")
    for k in range(10):
        print("\nExample {}".format(k))
        pp(k)
    tt.tock("printed some examples")

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

    selected = [torch.tensor(data[sel]) for sel in selection]
    tselected = [sel[:devstart] for sel in selected]
    vselected = [sel[devstart:teststart] for sel in selected]
    xselected = [sel[teststart:] for sel in selected]
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


class IOSpanDetector(torch.nn.Module):
    def __init__(self, bert, dropout=0., **kw):
        super(IOSpanDetector, self).__init__(**kw)
        self.bert = bert
        dim = self.bert.config.hidden_size
        self.lin = torch.nn.Linear(dim, dim)
        self.act = torch.nn.Tanh()
        self.linout = torch.nn.Linear(dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.actout = torch.nn.Sigmoid()

    def forward(self, x):       # x: (batsize, seqlen) ints
        mask = (x != 0).long()
        a, _ = self.bert(x.long(), attention_mask=mask, output_all_encoded_layers=False)
        a = self.dropout(a)
        b = self.act(self.lin(a))
        logits = self.linout(b).squeeze(-1)
        return logits


def test_io_span_detector():
    x = torch.randint(1, 800, (5, 10))
    x[0, 6:] = 0
    bert = BertModel.from_pretrained("bert-base-uncased")
    m = IOSpanDetector(bert)
    y = m(x)
    print(y)


def run_span_io(lr=0.001,
                dropout=.3,
                batsize=10,
                epochs=10,
                cuda=False,
                gpu=0
                ):
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
    # endregion

    # region model
    tt.tick("loading BERT")
    bert = BertModel.from_pretrained("bert-base-uncased")
    spandet = IOSpanDetector(bert)
    spandet.to(device)
    tt.tock("loaded BERT")
    # endregion

    # region training
    optim = BertAdam(spandet.parameters(), lr=lr)
    losses = [AutomaskedBCELoss(), AutomaskedBinarySeqAccuracy()]
    trainlosses = [q.LossWrapper(l) for l in losses]
    devlosses = [q.LossWrapper(l) for l in losses]
    testlosses = [q.LossWrapper(l) for l in losses]
    trainloop = partial(q.train_epoch, model=spandet, dataloader=trainloader, optim=optim, losses=trainlosses, device=device)
    devloop = partial(q.test_epoch, model=spandet, dataloader=devloader, losses=devlosses, device=device)
    testloop = partial(q.test_epoch, model=spandet, dataloader=testloader, losses=testlosses, device=device)

    tt.tick("training")
    q.run_training(trainloop, devloop)
    tt.tock("done training")
    # endregion


if __name__ == '__main__':
    # test_io_span_detector()
    q.argprun(run_span_io)