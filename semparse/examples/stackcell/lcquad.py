import torch
import qelos as q
import numpy as np
import json
import os
import re
import semparse
from functools import partial


class Node(object):
    NONLEAF=1
    LEAF=0
    REDUCE=2
    def __init__(self, name, children=None, **kw):
        super(Node, self).__init__(**kw)
        self.name = name
        self.children = children

    def __repr__(self):
        return f"{self.name}({','.join([str(x) for x in self.children])})" if self.children is not None else self.name

    def __str__(self):
        return repr(self)

    def to_transitions(self):
        que = [self]
        ret = []
        while len(que) > 0:
            head = que.pop(0)
            if head == "<RED>":
                ret.append(head)
            else:
                ret.append(head.name)
                if head.children is not None:
                    assert(len(head.children) > 0)
                    que = head.children + ["<RED>"] + que
        return " ".join(ret)

    @classmethod
    def from_transitions(cls, x, tok2act=None):  # sequence of tokens and transition action dictionary
        # produces tree
        que = x.strip().split()
        acts = [tok2act[x] for x in que]
        stack = []
        while len(que) > 0:
            x, a = que.pop(0), acts.pop(0)
            if a in (Node.LEAF, Node.NONLEAF):
                stack.append((Node(x), a))
            elif a == Node.REDUCE:
                children = []
                y, b = stack.pop(-1)
                while b != Node.NONLEAF:
                    children.append(y)
                    y, b = stack.pop(-1)
                y.children = children[::-1]
                stack.append((y, Node.LEAF))
        assert(len(stack) == 1)
        return stack[-1][0]



def fql2tree(x):
    uris = list(set(re.findall("<[^>]+>", x)))
    uriD = dict(zip(uris, [f"<URI-{i}>" for i in range(len(uris))]))
    RuriD = {v: k for k, v in uriD.items()}
    for k, v in uriD.items():
        x = x.replace(k, v)
    tokens = re.split("([\(\),])", x)
    stack = []
    for token in tokens:
        token = token.strip()
        if token in RuriD:
            token = RuriD[token]
        if token == "(":
            stack[-1].children = []     # should have children
        elif token == ")":
            children = []
            while stack[-1].children != []:
                children.append(stack.pop(-1))
            stack[-1].children = children[::-1]
        elif token == ",":
            pass
        elif token == "":
            pass
        else:
            stack.append(Node(token))
    assert(len(stack) == 1)
    return stack[-1]


def ent2placeholder(x):
    placeholders = {}
    que = [x]
    while len(que) > 0:
        head = que.pop(0)
        if head.name in ["find", "in"] and head.children is not None and len(head.children) > 0 \
                and re.match("<([^>]+)>", head.children[0].name):
            if head.children[0].name not in placeholders:
                placeholders[head.children[0].name] = f"<ENT-{len(placeholders)}>"
            head.children[0].name = placeholders[head.children[0].name]
        que += head.children if head.children is not None else []
    return x


def load_data(p="../../../data/lcquad-fql/"):
    trainp = os.path.join(p, "train.json")
    testp = os.path.join(p, "test.json")
    print(f"Loading data from: '{trainp}' (train) and '{testp}' (test)")
    traindata = json.load(open(trainp))
    print(f"Number of training examples: {len(traindata)}")
    testdata = json.load(open(testp))
    print(f"Number of test examples: {len(testdata)}")

    # process logical forms
    # parse to trees, replace entities with placeholders in queries
    tdata = [(example["question"], ent2placeholder(fql2tree(example["logical_form"])))
             for example in traindata]
    xdata = [(example["question"], ent2placeholder(fql2tree(example["logical_form"])))
             for example in testdata]

    # get node types that have children
    parentnodes = set()
    for (_, e) in tdata+xdata:
        que = [e]
        while len(que) > 0:
            head = que.pop(0)
            if head.children is not None:
                assert(len(head.children) > 0)
                parentnodes.add(head.name)
                que += head.children

    print(f"Types of nodes that have children ({len(parentnodes)}): \n{parentnodes}")

    # build string matrices
    teststart = len(tdata)
    xsm = q.StringMatrix(indicate_start_end=True)
    ysm = q.StringMatrix(indicate_start=True)
    ysm.tokenize = lambda x: x.split()
    for question, l in tdata+xdata:
        xsm.add(question)
        ysm.add(l.to_transitions() + " <MASK>")
    xsm.finalize()
    ysm.finalize()

    tok2act = {}
    for tok in ysm.D:
        if tok == "<RED>":
            tok2act[ysm.D[tok]] = 2
        elif tok in parentnodes:
            tok2act[ysm.D[tok]] = 1
        else:
            tok2act[ysm.D[tok]] = 0
    return xsm, ysm, teststart, tok2act


class Seq2Seq(torch.nn.Module):
    def __init__(self, xemb, xenc, dec, test=False, **kw):
        super(Seq2Seq, self).__init__(**kw)
        self.xemb, self.xenc, self.dec = xemb, xenc, dec
        self.test = test

    def forward(self, x, y):
        xemb, xmask = self.xemb(x)
        xenc = self.xenc(xemb, mask=xmask)
        xmask = xmask[:, :xenc.size(1)]

        self.dec.cell.save_ctx(xenc, xmask)
        if self.test:
            ylen = y.size(1)
            y = y[:, 0]

        ret = self.dec(y)

        if self.test:
            ret = ret[:, :ylen]
        return ret


def run_seq2seq(lr=0.001,
                batsize=128,
                evalbatsize=256,
                epochs=100,
                embdim=50,
                encdim=100,
                decdim=100,
                enclayers=2,
                declayers=2,
                dropout=.0,
                wreg=1e-6,
                cuda=False,
                gpu=0,
                ):
    args = locals().copy()
    q.pikax.optimize(parameters=[
        q.pikax.ChoiceHP("lr", [1e-3, 1e-4, 1e-5]),
        q.pikax.RangeHP("dropout", .05, .5),
        q.pikax.ChoiceHP("embdim", [64, 128, 256, 512], type="int"),
        q.pikax.ChoiceHP("encdim", [128, 256, 512], type="int"),
        q.pikax.ChoiceHP("batsize", [32, 64, 128], type="int"),
        q.pikax.ChoiceHP("numlayers", [1, 2, 3]),
        q.pikax.RangeHP("wreg", 1e-2, 1e-9, log_scale=True)
    ],
        minimize=False,
        evaluation_function=partial(run_seq2seq_, **args),
        savep="train_seq2seq_best_hp.json")


def run_seq2seq_(lr=0.001,
                batsize=32,
                evalbatsize=256,
                epochs=100,
                warmup=5,
                embdim=50,
                encdim=100,
                numlayers=2,
                dropout=.0,
                wreg=1e-6,
                cuda=False,
                gpu=0,
        ):
    settings = locals().copy()
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt = q.ticktock("script")
    tt.msg("running seq2seq on LC-QuAD")

    tt.tick("loading data")
    xsm, ysm, teststart, tok2act = load_data()
    _tok2act = {ysm.RD[k]: v for k, v in tok2act.items()}

    print("Some examples:")
    for i in range(5):
        print(f"{xsm[i]}\n ->{ysm[i]}\n -> {Node.from_transitions(' '.join(ysm[i].split()[1:]), _tok2act)}")

    print("Non-leaf tokens:")
    print({ysm.RD[k]: v for k, v in tok2act.items() if v > 0})

    devstart = teststart - 500
    trainds = torch.utils.data.TensorDataset(torch.tensor(xsm.matrix[:devstart]).long(),
                                             torch.tensor(ysm.matrix[:devstart, :-1]).long(),
                                             torch.tensor(ysm.matrix[:devstart, 1:]).long())
    valds =   torch.utils.data.TensorDataset(torch.tensor(xsm.matrix[devstart:teststart]).long(),
                                             torch.tensor(ysm.matrix[devstart:teststart, :-1]).long(),
                                             torch.tensor(ysm.matrix[devstart:teststart, 1:]).long())
    testds =  torch.utils.data.TensorDataset(torch.tensor(xsm.matrix[teststart:]).long(),
                                             torch.tensor(ysm.matrix[teststart:, :-1]).long(),
                                             torch.tensor(ysm.matrix[teststart:, 1:]).long())
    tt.msg(f"Data splits: train: {len(trainds)}, valid: {len(valds)}, test: {len(testds)}")

    tloader = torch.utils.data.DataLoader(trainds, batch_size=batsize, shuffle=True)
    vloader = torch.utils.data.DataLoader(valds, batch_size=evalbatsize, shuffle=False)
    xloader = torch.utils.data.DataLoader(testds, batch_size=evalbatsize, shuffle=False)
    tt.tock("data loaded")

    # model
    enclayers, declayers = numlayers, numlayers
    decdim = encdim
    xemb = q.WordEmb(embdim, worddic=xsm.D)
    yemb = q.WordEmb(embdim, worddic=ysm.D)
    encdims = [embdim] + [encdim//2] * enclayers
    xenc = q.LSTMEncoder(embdim, *encdims[1:], bidir=True, dropout_in_shared=dropout)
    decdims = [embdim] + [decdim] * declayers
    dec_core = torch.nn.Sequential(*[q.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout, dropout_rec=dropout)
                                     for i in range(1, len(decdims))])
    yout = q.WordLinout(encdim+decdim, worddic=ysm.D)
    dec_cell = semparse.rnn.LuongCell(emb=yemb, core=dec_core, out=yout, dropout=dropout)
    decoder = q.TFDecoder(dec_cell)
    testdecoder = q.FreeDecoder(dec_cell, maxtime=100)

    m = Seq2Seq(xemb, xenc, decoder)
    testm = Seq2Seq(xemb, xenc, testdecoder, test=True)

    # test model
    tt.tick("running a batch")
    test_y = m(*iter(tloader).next()[:-1])
    q.batch_reset(m)
    test_y = testm(*iter(vloader).next()[:-1])
    q.batch_reset(m)
    tt.tock(f"ran a batch: {test_y.size()}")

    optim = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wreg)
    tlosses = [q.CELoss(mode="logits", ignore_index=0), q.Accuracy(ignore_index=0), q.SeqAccuracy(ignore_index=0)]
    xlosses = [q.CELoss(mode="logits", ignore_index=0), q.Accuracy(ignore_index=0), q.SeqAccuracy(ignore_index=0)]
    tlosses = [q.LossWrapper(l) for l in tlosses]
    vlosses = [q.LossWrapper(l) for l in xlosses]
    xlosses = [q.LossWrapper(l) for l in xlosses]
    trainloop = partial(q.train_epoch, model=m, dataloader=tloader, optim=optim, losses=tlosses, device=device)
    devloop = partial(q.test_epoch, model=testm, dataloader=vloader, losses=vlosses, device=device)
    testloop = partial(q.test_epoch, model=testm, dataloader=xloader, losses=xlosses, device=device)

    lrplateau = q.util.ReduceLROnPlateau(optim, mode="max",
        factor=.1, patience=3, cooldown=1, warmup=warmup, threshold=0., verbose=True, eps=1e-9)
    on_after_valid = [lambda: lrplateau.step(vlosses[1].get_epoch_error())]
    _devloop = partial(devloop, on_end=on_after_valid)
    stoptrain = [lambda : all([pg["lr"] <= 1e-7 for pg in optim.param_groups])]

    tt.tick("training")
    q.run_training(trainloop, _devloop, max_epochs=epochs, check_stop=stoptrain)
    tt.tock("done training")

    tt.tick("testing")
    testres = testloop()
    print(testres)
    settings["testres"] = testres
    tt.tock("tested")

    devres = devloop()
    print(devres, vlosses[0].get_epoch_error())

    return vlosses[1].get_epoch_error()


if __name__ == '__main__':
    q.argprun(run_seq2seq)