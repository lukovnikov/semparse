# *
import qelos as q
import torch
from functools import partial
from semparse.attention import *
import numpy as np
from semparse.trees import parse as tparse


class EncDec(torch.nn.Module):
    def __init__(self, inpemb, enc, dec, **kw):
        super(EncDec, self).__init__(**kw)
        self.inpemb, self.enc, self.dec = inpemb, enc, dec
        self.enc.ret_all_states = True

    def forward(self, inpseq, outseq):
        inpemb, ctx_mask = self.inpemb(inpseq)
        ctx, states = self.enc(inpemb, mask=ctx_mask, ret_states=True)
        if ctx_mask is not None and ctx_mask.size(1) > ctx.size(1):
            ctx_mask = ctx_mask[:, :ctx.size(1)]
        self.dec.cell.core[-1].y_tm1 = states[-1][0].squeeze(1)
        self.dec.cell.core[-1].c_tm1 = states[-1][1].squeeze(1)
        outprobs = self.dec(outseq, ctx=ctx, ctx_mask=ctx_mask)
        return outprobs


class Test_EncDec(torch.nn.Module):
    def __init__(self, inpemb, enc, dec, **kw):
        super(Test_EncDec, self).__init__(**kw)
        self.inpemb, self.enc, self.dec = inpemb, enc, dec
        self.enc.ret_all_states = True

    def forward(self, inpseq, outseq):  # (batsize, inpseqlen), (batsize, outseqlen)
        inpemb, ctx_mask = self.inpemb(inpseq)
        ctx, states = self.enc(inpemb, mask=ctx_mask, ret_states=True)
        if ctx_mask is not None and ctx_mask.size(1) > ctx.size(1):
            ctx_mask = ctx_mask[:, :ctx.size(1)]
        _outseq = outseq[:, 0]
        self.dec.cell.core[-1].y_tm1 = states[-1][0].squeeze(1)
        self.dec.cell.core[-1].c_tm1 = states[-1][1].squeeze(1)
        outprobs = self.dec(_outseq, ctx=ctx, ctx_mask=ctx_mask)
        outprobs = outprobs[:, :outseq.size(1)]
        return outprobs


def gen_datasets(which="geo"):
    pprefix = "../data/"
    if which == "geo":
        pprefix = pprefix + "geoqueries/dong2016/"
        trainp = pprefix + "train.txt"
        validp = pprefix + "test.txt"
        testp = pprefix + "test.txt"
    elif which == "atis":
        pprefix += "atis/dong2016/"
        trainp = pprefix + "train.txt"
        validp = pprefix + "dev.txt"
        testp = pprefix + "test.txt"
    elif which == "jobs":
        pprefix += "jobqueries/dong2016/"
        trainp = pprefix + "train.txt"
        validp = pprefix + "test.txt"
        testp = pprefix + "test.txt"
    else:
        raise q.SumTingWongException("unknown dataset")

    nlsm = q.StringMatrix(indicate_start_end=True)
    nlsm.tokenize = lambda x: x.split()
    flsm = q.StringMatrix(indicate_start_end=True if which == "jobs" else False)
    flsm.tokenize = lambda x: x.split()
    devstart, teststart, i = 0, 0, 0
    with open(trainp) as tf, open(validp) as vf, open(testp) as xf:
        for line in tf:
            line_nl, line_fl = line.strip().split("\t")
            line_nl = " ".join(line_nl.split(" ")[::-1])
            nlsm.add(line_nl)
            flsm.add(line_fl)
            i += 1
        devstart = i
        for line in vf:
            line_nl, line_fl = line.strip().split("\t")
            line_nl = " ".join(line_nl.split(" ")[::-1])
            nlsm.add(line_nl)
            flsm.add(line_fl)
            i += 1
        teststart = i
        for line in xf:
            line_nl, line_fl = line.strip().split("\t")
            line_nl = " ".join(line_nl.split(" ")[::-1])
            nlsm.add(line_nl)
            flsm.add(line_fl)
            i += 1
    nlsm.finalize()
    flsm.finalize()

    nlmat = torch.tensor(nlsm.matrix).long()
    flmat = torch.tensor(flsm.matrix).long()
    gold = torch.tensor(flsm.matrix[:, 1:]).long()
    gold = torch.cat([gold, torch.zeros_like(gold[:, 0:1])], 1)
    tds = torch.utils.data.TensorDataset(nlmat[:devstart], flmat[:devstart], gold[:devstart])
    vds = torch.utils.data.TensorDataset(nlmat[devstart:teststart], flmat[devstart:teststart], gold[devstart:teststart])
    xds = torch.utils.data.TensorDataset(nlmat[teststart:], flmat[teststart:], gold[teststart:])
    return (tds, vds, xds), nlsm.D, flsm.D


class TreeAccuracyLambdaDFPar(torch.nn.Module):
    def __init__(self, flD, reduction="mean", **kw):
        super(TreeAccuracyLambdaDFPar, self).__init__(**kw)
        self.flD = flD
        self.rflD = {v: k for k, v in flD.items()}
        self.parsef = tparse.parse_lambda_depth_first_parentheses
        self.masktoken = "<MASK>"
        self.reduction = reduction

    def forward(self, probs, gold):     # probs: (batsize, seqlen, outvocsize)
        # build gold trees
        def parsetree(row):
            row = row.cpu().numpy()
            tree = ["("] + [self.rflD[x] for x in row if self.rflD[x] != self.masktoken]
            numunmatched = tree.count("(") - tree.count(")")
            tree += [")"] * max(0, numunmatched)     # too few --> add some
            if numunmatched < 0:        # too many --> stop where zero depth is reached
                depth = 0
                for i in range(len(tree)):
                    if tree[i] == "(":
                        depth += 1
                    elif tree[i] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    else: pass
                tree = tree[:i+1]
            tree = self.parsef(" ".join(tree))
            return tree
        predseq = torch.argmax(probs, 2)
        predtrees = [parsetree(prow) for prow in predseq]
        goldtrees = [parsetree(grow) for grow in gold]
        same = [float(x == y) for x, y in zip(goldtrees, predtrees)]
        same = torch.tensor(same)

        if self.reduction in ["mean", "elementwise_mean"]:
            samesum = same.sum() / same.size(0)
        elif self.reduction == "sum":
            samesum = same.sum()
        else:
            samesum = same
        return samesum


def run_normal(lr=0.0005,
               gradclip=5.,
               batsize=20,
               epochs=80,
               embdim=200,
               encdim=200,
               numlayer=1,
               cuda=False,
               gpu=0,
               wreg=1e-6,
               dropout=0.5,
               smoothing=0.,
               goldsmoothing=-0.1,
               which="geo",
               relatt=False,
               ):
    tt = q.ticktock("script")
    tt.msg("running normal att")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)

    # region data
    tt.tick("generating data")
    # dss, D = gen_sort_data(seqlen=seqlen, numvoc=numvoc, numex=numex, prepend_inp=False)
    dss, nlD, flD = gen_datasets(which=which)
    tloader, vloader, xloader = [torch.utils.data.DataLoader(ds, batch_size=batsize, shuffle=True) for ds in dss]
    seqlen = len(dss[0][0][1])
    tt.tock("data generated")
    # endregion

    # region model
    tt.tick("building model")
    # source side
    inpemb = q.WordEmb(embdim, worddic=nlD)
    encdims = [encdim] * numlayer
    encoder = q.LSTMEncoder(embdim, *encdims, bidir=False, dropout_in_shared=dropout)

    # target side
    decemb = q.WordEmb(embdim, worddic=flD)
    decinpdim = embdim
    decdims = [decinpdim] + [encdim] * numlayer
    dec_core = torch.nn.Sequential(
        *[q.rnn.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout) for i in range(1, len(decdims))]
    )
    if relatt:
        att = ComboAbsRelAttention(ctxdim=encdim, vecdim=encdim)
    else:
        att = BasicAttention()
    out = torch.nn.Sequential(
        q.WordLinout(encdim, worddic=flD),
        # torch.nn.Softmax(-1)
    )
    merge = q.rnn.FwdDecCellMerge(decdims[-1], encdims[-1], outdim=encdim)
    deccell = q.rnn.DecoderCell(emb=decemb, core=dec_core,
                               att=att, out=out, merge=merge)
    train_dec = q.TFDecoder(deccell)
    test_dec = q.FreeDecoder(deccell, maxtime=seqlen+10)
    train_encdec = EncDec(inpemb, encoder, train_dec)
    test_encdec = Test_EncDec(inpemb, encoder, test_dec)

    train_encdec.to(device)
    test_encdec.to(device)
    tt.tock("built model")
    # endregion

    # region training
    # losses:
    if smoothing == 0:
        ce = q.loss.CELoss(mode="logits", ignore_index=0)
    elif goldsmoothing < 0.:
        ce = q.loss.SmoothedCELoss(mode="logits", ignore_index=0, smoothing=smoothing)
    else:
        ce = q.loss.DiffSmoothedCELoss(mode="logits", ignore_index=0, alpha=goldsmoothing, beta=smoothing)
    acc = q.loss.SeqAccuracy(ignore_index=0)
    elemacc = q.loss.SeqElemAccuracy(ignore_index=0)
    treeacc = TreeAccuracyLambdaDFPar(flD=flD)
    # optim
    optim = torch.optim.Adam(train_encdec.parameters(), lr=lr, weight_decay=wreg)
    clipgradnorm = lambda: torch.nn.utils.clip_grad_value_(train_encdec.parameters(), clip_value=gradclip)
    # lööps
    batchloop = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainloop = partial(q.train_epoch, model=train_encdec, dataloader=tloader, optim=optim, device=device,
                        losses=[q.LossWrapper(ce), q.LossWrapper(elemacc), q.LossWrapper(acc)],
                        print_every_batch=False, _train_batch=batchloop)
    validloop = partial(q.test_epoch, model=test_encdec, dataloader=vloader, device=device,
                        losses=[q.LossWrapper(treeacc)],
                        print_every_batch=False)

    tt.tick("training")
    q.run_training(trainloop, validloop, max_epochs=epochs)
    tt.tock("trained")

    tt.tick("testing")
    test_results = validloop(model=test_encdec, dataloader=xloader)
    print("Test results (freerunning): {}".format(test_results))
    test_results = validloop(model=train_encdec, dataloader=xloader)
    print("Test results (TF): {}".format(test_results))
    tt.tock("tested")
    # endregion
    tt.msg("done")


from semparse.rnn import *


def run_gatedtree(lr=0.01,
               gradclip=5.,
               batsize=20,
               epochs=80,
               embdim=200,
               encdim=200,
               numlayer=1,
               cuda=False,
               gpu=0,
               wreg=1e-8,
               dropout=0.5,
               smoothing=0.4,
               goldsmoothing=-0.1,
               which="geo",
               relatt=False,
               ):
    tt = q.ticktock("script")
    tt.msg("running gated tree decoder")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)

    # region data
    tt.tick("generating data")
    # dss, D = gen_sort_data(seqlen=seqlen, numvoc=numvoc, numex=numex, prepend_inp=False)
    dss, nlD, flD = gen_datasets(which=which)
    tloader, vloader, xloader = [torch.utils.data.DataLoader(ds, batch_size=batsize, shuffle=True) for ds in dss]
    seqlen = len(dss[0][0][1])
    id2pushpop = torch.zeros(len(flD), dtype=torch.long, device=device)
    id2pushpop[flD["("]] = +1
    id2pushpop[flD[")"]] = -1

    tt.tock("data generated")
    # endregion

    # region model
    tt.tick("building model")
    # source side
    inpemb = q.WordEmb(embdim, worddic=nlD)
    encdims = [encdim] * numlayer
    encoder = q.LSTMEncoder(embdim, *encdims, bidir=False, dropout_in_shared=dropout)

    # target side
    decemb = q.WordEmb(embdim, worddic=flD)
    decinpdim = embdim
    decdims = [decinpdim] + [encdim] * numlayer
    dec_core = \
        [GatedTreeLSTMCell(decdims[i-1], decdims[i], dropout_in=dropout) for i in range(1, len(decdims))]        ###
    dec_core = TreeRNNDecoderCellCore(*dec_core)
    if relatt:
        att = ComboAbsRelAttention(ctxdim=encdim, vecdim=encdim)
    else:
        att = BasicAttention()
    out = torch.nn.Sequential(
        q.WordLinout(encdim, worddic=flD),
        # torch.nn.Softmax(-1)
    )
    merge = q.rnn.FwdDecCellMerge(decdims[-1], encdims[-1], outdim=encdim)
    deccell = TreeRNNDecoderCell(emb=decemb, core=dec_core,
                               att=att, out=out, merge=merge, id2pushpop=id2pushpop)
    train_dec = q.TFDecoder(deccell)
    test_dec = q.FreeDecoder(deccell, maxtime=seqlen+10)
    train_encdec = EncDec(inpemb, encoder, train_dec)
    test_encdec = Test_EncDec(inpemb, encoder, test_dec)

    train_encdec.to(device)
    test_encdec.to(device)
    tt.tock("built model")
    # endregion

    # region training
    # losses:
    if smoothing == 0:
        ce = q.loss.CELoss(mode="logits", ignore_index=0)
    elif goldsmoothing < 0.:
        ce = q.loss.SmoothedCELoss(mode="logits", ignore_index=0, smoothing=smoothing)
    else:
        ce = q.loss.DiffSmoothedCELoss(mode="logits", ignore_index=0, alpha=goldsmoothing, beta=smoothing)
    acc = q.loss.SeqAccuracy(ignore_index=0)
    elemacc = q.loss.SeqElemAccuracy(ignore_index=0)
    treeacc = TreeAccuracyLambdaDFPar(flD=flD)
    # optim
    optim = torch.optim.RMSprop(train_encdec.parameters(), lr=lr, alpha=0.95, weight_decay=wreg)
    clipgradnorm = lambda: torch.nn.utils.clip_grad_value_(train_encdec.parameters(), clip_value=gradclip)
    # lööps
    batchloop = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainloop = partial(q.train_epoch, model=train_encdec, dataloader=tloader, optim=optim, device=device,
                        losses=[q.LossWrapper(ce), q.LossWrapper(elemacc), q.LossWrapper(acc)],
                        print_every_batch=False, _train_batch=batchloop)
    validloop = partial(q.test_epoch, model=test_encdec, dataloader=vloader, device=device,
                        losses=[q.LossWrapper(treeacc)],
                        print_every_batch=False)

    tt.tick("training")
    q.run_training(trainloop, validloop, max_epochs=epochs)
    tt.tock("trained")

    tt.tick("testing")
    test_results = validloop(model=test_encdec, dataloader=xloader)
    print("Test results (freerunning): {}".format(test_results))
    test_results = validloop(model=train_encdec, dataloader=xloader)
    print("Test results (TF): {}".format(test_results))
    tt.tock("tested")
    # endregion
    tt.msg("done")


if __name__ == '__main__':
    q.argprun(run_normal)