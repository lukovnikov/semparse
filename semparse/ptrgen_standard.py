import qelos as q
import torch
from functools import partial
import numpy as np
from semparse.rnn import *
from semparse import attention
import semparse.trees.parse as tparse

# import spacy
# nlp = spacy.load("C:\Users\Denis\Miniconda3\envs\torch\lib\site-packages\en_core_web_sm")
# doc = nlp("what is the capital of states that have cities named durham ?")
# print(doc)


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
        # self.dec.cell.core[-1].y_tm1 = states[-1][0].squeeze(1)
        # self.dec.cell.core[-1].c_tm1 = states[-1][1].squeeze(1)
        self.dec.cell.out.ctx_ids = inpseq
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
        # self.dec.cell.core[-1].y_tm1 = states[-1][0].squeeze(1)
        # self.dec.cell.core[-1].c_tm1 = states[-1][1].squeeze(1)
        self.dec.cell.out.ctx_ids = inpseq
        outprobs = self.dec(_outseq, ctx=ctx, ctx_mask=ctx_mask)
        outprobs = outprobs[:, :outseq.size(1)]
        return outprobs


def gen_datasets(which="geo"):
    pprefix = "../data/"
    if which == "geo":
        pprefix = pprefix + "geoqueries/jia2016/"
        trainp = pprefix + "train.txt"
        validp = pprefix + "test.txt"
        testp = pprefix + "test.txt"
    elif which == "atis":
        pprefix += "atis/jia2016/"
        trainp = pprefix + "train.txt"
        validp = pprefix + "dev.txt"
        testp = pprefix + "test.txt"
    elif which == "jobs":
        assert(False) # jia didn't do jobs
        pprefix += "jobqueries"
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
    trainwords = set()
    trainwordcounts = {}
    testwords = set()
    trainwords_fl = set()
    trainwordcounts_fl = {}
    testwords_fl = set()
    with open(trainp) as tf, open(validp) as vf, open(testp) as xf:
        for line in tf:
            line_nl, line_fl = line.strip().split("\t")
            line_fl = line_fl.replace("' ", "")
            # line_nl = " ".join(line_nl.split(" ")[::-1])
            nlsm.add(line_nl)
            flsm.add(line_fl)
            trainwords |= set(line_nl.split())
            for word in set(line_nl.split()):
                if word not in trainwordcounts:
                    trainwordcounts[word] = 0
                trainwordcounts[word] += 1
            trainwords_fl |= set(line_fl.split())
            for word in set(line_fl.split()):
                if word not in trainwordcounts_fl:
                    trainwordcounts_fl[word] = 0
                trainwordcounts_fl[word] += 1
            i += 1
        devstart = i
        for line in vf:
            line_nl, line_fl = line.strip().split("\t")
            line_fl = line_fl.replace("' ", "")
            # line_nl = " ".join(line_nl.split(" ")[::-1])
            nlsm.add(line_nl)
            flsm.add(line_fl)
            i += 1
        teststart = i
        for line in xf:
            line_nl, line_fl = line.strip().split("\t")
            line_fl = line_fl.replace("' ", "")
            # line_nl = " ".join(line_nl.split(" ")[::-1])
            nlsm.add(line_nl)
            flsm.add(line_fl)
            testwords |= set(line_nl.split())
            testwords_fl |= set(line_fl.split())
            i += 1
    nlsm.finalize()
    flsm.finalize()

    # region get gate sup
    gatesups = torch.zeros(flsm.matrix.shape[0], flsm.matrix.shape[1]+1, dtype=torch.long)
    for i in range(nlsm.matrix.shape[0]):
        nl_sent = nlsm[i].split()
        fl_sent = flsm[i].split()
        inid = False
        for j, fl_sent_token in enumerate(fl_sent):
            if re.match("_\w+id", fl_sent_token):
                inid = True
            elif fl_sent_token == ")":
                inid = False
            elif fl_sent_token == "(":
                pass
            else:
                if inid:
                    if fl_sent_token in nl_sent:
                        gatesups[i, j] = 1




    # endregion

    # region print analysis
    print("{} unique words in train, {} unique words in test, {} in test but not in train"
          .format(len(trainwords), len(testwords), len(testwords - trainwords)))
    print(testwords - trainwords)
    trainwords_once = set([k for k, v in trainwordcounts.items() if v < 2])
    print("{} unique words in train that occur only once ({} of them is in test)".format(len(trainwords_once), len(trainwords_once & testwords)))
    print(trainwords_once)
    trainwords_twice = set([k for k, v in trainwordcounts.items() if v < 3])
    print("{} unique words in train that occur only twice ({} of them is in test)".format(len(trainwords_twice), len(trainwords_twice & testwords)))
    rarerep = trainwords_once | (testwords - trainwords)
    print("{} unique rare representation words".format(len(rarerep)))
    print(rarerep)

    trainwords_fl_once = set([k for k, v in trainwordcounts_fl.items() if v < 2])
    rarerep_fl = trainwords_fl_once | (testwords_fl - trainwords_fl)
    print("{} unique rare rep words in logical forms".format(len(rarerep_fl)))
    print(rarerep_fl)
    # endregion

    # endregion create datasets
    nlmat = torch.tensor(nlsm.matrix).long()
    flmat = torch.tensor(flsm.matrix).long()
    gold = torch.tensor(flsm.matrix[:, 1:]).long()
    gold = torch.cat([gold, torch.zeros_like(gold[:, 0:1])], 1)
    tds = torch.utils.data.TensorDataset(nlmat[:devstart], flmat[:devstart], gold[:devstart], gatesups[:devstart][:, 1:])
    vds = torch.utils.data.TensorDataset(nlmat[devstart:teststart], flmat[devstart:teststart], gold[devstart:teststart])
    xds = torch.utils.data.TensorDataset(nlmat[teststart:], flmat[teststart:], gold[teststart:])
    # endregion
    return (tds, vds, xds), nlsm.D, flsm.D, rarerep, rarerep_fl


class TreeAccuracyPrologPar(torch.nn.Module):
    def __init__(self, flD, reduction="mean", **kw):
        super(TreeAccuracyPrologPar, self).__init__(**kw)
        self.flD = flD
        self.rflD = {v: k for k, v in flD.items()}
        self.parsef = tparse.parse_prolog
        self.masktoken = "<MASK>"
        self.reduction = reduction

    def forward(self, probs, gold):     # probs: (batsize, seqlen, outvocsize)
        # build gold trees
        def parsetree(row):
            row = row.cpu().numpy()
            tree = ["_answer"] + [self.rflD[x] for x in row if self.rflD[x] != self.masktoken]
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


class TrainModel(torch.nn.Module):
    """
    Does the normal losses specified, adds a penalty for supervising pointergen gate
    """
    def __init__(self, model, losses, gate_pw=1., **kw):
        super(TrainModel, self).__init__(**kw)
        self.model = model
        self.losses = [q.LossWrapper(l) for l in losses]

    def forward(self, x, y, g, gs):
        p = self.model(x, y)        # (batsize, seqlen, vocsize)
        lossvals = [loss(p, g) for loss in self.losses]
        # compute gate


def run_normal(lr=0.001,
               gradclip=5.,
               batsize=20,
               epochs=150,
               embdim=100,
               encdim=200,
               numlayer=1,
               cuda=False,
               gpu=0,
               wreg=1e-8,
               dropout=0.5,
               smoothing=0.,
               goldsmoothing=-0.1,
               selfptr=False,
               which="geo"):
    tt = q.ticktock("script")
    tt.msg("running normal att")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)

    # region data
    tt.tick("generating data")
    # dss, D = gen_sort_data(seqlen=seqlen, numvoc=numvoc, numex=numex, prepend_inp=False)
    dss, nlD, flD, rare_nl, rare_fl = gen_datasets(which=which)
    tloader, vloader, xloader = [torch.utils.data.DataLoader(ds, batch_size=batsize, shuffle=True) for ds in dss]
    seqlen = len(dss[0][0][1])
    # merge nlD into flD and make mapper
    nextflDid = max(flD.values()) + 1
    sourcemap = torch.zeros(len(nlD), dtype=torch.long, device=device)
    for k, v in nlD.items():
        if k not in flD:
            flD[k] = nextflDid
            nextflDid += 1
        sourcemap[v] = flD[k]
    tt.tock("data generated")
    # endregion

    # region model
    tt.tick("building model")
    # source side
    inpemb = q.UnkReplWordEmb(embdim, worddic=nlD, unk_tokens=rare_nl)
    encdims = [encdim] * numlayer
    encoder = q.LSTMEncoder(embdim, *encdims, bidir=True, dropout_in_shared=dropout)

    # target side
    decemb = q.UnkReplWordEmb(embdim, worddic=flD, unk_tokens=rare_fl)
    decinpdim = embdim
    decdims = [decinpdim] + [encdim] * numlayer
    dec_core = torch.nn.Sequential(
        *[q.rnn.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout) for i in range(1, len(decdims))]
    )
    att = attention.FwdAttention(decdims[-1], encdim * 2, decdims[-1])
    out = torch.nn.Sequential(
        q.UnkReplWordLinout(decdims[-1]+encdim*2, worddic=flD, unk_tokens=rare_fl),
        # torch.nn.Softmax(-1)
    )
    if selfptr:
        outgate = PointerGeneratorOutGate(decdims[-1] + encdim * 2, encdim, 3)
        out = SelfPointerGeneratorOut(out, sourcemap=sourcemap, gate=outgate)
        selfatt = attention.FwdAttention(decdims[-1], decdims[-1], decdims[-1])
        deccell = SelfPointerGeneratorCell(emb=decemb, core=dec_core, att=att, selfatt=selfatt, out=out)
    else:
        outgate = PointerGeneratorOutGate(decdims[-1] + encdim * 2, encdim, 0)
        out = PointerGeneratorOut(out, sourcemap=sourcemap, gate=outgate)
        deccell = PointerGeneratorCell(emb=decemb, core=dec_core, att=att, out=out)
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
        ce = q.loss.CELoss(mode="probs", ignore_index=0)
    elif goldsmoothing < 0.:
        ce = q.loss.SmoothedCELoss(mode="probs", ignore_index=0, smoothing=smoothing)
    else:
        ce = q.loss.DiffSmoothedCELoss(mode="probs", ignore_index=0, alpha=goldsmoothing, beta=smoothing)
    acc = q.loss.SeqAccuracy(ignore_index=0)
    elemacc = q.loss.SeqElemAccuracy(ignore_index=0)
    trainmodel = TrainModel(train_encdec, [ce, elemacc, acc])
    treeacc = TreeAccuracyPrologPar(flD=flD)
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


if __name__ == '__main__':
    q.argprun(run_normal)