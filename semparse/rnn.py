import torch
import qelos as q


__all__ = ["PointerGeneratorCell", "PointerGeneratorOutGate", "PointerGeneratorOut"]


class PointerGeneratorOutGate(torch.nn.Module):
    def __init__(self, inpdim, vecdim, outdim=2, **kw):
        super(PointerGeneratorOutGate, self).__init__(**kw)
        self.inpdim, self.vecdim, self.outdim = inpdim, vecdim, outdim
        self.lin1 = torch.nn.Linear(inpdim, vecdim)
        self.act1 = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(vecdim, outdim)
        self.act2 = torch.nn.Softmax(-1)

    def forward(self, x):
        r = self.act1(self.lin1(x))
        o = self.act2(self.lin2(r))
        return o


class PointerGeneratorOut(torch.nn.Module):     # integrates q.rnn.AutoMaskedOut
    """
    Performs the generation of tokens or copying of input.

    """
    def __init__(self, genout=None, sourcemap=None,
                 gate:PointerGeneratorOutGate=None,
                 automasker:q.rnn.AutoMasker=None, **kw):
        super(PointerGeneratorOut, self).__init__(**kw)
        self.genout = genout
        self.sourcemap = sourcemap      # maps from input ids to output ids (all input ids must be part of output dict). must be 1D long tensor containing output dict ids
        self.automasker = automasker    # automasker for masking out invalid tokens
        self.gate = gate                # module that takes in the vector and outputs scores of how to mix the gen and cpy distributions
        self._ctx_ids = None     # must be set in every batch, before decoding, contains mapped input sequence ids

    @property
    def ctx_ids(self):
        return self._ctx_ids

    @ctx_ids.setter
    def ctx_ids(self, value):
        self._ctx_ids = self.sourcemap[value]       # already maps to output dict when setting ctx_ids

    def batch_reset(self):
        self._ctx_ids = None

    def update(self, x):        # from automasker
        if self.automasker is not None:
            self.automasker.update(x)

    def forward(self, x, scores=None):
        """
        :param x:       vector for generation
        :param scores:  attention scores (unnormalized)     (batsize, seqlen)
        :return:        probabilities over output tokens
        """
        assert(self._ctx_ids is not None)

        out_gen = self.genout(x)        # output scores from generator      (batsize, outvocsize)
        out_gen = torch.nn.functional.softmax(out_gen, -1)

        # region copying:
        alphas = torch.nn.functional.softmax(scores, -1)
        out_cpy = torch.zeros_like(out_gen)     # (batsize, outvocsize)
        out_cpy.scatter_add_(-1, self._ctx_ids, alphas)
        # endregion

        # mix
        mix = self.gate(x)      # (batsize, 2)
        out = out_gen * mix[0] + out_cpy * mix[1]

        # region automasking
        if self.automasker is not None:
            mask = self.automasker.get_out_mask().to(out.device).float()  # 0/1 mask
            out += torch.log(mask)
        # endregion

        return out


class PointerGeneratorCell(torch.nn.Module):        # from q.rnn.LuongCell
    def __init__(self, emb=None, core=None, att=None, merge:q.rnn.DecCellMerge=q.rnn.ConcatDecCellMerge(),
                 out=None, feed_att=False, return_alphas=False, return_scores=False, return_other=False,
                 dropout=0, **kw):
        """

        :param emb:
        :param core:
        :param att:
        :param merge:
        :param out:         if None, out_vec (after merge) is returned
        :param feed_att:
        :param h_hat_0:
        :param kw:
        """
        super(PointerGeneratorCell, self).__init__(**kw)
        self.emb, self.core, self.att, self.merge, self.out = emb, core, att, merge, out
        self.feed_att = feed_att
        self._outvec_tm1 = None    # previous attention summary
        self.outvec_t0 = None
        self.return_alphas = return_alphas
        self.return_scores = return_scores
        self.return_other = return_other
        self.dropout = torch.nn.Dropout(dropout)

    def batch_reset(self):
        self.outvec_t0 = None
        self._outvec_tm1 = None

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        assert (ctx is not None)

        if isinstance(self.out, q.rnn.AutoMaskedOut):
            self.out.update(x_t)

        embs = self.emb(x_t)        # embed input tokens
        if q.issequence(embs) and len(embs) == 2:   # unpack if necessary
            embs, mask = embs

        if self.feed_att:
            if self._outvec_tm1 is None:
                assert (self.outvec_t0 is not None)   #"h_hat_0 must be set when feed_att=True"
                self._outvec_tm1 = self.outvec_t0
            core_inp = torch.cat([embs, self._outvec_tm1], 1)     # append previous attention summary
        else:
            core_inp = embs

        core_out = self.core(core_inp)  # feed through rnn

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)  # do attention
        out_vec = self.merge(core_out, summaries, core_inp)
        out_vec = self.dropout(out_vec)
        self._outvec_tm1 = out_vec      # store outvec (this is how Luong, 2015 does it)

        ret = tuple()
        if self.out is None:
            ret += (out_vec,)
        else:
            _out_vec = self.out(out_vec, scores=scores)
            ret += (_out_vec,)

        # other returns
        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret