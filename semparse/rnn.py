import torch
import qelos as q
import numpy as np


# region pointer generator
class PointerGeneratorOutGate(torch.nn.Module):
    def __init__(self, inpdim, vecdim, outdim=2, **kw):
        super(PointerGeneratorOutGate, self).__init__(**kw)
        self.inpdim, self.vecdim, self.outdim = inpdim, vecdim, outdim
        if outdim == 0:
            outdim = 1
        self.lin1 = torch.nn.Linear(inpdim, vecdim)
        self.act1 = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(vecdim, outdim)
        if self.outdim == 0:
            self.act2 = torch.nn.Sigmoid()
        else:
            self.act2 = torch.nn.Softmax(-1)

    def forward(self, x, mask=None):
        r = self.act1(self.lin1(x))
        o = self.lin2(r)
        if mask is not None:
            o = o + torch.log(mask.float())
        o = self.act2(o)
        if self.outdim == 0:
            o = torch.cat([o, 1 - o], 1)
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
        self._mix_history = None

    @property
    def ctx_ids(self):
        return self._ctx_ids

    @ctx_ids.setter
    def ctx_ids(self, value):
        self._ctx_ids = self.sourcemap[value]       # already maps to output dict when setting ctx_ids

    def batch_reset(self):
        self._ctx_ids = None
        self._mix_history = None

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
        ctx_ids = self._ctx_ids
        if alphas.size(1) < self._ctx_ids.size(1):
            ctx_ids = ctx_ids[:, :alphas.size(1)]
        out_cpy.scatter_add_(-1, ctx_ids, alphas)
        # endregion

        # mix
        mix = self.gate(x)      # (batsize, 2)
        out =   out_gen * mix[:, 0].unsqueeze(1) \
              + out_cpy * mix[:, 1].unsqueeze(1)

        # TODO: save mix in mix history for supervision later

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
# endregion


# region self-pointing pointer-generator (two pointers, one generator)
class SelfPointerGeneratorOut(torch.nn.Module):     # integrates q.rnn.AutoMaskedOut
    """
    Performs the generation of tokens or copying of input.

    """
    def __init__(self, genout=None, sourcemap=None,
                 gate:PointerGeneratorOutGate=None,
                 automasker:q.rnn.AutoMasker=None, **kw):
        super(SelfPointerGeneratorOut, self).__init__(**kw)
        self.genout = genout
        self.sourcemap = sourcemap      # maps from input ids to output ids (all input ids must be part of output dict). must be 1D long tensor containing output dict ids
        self.automasker = automasker    # automasker for masking out invalid tokens
        self.gate = gate                # module that takes in the vector and outputs scores of how to mix the gen and cpy distributions
        self._ctx_ids = None     # must be set in every batch, before decoding, contains mapped input sequence ids
        self.prev_x_tokens = None

    @property
    def ctx_ids(self):
        return self._ctx_ids

    @ctx_ids.setter
    def ctx_ids(self, value):
        self._ctx_ids = self.sourcemap[value]       # already maps to output dict when setting ctx_ids

    def batch_reset(self):
        self._ctx_ids = None
        self.prev_x_tokens = None

    def update(self, x):        # from automasker
        if self.automasker is not None:
            self.automasker.update(x)
        if self.prev_x_tokens is None:
            self.prev_x_tokens = x.unsqueeze(1)       # introduce sequence dimension
        else:
            self.prev_x_tokens = torch.cat([self.prev_x_tokens, x.unsqueeze(1)], 1)

    def forward(self, x, scores=None, selfscores=None):
        """
        :param x:       vector for generation
        :param scores:  attention scores (unnormalized)     (batsize, seqlen)
        :return:        probabilities over output tokens
        """
        assert(self._ctx_ids is not None)

        out_gen = self.genout(x)        # output scores from generator      (batsize, outvocsize)
        out_gen = torch.nn.functional.softmax(out_gen, -1)

        # region copying from input
        alphas = torch.nn.functional.softmax(scores, -1)
        out_cpy = torch.zeros_like(out_gen)     # (batsize, outvocsize)
        ctx_ids = self._ctx_ids
        if alphas.size(1) < ctx_ids.size(1):
            ctx_ids = ctx_ids[:, :alphas.size(1)]
        out_cpy.scatter_add_(-1, ctx_ids, alphas)
        # endregion

        # region copying from previous output
        if selfscores is not None:
            selfalphas = torch.nn.functional.softmax(selfscores, -1)
            out_slf = torch.zeros_like(out_gen)     # (batsize, outvocsize)
            out_slf.scatter_add_(-1, self.prev_x_tokens[:, :-1], selfalphas)
        else:
            out_slf = torch.zeros_like(out_gen)
        # endregion

        # mix
        mask = None
        if selfscores is None:
            mask = torch.tensor([1, 1, 0]).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        mix = self.gate(x, mask=mask)      # (batsize, 3)
        out =   out_gen * mix[:, 0].unsqueeze(1) \
              + out_cpy * mix[:, 1].unsqueeze(1) \
              + out_slf * mix[:, 2].unsqueeze(1)

        # region automasking
        if self.automasker is not None:
            mask = self.automasker.get_out_mask().to(out.device).float()  # 0/1 mask
            out += torch.log(mask)
        # endregion

        return out


class SelfPointerGeneratorCell(torch.nn.Module):        # from q.rnn.LuongCell
    def __init__(self, emb=None, core=None, att=None, selfatt=None, merge:q.rnn.DecCellMerge=q.rnn.ConcatDecCellMerge(),
                 out:SelfPointerGeneratorOut=None, feed_att=False, return_alphas=False, return_scores=False, return_other=False,
                 dropout=0, **kw):
        """

        :param emb:
        :param core:
        :param att:
        :param selfatt:
        :param merge:
        :param out:         if None, out_vec (after merge) is returned
        :param feed_att:
        :param h_hat_0:
        :param kw:
        """
        super(SelfPointerGeneratorCell, self).__init__(**kw)
        self.emb, self.core, self.att, self.merge, self.out = emb, core, att, merge, out
        self.selfatt = selfatt
        self.feed_att = feed_att
        self._outvec_tm1 = None    # previous attention summary
        self.outvec_t0 = None
        self.return_alphas = return_alphas
        self.return_scores = return_scores
        self.return_other = return_other
        self.dropout = torch.nn.Dropout(dropout)
        self.prev_coreouts = None   # previous states of decoder that have been used as queries

    def batch_reset(self):
        self.outvec_t0 = None
        self._outvec_tm1 = None
        self.prev_coreouts = None   # previous states of decoder that have been used as queries

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        assert (ctx is not None)

        # if isinstance(self.out, q.rnn.AutoMaskedOut):
        #     self.out.update(x_t)
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

        # do normal attention over input
        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)  # do attention

        # do self-attention
        if self.prev_coreouts is not None:
            selfalphas, selfsummaries, selfscores = self.selfatt(core_out, self.prev_coreouts)  # do self-attention
        else:
            selfalphas, selfsummaries, selfscores = None, None, None
        # TODO ??? use self-attention summaries for output generation too?

        out_vec = self.merge(core_out, summaries, core_inp)
        out_vec = self.dropout(out_vec)
        self._outvec_tm1 = out_vec      # store outvec (this is how Luong, 2015 does it)

        # save coreouts
        if self.prev_coreouts is None:
            self.prev_coreouts = core_out.unsqueeze(1)      # introduce a sequence dimension
        else:
            self.prev_coreouts = torch.cat([self.prev_coreouts, core_out.unsqueeze(1)], 1)

        ret = tuple()
        if self.out is None:
            ret += (out_vec,)
        else:
            _out_vec = self.out(out_vec, scores=scores, selfscores=selfscores)
            ret += (_out_vec,)

        # other returns
        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret

# endregion


# region tree decoders
def parent_merge_parent_first(parent_alphas, alphas):
    _z = torch.min(torch.tensor(1.0).to(alphas.device), (1 - parent_alphas) / alphas)
    z = parent_alphas + alphas * _z.detach()
    return z


class SimpleTreeLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, dropout_in=0., dropout_rec=0., **kw):
        super(SimpleTreeLSTMCell, self).__init__(**kw)
        self.input_size, self.hidden_size, self.usebias = input_size, hidden_size, bias
        self.weight_ih = torch.nn.Parameter(torch.randn(hidden_size * 4, input_size + hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(hidden_size * 4, hidden_size))
        self.bias = torch.nn.Parameter(torch.randn(hidden_size * 4)) if bias is True else None

        # dropouts etc
        self.dropout_in, self.dropout_rec, self.dropout_rec_c = None, None, None
        if dropout_in > 0.:
            self.dropout_in = q.RecDropout(p=dropout_in)
        if dropout_rec > 0.:
            self.dropout_rec = q.RecDropout(p=dropout_rec)
            self.dropout_rec_c = q.RecDropout(p=dropout_rec)
        assert(isinstance(self.dropout_in, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec_c, (torch.nn.Dropout, q.RecDropout, type(None))))
        self.c_tm1 = None
        self.y_tm1 = None
        self.register_buffer("c_0", torch.zeros(1, self.hidden_size))
        self.register_buffer("y_0", torch.zeros(1, self.hidden_size))
        self.reset_parameters()

        self.prev_cs = None
        self.prev_ys = None
        self.prev_parent_ptr = None

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.orthogonal_(self.weight_hh)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def batch_reset(self):
        self.c_tm1 = None
        self.y_tm1 = None
        self.prev_cs = None
        self.prev_ys = None
        self.prev_parent_ptr = None

    def forward(self, x_t, prev_pushpop=None):
        """
        :param x_t:       (batsize, input_size)
        :param hx:      tuple (h_0, c_0), each is (batsize, hidden_size)
        :return:        tuple (h, c), each is (batsize, hidden_size)
        """
        batsize = x_t.size(0)
        x_t = self.dropout_in(x_t) if self.dropout_in else x_t

        # previous states
        # zero states
        y_tm1 = self.y_0.expand(batsize, -1) if self.y_tm1 is None else self.y_tm1
        c_tm1 = self.c_0.expand(batsize, -1) if self.c_tm1 is None else self.c_tm1
        # apply dropout
        y_tm1 = self.dropout_rec(y_tm1) if self.dropout_rec is not None else y_tm1
        c_tm1 = self.dropout_rec_c(c_tm1) if self.dropout_rec_c is not None else c_tm1

        # get the right states:     * always need parent's y_tm1 to concat as input
                                #   * c_tm1 and y_tm1 are overridden based on prev_pushpop:
                                #       - if prev was POP: the states of the parent of which the children were just finished is used
                                #       - else: everything stays the same
                                #   * memory is updated based on prev_pushpop:
                                #       - if prev was PUSH: new parent states on stack
                                #       - if prev was POP: remove some parent states from stack
                                #       - else: don't change stack
        # TODO: what about dropouts?
        parent_y_tm1, c_tm1, y_tm1, retain_gate = self.update_parents(c_tm1, y_tm1, prev_pushpop=prev_pushpop)     # return: (batsize, hidden_size), same, (batsize,)
        x_t = torch.cat([x_t, parent_y_tm1], 1)

        # do cell:
        ih_acts = torch.einsum("oi,bi->bo", self.weight_ih, x_t)
        hh_acts = torch.einsum("oi,bi->bo", self.weight_hh, y_tm1)
        h_acts = ih_acts + hh_acts + self.bias

        # compute gates
        input_gate, forget_gate, output_gate, update = torch.chunk(h_acts, 4, 1)
        input_gate, forget_gate, output_gate = \
            [torch.sigmoid(a) for a in [input_gate, forget_gate, output_gate]]
        update = torch.tanh(update)

        # update states
        c_t = (1 - forget_gate) * c_tm1 + input_gate * update
        y_t = output_gate * torch.tanh(c_t)

        # return
        # y_t, c_t = self.apply_mask_t((y_tm1, y_t), (c_tm1, c_t), mask_t=mask_t)
        self.y_tm1, self.c_tm1 = y_t, c_t
        return y_t

    def update_parents(self, c_tm1, y_tm1, prev_pushpop=None):   # (batsize, hidden_size)        TODO
        """
        Given currently computed retain_gate and prev_pushpop, updates the retain structures, returns actual retain gate and mask.
        """
        # get parent retain gate
        if self.prev_cs is None:       # no history (means first time step (or something wrong))
            # initialize prev_retains
            self.prev_cs = torch.zeros_like(c_tm1).unsqueeze(1)
            self.prev_ys = torch.zeros_like(y_tm1).unsqueeze(1)
            self.prev_parent_ptr = [[0] for _ in range(c_tm1.size(0))]     # the all-zero first element of prev_retains in parent to all

        # get parent retain gate for computing child retain
        parent_ptr = torch.tensor(
            [prev_parent_ptr_i[-1] for prev_parent_ptr_i in self.prev_parent_ptr]).long().to(c_tm1.device)
        parent_c_tm1 = self.prev_cs.gather(1, parent_ptr.unsqueeze(-1).unsqueeze(-1)
                                                 .repeat(1, 1, self.prev_cs.size(2))).squeeze(1)
        parent_y_tm1 = self.prev_ys.gather(1, parent_ptr.unsqueeze(-1).unsqueeze(-1)
                                                 .repeat(1, 1, self.prev_cs.size(2))).squeeze(1)

        self.prev_cs = torch.cat([self.prev_cs, c_tm1.unsqueeze(1)], 1)
        self.prev_ys = torch.cat([self.prev_ys, y_tm1.unsqueeze(1)], 1)

        # update pointers
        k = self.prev_parents.size(1) - 1
        for i in range(len(prev_pushpop)):
            pp = prev_pushpop[i].cpu().item()
            if pp > 0:        # PUSH: push on stack
                self.prev_parent_ptr[i].append(k)
            elif pp < 0:      # POP: pop stack
                while pp < 0 and len(self.prev_parent_ptr[i]) > 1:
                    self.prev_parent_ptr[i].pop(-1)
                    pp += 1
            else:             # neither --> sibling
                pass

        # get parent retain gate to apply to the current timestep (produces next token, so if was PUSH --> uses new one, ...)
        parent_retain_ptr = torch.tensor(
            [prev_retains_ptr_i[-1] for prev_retains_ptr_i in self.prev_parent_ptr]).long().to(retain_gate.device)
        parent_retain = self.prev_parents.gather(1, parent_retain_ptr.unsqueeze(-1).unsqueeze(-1)
                                                 .repeat(1, 1, self.prev_parents.size(2))).squeeze(1)
        # print(parent_retain_ptr)
        # print(parent_retain)
        return parent_y_tm1, c_tm1, y_tm1, retain_gate


class GatedTreeLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, dropout_in=0., dropout_rec=0., init_no_retain=True, **kw):
        super(GatedTreeLSTMCell, self).__init__(**kw)
        self.input_size, self.hidden_size, self.usebias = input_size, hidden_size, bias
        self.weight_ih = torch.nn.Parameter(torch.randn(hidden_size * 5, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(hidden_size * 5, hidden_size))
        self.bias = torch.nn.Parameter(torch.randn(hidden_size * 5)) if bias is True else None

        # dropouts etc
        self.dropout_in, self.dropout_rec, self.dropout_rec_c = None, None, None
        if dropout_in > 0.:
            self.dropout_in = q.RecDropout(p=dropout_in)
        if dropout_rec > 0.:
            self.dropout_rec = q.RecDropout(p=dropout_rec)
            self.dropout_rec_c = q.RecDropout(p=dropout_rec)
        assert(isinstance(self.dropout_in, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec_c, (torch.nn.Dropout, q.RecDropout, type(None))))
        self.c_tm1 = None
        self.y_tm1 = None
        self.register_buffer("c_0", torch.zeros(1, self.hidden_size))
        self.register_buffer("y_0", torch.zeros(1, self.hidden_size))

        self.prev_retains = None
        self.prev_retains_ptr = None

        self._init_no_retain = init_no_retain

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.orthogonal_(self.weight_hh)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)
            if self._init_no_retain:
                self.bias.data[:self.hidden_size] = -5       # bias the retain gate to initially retain nothing
                                                    # otherwise, retain might get fully saturated and not update at all for deeper levels

    def batch_reset(self):
        self.c_tm1 = None
        self.y_tm1 = None
        self.prev_retains = None
        self.prev_retains_ptr = None

    def forward(self, x_t, prev_pushpop=None):
        """
        :param x_t:       (batsize, input_size)
        :param hx:      tuple (h_0, c_0), each is (batsize, hidden_size)
        :return:        tuple (h, c), each is (batsize, hidden_size)
        """
        batsize = x_t.size(0)
        x_t = self.dropout_in(x_t) if self.dropout_in else x_t

        # previous states
        # zero states
        y_tm1 = self.y_0.expand(batsize, -1) if self.y_tm1 is None else self.y_tm1
        c_tm1 = self.c_0.expand(batsize, -1) if self.c_tm1 is None else self.c_tm1
        # apply dropout
        y_tm1 = self.dropout_rec(y_tm1) if self.dropout_rec is not None else y_tm1
        c_tm1 = self.dropout_rec_c(c_tm1) if self.dropout_rec_c is not None else c_tm1

        # do cell:
        ih_acts = torch.einsum("oi,bi->bo", self.weight_ih, x_t)
        hh_acts = torch.einsum("oi,bi->bo", self.weight_hh, y_tm1)
        h_acts = ih_acts + hh_acts + self.bias

        # compute gates
        retain_gate, input_gate, forget_gate, output_gate, update = torch.chunk(h_acts, 5, 1)
        retain_gate, input_gate, forget_gate, output_gate = \
            [torch.sigmoid(a) for a in [retain_gate, input_gate, forget_gate, output_gate]]
        update = torch.tanh(update)

        # decide retain gate
        retain_gate = self.update_retains(retain_gate, prev_pushpop=prev_pushpop)

        # update states
        c_t = (1 - forget_gate) * c_tm1 + input_gate * update
        c_t = retain_gate * c_tm1 + (1 - retain_gate) * c_t
        y_t = output_gate * torch.tanh(c_t)

        # return
        # y_t, c_t = self.apply_mask_t((y_tm1, y_t), (c_tm1, c_t), mask_t=mask_t)
        self.y_tm1, self.c_tm1 = y_t, c_t
        return y_t

    def update_retains(self, retain_gate, prev_pushpop=None):   # (batsize, hidden_size)
        """
        Given currently computed retain_gate and prev_pushpop, updates the retain structures, returns actual retain gate and mask.
        """
        # get parent retain gate
        if self.prev_retains is None:       # no history (means first time step (or something wrong))
            # initialize prev_retains
            self.prev_retains = torch.zeros_like(retain_gate).unsqueeze(1)
            self.prev_retains_ptr = [[0] for _ in range(retain_gate.size(0))]     # the all-zero first element of prev_retains in parent to all

        # get parent retain gate for computing child retain
        parent_retain_ptr = torch.tensor(
            [prev_retains_ptr_i[-1] for prev_retains_ptr_i in self.prev_retains_ptr]).long().to(retain_gate.device)
        parent_retain = self.prev_retains.gather(1, parent_retain_ptr.unsqueeze(-1).unsqueeze(-1)
                                                 .repeat(1, 1, self.prev_retains.size(2))).squeeze(1)

        # compute child retain gates for every example
        child_retain = parent_merge_parent_first(parent_retain, retain_gate)
        self.prev_retains = torch.cat([self.prev_retains, child_retain.unsqueeze(1)], 1)

        # update pointers
        k = self.prev_retains.size(1) - 1
        for i in range(len(prev_pushpop)):
            pp = prev_pushpop[i].cpu().item()
            if pp > 0:        # PUSH: push on stack
                self.prev_retains_ptr[i].append(k)
            elif pp < 0:      # POP: pop stack
                while pp < 0 and len(self.prev_retains_ptr[i]) > 1:
                    self.prev_retains_ptr[i].pop(-1)
                    pp += 1
            else:             # neither --> sibling
                pass

        # get parent retain gate to apply to the current timestep (produces next token, so if was PUSH --> uses new one, ...)
        parent_retain_ptr = torch.tensor(
            [prev_retains_ptr_i[-1] for prev_retains_ptr_i in self.prev_retains_ptr]).long().to(retain_gate.device)
        parent_retain = self.prev_retains.gather(1, parent_retain_ptr.unsqueeze(-1).unsqueeze(-1)
                                                 .repeat(1, 1, self.prev_retains.size(2))).squeeze(1)
        # print(parent_retain_ptr)
        # print(parent_retain)
        return parent_retain


class TreeRNNDecoderCellCore(torch.nn.Module):
    """ contains multiple TreeRNNCells"""
    def __init__(self, *cells, **kw):
        super(TreeRNNDecoderCellCore, self).__init__(**kw)
        self.cells = torch.nn.ModuleList(cells)

    def __getitem__(self, item):
        return self.cells[item]

    def forward(self, x_t, prev_pushpop=None):
        assert(prev_pushpop is not None)
        y_t = x_t
        for cell in self.cells:
            y_t = cell(y_t, prev_pushpop=prev_pushpop)
        return y_t


class TreeRNNDecoderCell(torch.nn.Module):
    def __init__(self, emb=None, core:TreeRNNDecoderCellCore=None, att=None, merge:q.rnn.DecCellMerge=q.rnn.ConcatDecCellMerge(),
                 out=None, feed_att=False, return_alphas=False, return_scores=False, return_other=False,
                 dropout=0, id2pushpop=None, **kw):
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
        super(TreeRNNDecoderCell, self).__init__(**kw)
        self.emb, self.core, self.att, self.merge, self.out = emb, core, att, merge, out
        self.feed_att = feed_att
        self._outvec_tm1 = None    # previous attention summary
        self.outvec_t0 = None
        self.return_alphas = return_alphas
        self.return_scores = return_scores
        self.return_other = return_other
        self.dropout = torch.nn.Dropout(dropout)
        self._id2pushpop = id2pushpop

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

        prev_pushpop = self.get_pushpop_from(x_t)           # THIS LINE IS ADDED

        core_out = self.core(core_inp, prev_pushpop=prev_pushpop)  # feed through rnn   # THIS LINE IS CHANGED

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)  # do attention
        out_vec = self.merge(core_out, summaries, core_inp)
        out_vec = self.dropout(out_vec)
        self._outvec_tm1 = out_vec      # store outvec (this is how Luong, 2015 does it)

        ret = tuple()
        if self.out is None:
            ret += (out_vec,)
        else:
            _out_vec = self.out(out_vec)
            ret += (_out_vec,)

        # other returns
        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret

    def get_pushpop_from(self, x_t):    # (batsize,) ids        # THIS METHOD IS ADDED
        """ Get pushpop from x_t: based on x_t, decides whether to push (>0), do nothing (0) or pop (<0) previous attentions """
        if self._id2pushpop is not None:
            return self._id2pushpop[x_t]
        else:
            raise NotImplemented()


class NumChildPushPopper(object):
    def __init__(self, tok_desc=None, **kw):
        super(NumChildPushPopper, self).__init__(**kw)
        self.pushpop_stats = None    # stack for pushpop, keeps track how many children left to decode
        self.tok_desc = tok_desc # must be set !!!
        # token descriptions: dict from token id to how many children expected (if parent then >0, if leaf then 0, if terminator then -1)

    def batch_reset(self):
        super(NumChildPushPopper, self).batch_reset()
        self.pushpop_stats = None

    def get_pushpop_from(self, x_t):    # (batsize,) ids
        assert(self.tok_desc is not None)
        # initialize pushpop stats for this batch
        if self.pushpop_stats is None:
            self.pushpop_stats = [[np.infty] for i in range(len(x_t))]
        outpushpop = []
        for i in range(len(x_t)):
            x_t_i = x_t[i].cpu().item()
            exp_num_children = self.tok_desc[x_t_i]
            dopop = False
            # if len(self.pushpop_stats[i]) == 0: # should only be the case when complete, all that can follow are PADs
            #     outpushpop.append(0)
            #     continue
            self.pushpop_stats[i][-1] -= 1  # one less child to expect
            if exp_num_children > 0:    # x_t_i is parent
                self.pushpop_stats[i].append(exp_num_children)
                outpushpop.append(1)    # push in any case
            elif exp_num_children == 0: # x_t_i is leaf
                # check if sufficient children
                if self.pushpop_stats[i][-1] == 0:  # all children satisfied --> pop
                    dopop = True
                else:
                    outpushpop.append(0)
            else:
                dopop = True
            if dopop is True:   # if enough children or explicit terminator
                # pop out of current level and keep going up until found where more children are needed
                numpops = 1
                self.pushpop_stats[i].pop()
                while self.pushpop_stats[i][-1] <= 0:
                    self.pushpop_stats[i].pop()
                    numpops += 1
                outpushpop.append(-numpops)
        ret = torch.tensor(outpushpop)
        return ret


class TreeRNNDecoderCellNumChild(NumChildPushPopper, TreeRNNDecoderCell):
    pass



def test_gated_tree_lstm_cell(lr=0.):
    m = GatedTreeLSTMCell(4, 3)

    x = torch.randn(2, 6, 4)
    prev_pushpop = torch.tensor(
        [
            [1, 0, 1, 0, -1, -1],
            [1, 1, 1, 0, -3, 0]
        ]
    )

    for t in range(6):
        y = m(x[:, t], prev_pushpop[:, t])

    print("done")
# endregion

if __name__ == '__main__':
    q.argprun(test_gated_tree_lstm_cell)

