import torch
import qelos as q
import numpy as np
import math


# region normal attention
class AttComp(torch.nn.Module):
    """ computes attention scores """
    def forward(self, qry, ctx, ctx_mask=None):
        raise NotImplemented()


class SummComp(torch.nn.Module):
    def forward(self, values, alphas):
        raise NotImplemented()


class Attention(torch.nn.Module):
    """ Computes phrase attention. For use with encoders and decoders from rnn.py """
    def __init__(self, attcomp:AttComp=None, summcomp:SummComp=None, score_norm=torch.nn.Softmax(-1)):
        """
        :param attcomp:     used to compute attention scores
        :param summcomp:    used to compute summary
        """
        super(Attention, self).__init__()
        # self.prevatts = None    # holds previous attention vectors
        # self.prevatt_ptr = None     # for every example, contains a list with pointers to indexes of prevatts
        self.attcomp = attcomp if attcomp is not None else DotAttComp()
        self.summcomp = summcomp if summcomp is not None else SumSummComp()
        self.score_norm = score_norm

    def forward(self, qry, ctx, ctx_mask=None, values=None):
        """
        :param qry:     (batsize, dim)
        :param ctx:     (batsize, seqlen, dim)
        :param ctx_mask: (batsize, seqlen)
        :param values:  (batsize, seqlen, dim)
        :return:
        """
        scores = self.attcomp(qry, ctx, ctx_mask=ctx_mask)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        alphas = self.score_norm(scores)
        values = ctx if values is None else values
        summary = self.summcomp(values, alphas)
        return alphas, summary, scores


class DotAttComp(AttComp):
    def forward(self, qry, ctx, ctx_mask=None):
        """
        :param qry:         (batsize, dim) or (batsize, zeqlen, dim)
        :param ctx:         (batsize, seqlen, dim)
        :param ctx_mask:
        :return:
        """
        if qry.dim() == 2:
            ret = torch.einsum("bd,bsd->bs", [qry, ctx])
        elif qry.dim() == 3:
            ret = torch.einsum("bzd,bsd->bzs", [qry, ctx])
        else:
            raise q.SumTingWongException("qry has unsupported dimension: {}".format(qry.dim()))
        return ret


class FwdAttComp(AttComp):
    def __init__(self, qrydim=None, ctxdim=None, encdim=None, numlayers=1, dropout=0, **kw):
        super(FwdAttComp, self).__init__(**kw)
        layers = [torch.nn.Linear(qrydim + ctxdim, encdim)] \
                 + [torch.nn.Linear(encdim, encdim) for _ in range(numlayers - 1)]
        acts = [torch.nn.Tanh() for _ in range(len(layers))]
        layers = [a for b in zip(layers, acts) for a in b]
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(encdim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, qry, ctx, ctx_mask=None):
        """
        :param qry:     (batsize, qrydim)
        :param ctx:     (batsize, seqlen, ctxdim)
        :param ctx_mask:    (batsize, seqlen)
        :return:
        """
        inp = torch.cat([ctx, qry.unsqueeze(1).repeat(1, ctx.size(1), 1)], 2)
        out = self.mlp(inp)
        ret = out.squeeze(-1)
        return ret


class SumSummComp(SummComp):
    def forward(self, values, alphas):
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return summary


class BasicAttention(Attention):
    def __init__(self, **kw):
        attcomp = DotAttComp()
        summcomp = SumSummComp()
        super(BasicAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, **kw)


class FwdAttention(Attention):
    def __init__(self, qrydim, ctxdim, encdim, dropout=0., **kw):
        attcomp = FwdAttComp(qrydim=qrydim, ctxdim=ctxdim, encdim=encdim, dropout=dropout)
        summcomp = SumSummComp()
        super(FwdAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, **kw)


# endregion


# region Relative Attention
class VecComp(torch.nn.Module):
    """ maps ctx~(batsize, seqlen, ctxdim) to relvecs~(batsize, seqlen, seqlen, vecdim) ~~ self-attention """
    def __init__(self, ctxdim, vecdim, **kw):
        super(VecComp, self).__init__(**kw)
        self.ctxdim, self.vecdim = ctxdim, vecdim

    def forward(self, ctx):
        """
        :param ctx:     (batsize, seqlen, ctxdim)
        :return:        (batsize, seqlen, seqlen, vecdim)
        """
        raise NotImplemented()


class FwdVecComp(VecComp):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        super(FwdVecComp, self).__init__(ctxdim, vecdim, **kw)
        self.lin1 = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin2 = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.nonlin = torch.nn.Tanh()

    def forward(self, ctx):
        out1 = self.lin1(ctx)
        out2 = self.lin2(ctx)       # (batsize, seqlen, vecdim)
        out = self.nonlin(out1.unsqueeze(1) + out2.unsqueeze(2))
        return out


class ComboFwdVecComp(VecComp):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        super(ComboFwdVecComp, self).__init__(ctxdim, vecdim, **kw)
        self.lin1 = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin2 = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin_mul = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin_diff = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.nonlin = torch.nn.Tanh()

    def forward(self, ctx):
        out1 = self.lin1(ctx)
        out2 = self.lin2(ctx)       # (batsize, seqlen, vecdim)
        mul = ctx.unsqueeze(1) * ctx.unsqueeze(2)       # (batsize, seqlen, seqlen, vecdim)
        diff = ctx.unsqueeze(1) - ctx.unsqueeze(2)
        outmul = self.lin_mul(mul)
        outdiff = self.lin_diff(diff)
        out = self.nonlin(out1.unsqueeze(1) + out2.unsqueeze(2) + outmul + outdiff)
        return out


class BilinVecComp(VecComp):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        super(BilinVecComp, self).__init__(ctxdim, vecdim, **kw)
        self.W = torch.nn.Parameter(torch.Tensor(ctxdim, ctxdim, vecdim))
        self.bias = torch.nn.Parameter(torch.Tensor(vecdim)) if bias else None
        self.nonlin = torch.nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        torch.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.init.uniform_(self.bias, -bound, bound)

    def forward(self, ctx):
        out = torch.einsum("bsi,bzj,ijk->bszk", ctx, ctx, self.W)
        if self.bias is not None:
            out = out + self.bias
        out = self.nonlin(out)
        return out


class BilinAndFwdComboVecComp(VecComp):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        super(BilinAndFwdComboVecComp, self).__init__(ctxdim, vecdim, **kw)
        self.W = torch.nn.Parameter(torch.Tensor(ctxdim, ctxdim, vecdim))
        self.bias = torch.nn.Parameter(torch.Tensor(vecdim)) if bias else None

        self.lin1 = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin2 = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin_mul = torch.nn.Linear(ctxdim, vecdim, bias=bias)
        self.lin_diff = torch.nn.Linear(ctxdim, vecdim, bias=bias)

        self.nonlin = torch.nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        torch.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.init.uniform_(self.bias, -bound, bound)

    def forward(self, ctx):
        out = torch.einsum("bsi,bzj,ijk->bszk", ctx, ctx, self.W)
        if self.bias is not None:
            out = out + self.bias

        out1 = self.lin1(ctx)
        out2 = self.lin2(ctx)  # (batsize, seqlen, vecdim)
        mul = ctx.unsqueeze(1) * ctx.unsqueeze(2)  # (batsize, seqlen, seqlen, vecdim)
        diff = ctx.unsqueeze(1) - ctx.unsqueeze(2)
        outmul = self.lin_mul(mul)
        outdiff = self.lin_diff(diff)

        out = self.nonlin(out + out1.unsqueeze(1) + out2.unsqueeze(2) + outmul + outdiff)
        return out


class RelAttention(torch.nn.Module):
    def __init__(self, veccomp:VecComp=None, attcomp:AttComp=DotAttComp(), summcomp:SummComp=SumSummComp(), temperature=1., threshold=1e-6, **kw):
        super(RelAttention, self).__init__(**kw)
        self.threshold, self.temperature = threshold, temperature
        self.veccomp = veccomp      # maps ctx~(batsize, seqlen, ctxdim) to relvecs~(batsize, seqlen, seqlen, vecdim) ~~ self-attention
        self.attcomp, self.summcomp = attcomp, summcomp
        self.scorenorm = torch.nn.Softmax(-1)
        self.prevatts = None            # (batsize, seqlen)
        self.relvecs = None     # (batsize, seqlen, seqlen, vecdim)
        self.prevatts_history = []      # will become decseqlen-sized list of prevatts (batsize, seqlen)
        self.feed_prevatts_acc = None     # decseqlen-sized list of prevatts (batsize, seqlen)
        self.t = 0
        self._bad_prevatts = False

    def batch_reset(self):
        self.prevatts = None
        self.relvecs = None
        self.prevatts_history = []
        self.feed_prevatts_acc = None
        self.t = 0

    def forward(self, qry, ctx, ctx_mask=None, values=None):
        """
        :param qry:         (batsize, qdim)
        :param ctx:         (batsize, seqlen, ctxdim)
        :param ctx_mask:    (batsize, seqlen)
        :param values:      (batsize, seqlen)
        :return:
        """
        # initialize prevatts if None: init assigns all prob to first element --> first element of ctx must be a start token
        if self.prevatts is None:
            self.prevatts = torch.zeros_like(ctx[:, :, 0])
            self.prevatts[:, 0] = 1.
            # create and store relation vectors
            self.relvecs = self.veccomp(ctx)

        # get non-negligible part of relvecs        # TODO: do sparse for more efficiency
        # relvecs_idxs = torch.nonzero(self.prevatts > self.threshold)

        # do attcomp with qry over relvecs
        flatrelvecs = self.relvecs.view(self.relvecs.size(0), self.relvecs.size(1) * self.relvecs.size(2), self.relvecs.size(3))    # (batsize, seqlen*seqlen, vecdim)
        flatrelatt_scores = self.attcomp(qry, flatrelvecs)      # (batsize, seqlen * seqlen)
        relatt_scores = flatrelatt_scores.view(self.relvecs.size(0), self.relvecs.size(1), self.relvecs.size(2))    # (batsize, seqlen, seqlen)

        # apply ctx_mask before summary
        relatt_scores = relatt_scores + (torch.log(ctx_mask.float().unsqueeze(1)) if ctx_mask is not None else 0)
        relatt_alphas = self.scorenorm(relatt_scores)       # (batsize, seqlen, seqlen)
        alphas = torch.einsum("bsz,bs->bz", relatt_alphas, self.prevatts)

        self.prevatts = alphas
        if self._bad_prevatts is True:
            self.prevatts = torch.zeros_like(ctx[:, :, 0])
            self.prevatts[:, 2] = 1.

        # saving history and using feed:
        self.prevatts_history.append(self.prevatts)
        if self.feed_prevatts_acc is not None:
            self.prevatts = self.feed_prevatts_acc[self.t]

        values = ctx if values is None else values
        summary = self.summcomp(values, alphas)

        self.t += 1
        return alphas, summary, relatt_scores


class BasicRelAttention(RelAttention):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        veccomp = FwdVecComp(ctxdim, vecdim, bias=bias)
        super(BasicRelAttention, self).__init__(veccomp, **kw)


class ComboRelAttention(RelAttention):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        veccomp = ComboFwdVecComp(ctxdim, vecdim, bias=bias)
        super(ComboRelAttention, self).__init__(veccomp, **kw)


class AbsRelAttention(torch.nn.Module):
    """
    Attention mechanism consisting of first absolute attention, followed by a relative attention step
    --> doesn't need prevatts
    """
    def __init__(self, prevattcomp:AttComp=DotAttComp(), veccomp:VecComp=None, attcomp:AttComp=DotAttComp(), summcomp:SummComp=SumSummComp(), temperature=1., threshold=1e-6, **kw):
        super(AbsRelAttention, self).__init__(**kw)
        self.threshold, self.temperature = threshold, temperature
        self.prevattcomp = prevattcomp
        self.veccomp = veccomp      # maps ctx~(batsize, seqlen, ctxdim) to relvecs~(batsize, seqlen, seqlen, vecdim) ~~ self-attention
        self.attcomp, self.summcomp = attcomp, summcomp
        self.scorenorm = torch.nn.Softmax(-1)
        self.relvecs = None     # (batsize, seqlen, seqlen, vecdim)
        self.t = 0

    def batch_reset(self):
        self.relvecs = None
        self.t = 0

    def forward(self, qry, ctx, ctx_mask=None, values=None):
        """
        :param qry:         (batsize, qdim)
        :param ctx:         (batsize, seqlen, ctxdim)
        :param ctx_mask:    (batsize, seqlen)
        :param values:      (batsize, seqlen)
        :return:
        """
        if self.relvecs is None:
            # create and store relation vectors
            self.relvecs = self.veccomp(ctx)

        prevatts = self.prevattcomp(qry, ctx, ctx_mask=ctx_mask)
        prevatts += (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        prevatts = self.scorenorm(prevatts)

        # get non-negligible part of relvecs        # TODO: do sparse for more efficiency
        # relvecs_idxs = torch.nonzero(self.prevatts > self.threshold)

        # do attcomp with qry over relvecs
        flatrelvecs = self.relvecs.view(self.relvecs.size(0), self.relvecs.size(1) * self.relvecs.size(2), self.relvecs.size(3))    # (batsize, seqlen*seqlen, vecdim)
        flatrelatt_scores = self.attcomp(qry, flatrelvecs)      # (batsize, seqlen * seqlen)
        relatt_scores = flatrelatt_scores.view(self.relvecs.size(0), self.relvecs.size(1), self.relvecs.size(2))    # (batsize, seqlen, seqlen)

        # apply ctx_mask before summary
        relatt_scores = relatt_scores + (torch.log(ctx_mask.float().unsqueeze(1)) if ctx_mask is not None else 0)
        relatt_alphas = self.scorenorm(relatt_scores)       # (batsize, seqlen, seqlen)
        alphas = torch.einsum("bsz,bs->bz", relatt_alphas, prevatts)

        values = ctx if values is None else values
        summary = self.summcomp(values, alphas)

        self.t += 1
        return alphas, summary, relatt_scores


class BasicAbsRelAttention(AbsRelAttention):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        veccomp = FwdVecComp(ctxdim, vecdim, bias=bias)
        super(BasicAbsRelAttention, self).__init__(veccomp=veccomp, **kw)


class ComboAbsRelAttention(AbsRelAttention):
    def __init__(self, ctxdim, vecdim, bias=True, **kw):
        veccomp = ComboFwdVecComp(ctxdim, vecdim, bias=bias)
        super(ComboAbsRelAttention, self).__init__(veccomp=veccomp, **kw)


def test_rel_attention(lr=0.):
    qry = torch.randn(2, 5)
    ctx = torch.randn(2, 3, 6)
    ctx_mask = torch.tensor([
        [1,1,0],
        [1,1,1]
    ])

    m = ComboRelAttention(6, 5)

    y = m(qry, ctx, ctx_mask=ctx_mask)
    print(y)






# endregion


# region Phrase Attention

# function with a custom backward for getting gradients to both parent and children in PhraseAttention
# forward is an elementwise min
# backward:     - alphas always gets whole gradient
#               - parent_alphas is increased when gradient > 0 else nothing
#                   (so parent's attention can only be increased here if the child needs to attend more to certain places)
#                   (if we equally tried to decrease parent's attentions here too, then would have conflicting signals from its children's attentions, which may not overlap)
#                   (decrease comes from overlap penalty and gradient on parent attention itself)
class ParentOverlapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parent_alphas, alphas):
        ctx.save_for_backward(parent_alphas, alphas)
        ret = torch.min(parent_alphas, alphas)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        gradzeros = torch.zeros_like(grad_output)
        parent_grads = torch.max(gradzeros, grad_output)
        return parent_grads, grad_output


def parent_overlap_f_parent_first(parent_alphas, alphas):
    alphas = 1 - alphas
    _z = torch.min(torch.tensor(1.0), parent_alphas / alphas)
    z = parent_alphas - alphas * _z.detach()
    return z


parent_overlap_f = ParentOverlapFunction.apply
# parent_overlap_f = parent_overlap_f_parent_first


def test_custom_f(lr=0):
    x = torch.rand(5)
    x.requires_grad = True
    y = torch.rand(5)
    y.requires_grad = True
    z = parent_overlap_f(x, y)
    l = z #z.sum()
    l.backward(gradient=torch.tensor([-1,1,-1,1,1]).float())
    print(x)
    print(y)
    print(z)
    print(x.grad)
    print(y.grad)


class PhraseAttention(Attention):       # for depth-first decoding
    """ Assumes masking by termination of tree structure assuming single root (which is also start token) """
    def __init__(self, attcomp:AttComp=None, summcomp:SummComp=None, hard=False, **kw):
        score_norm = torch.nn.Sigmoid()
        super(PhraseAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, score_norm=score_norm)
        self.hard = hard
        self.prevatts_probs = None            # (batsize, declen_so_far, enclen)
        if self.hard is True:
            self.prevatts_samples = None
            self.prevatts_mask = None
        self.prevatt_ptr = None         # for every example, keeps a list of pointers to positions in prevatts
        # structure: batsize x stackdepth x numsiblings
        # For every example, the stack contains groups of siblings.
        # Could have had just batsize x stackdepth, but need to remember siblings for sibling overlap penalty (see prevatt_siblings)
        self.prevatt_siblings = None    # for every example, keeps a list of sets of pointers to groups of siblings
        # structure: batsize x num_sibling_groups x num_siblings_in_group
        # Mainly populated during forward (with a finalization step for top-level siblings in get_sibling_overlap).
        # Consumed in get_sibling_overlap.

    def batch_reset(self):
        self.prevatts_probs, self.prevatt_ptr = None, None
        self.prevatt_siblings = None
        if self.hard is True:
            self.prevatts_samples = None
            self.prevatts_mask = None

    def get_sibling_overlap(self):  # called after all forwards are done
        """
        Gets overlap in siblings based on current state of prevatts and prevatt_ptr.
        Must be called after a batch and before batch reset.
        """
        # finalize prevattr_ptr
        for i, prevattr_ptr_e in enumerate(self.prevatt_ptr):
            if len(prevattr_ptr_e) != 2:    # must contain only the zero-group and top-level group
                pass
                # raise q.SumTingWongException()
            while len(prevattr_ptr_e) > 0:
                ptr_group = prevattr_ptr_e.pop()
                if len(ptr_group) > 1:
                    pass
                    # self.prevatt_siblings[i].append(ptr_group)    # don't add overlap of top-level siblings (we assume single top child, everything else is mask)

        # generate ids by which to gather from prevatts
        ids = torch.zeros(self.prevatts_probs.size(0), self.prevatts_probs.size(1), self.prevatts_probs.size(1),
                          dtype=torch.long, device=self.prevatts_probs.device)
        maxnumsiblingses, maxnumsiblings = 0, 0
        for eid, siblingses in enumerate(self.prevatt_siblings):    # list of lists of ids in prevatts
            maxnumsiblingses = max(maxnumsiblingses, len(siblingses))
            for sgid, siblings in enumerate(siblingses):             # list of ids in prevatts
                maxnumsiblings = max(maxnumsiblings, len(siblings))
                for sid, sibling in enumerate(siblings):
                    ids[eid, sgid, sid] = sibling
        ids = ids[:, :maxnumsiblingses, :maxnumsiblings]

        prevatts = self.prevatts_probs

        idsmask= ((ids != 0).sum(2, keepdim=True) > 1).float()

        # gather from prevatts
        _ids = ids.contiguous().view(ids.size(0), -1).unsqueeze(-1).repeat(1, 1, prevatts.size(2))
        prevatts_gathered = torch.gather(prevatts, 1, _ids)
        prevatts_gathered = prevatts_gathered.view(prevatts.size(0), ids.size(1), ids.size(2), prevatts.size(2))

        # compute overlaps
        overlaps = prevatts_gathered.prod(2)
        overlaps = overlaps * idsmask
        overlaps = overlaps.sum(2).sum(1)
        # overlaps = overlaps.mean(0)
        return overlaps

    def get_logprob_of_sampled_alphas(self):
        if self.hard is False:
            raise q.SumTingWongException("Use this only for RL on hard attention (must be in hard mode).")
        probs = self.prevatts_probs * self.prevatts_samples + (1 - self.prevatts_probs) * (1 - self.prevatts_samples)
        logprobs = torch.log(probs)
        logprobs = logprobs * self.prevatts_mask        # mask the logprobs
        average_within_timestep = True
        if average_within_timestep:
            totals = self.prevatts_mask.sum(2) + 1e-6
            logprobs = logprobs.sum(2) / totals
        else:
            logprobs = logprobs.mean(2)
        return logprobs[:, 2:]     # (batsize, seqlen)  -- decoder mask should be applied on this later

    def get_entropies_of_alpha_dists(self):
        probs = self.prevatts_probs
        dists = torch.distributions.Bernoulli(probs=probs)
        entropies = dists.entropy()
        entropies = entropies * self.prevatts_mask
        average_within_timestep = True
        if average_within_timestep:
            totals = self.prevatts_mask.sum(2) + 1e-12
            entropies = entropies.sum(2) / totals
        else:
            entropies = entropies.mean(2)
        return entropies[:, 2:]     # (batsize, seqlen)  -- decoder mask should be applied on this later

    def forward(self, qry, ctx, ctx_mask=None, values=None, prev_pushpop=None):
        """
        :param qry:     (batsize, dim)
        :param ctx:     (batsize, seqlen, dim)
        :param ctx_mask: (batsize, seqlen)
        :param values:  (batsize, seqlen, dim)
        :param prev_pushpop: (batsize,) - whether PREVIOUS token was a push or pop (or do-nothing) token:
                                -N ==> pop (N=how many to pop),
                                0 ==> nothing,
                                +N ==> push (N doesn't matter, always pushes one)
                ! push/pop happens AFTER the element
        :return:
        """
        # compute attention for token that we will produce next
        # compute attention scores
        scores = self.attcomp(qry, ctx, ctx_mask=ctx_mask)
        # apply ctx mask to attention scores
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        # normalize attention scores
        alphas_probs = self.score_norm(scores)      # sigmoid probs
        if self.hard:
            alphas_dist = torch.distributions.Bernoulli(probs=alphas_probs)
            alphas_samples = alphas_dist.sample()

        # constrain alphas to parent's alphas:
        if self.prevatts_probs is None:   # there is no history
            # initialize prevatts (history)
            self.prevatts_probs = torch.ones_like(alphas_probs).unsqueeze(1).repeat(1, 2, 1)        # means everything is attended to
            if ctx_mask is not None:
                self.prevatts_probs = self.prevatts_probs * ctx_mask.float().unsqueeze(1)
            if self.hard is True:
                self.prevatts_samples = self.prevatts_probs.clone().detach()    #torch.ones_like(alphas_probs).unsqueeze(1).repeat(1, 2, 1)
                self.prevatts_mask = torch.zeros_like(alphas_probs).unsqueeze(1).repeat(1, 2, 1)
            # --> we assume the previous (first ever) attention, used to compute initial (current input) token attended over whole sequence
            # initialize prevatt_ptr
            self.prevatt_ptr = [[[0], []] for _ in range(len(prev_pushpop))]
            # initialize prevatt_siblings
            self.prevatt_siblings = [[] for _ in range(len(prev_pushpop))]


        # update pointers to prevatt
        k = self.prevatts_probs.size(1) - 1       # index of the last produced attention alphas (that were used for prev token)
        for i in range(len(prev_pushpop)):
            self.prevatt_ptr[i][-1].append(k)   # make last token a sibling of the children of the same parent before it (if any)
            if prev_pushpop[i].cpu().item() > 0:  # PUSH: previous token requires children --> make stack deeper by one level
                self.prevatt_ptr[i].append([])
            elif prev_pushpop[i].cpu().item() < 0:    # POP: previous token was last in its row of siblings (and possibly terminates upwards levels too)
                pp = prev_pushpop[i].cpu().item()
                while pp < 0 and len(self.prevatt_ptr[i]) > 2:
                    siblings = self.prevatt_ptr[i].pop(-1)      # pop the list from stack
                    if len(siblings) > 1:       # if longer than 1, add to the list of siblings
                        self.prevatt_siblings[i].append(siblings)
                    pp += 1
            else:
                pass

        # constrain alphas to parent's alphas
        parent_ptr = [prevatt_ptr_e[-2][-1] for prevatt_ptr_e in self.prevatt_ptr]
        parent_ptr = torch.tensor(parent_ptr).long().to(self.prevatts_probs.device)

        if self.hard is True:   # no backprop through alphas
            # parent_alphas_probs = self.prevatts_probs.gather(1, parent_ptr.unsqueeze(-1).unsqueeze(-1)
            #                                                  .repeat(1, 1, self.prevatts_probs.size(-1))).squeeze(1)
            parent_alphas_samples = self.prevatts_samples.gather(1, parent_ptr.unsqueeze(-1).unsqueeze(-1)
                                                             .repeat(1, 1, self.prevatts_samples.size(-1))).squeeze(1)
            alphas_samples = torch.min(parent_alphas_samples, alphas_samples)
            # save parent-masked samples
            self.prevatts_samples = torch.cat([self.prevatts_samples, alphas_samples.unsqueeze(1)], 1)
            self.prevatts_mask = torch.cat([self.prevatts_mask, parent_alphas_samples.unsqueeze(1)], 1)
            alphas = alphas_samples
        else:   # need to backprop differently
            parent_alphas_probs = self.prevatts_probs.gather(1, parent_ptr.unsqueeze(-1).unsqueeze(-1)
                                                             .repeat(1, 1, self.prevatts_probs.size(-1))).squeeze(1)
            alphas_probs = parent_overlap_f(parent_alphas_probs, alphas_probs)
            alphas = alphas_probs

        # append current alpha probs to prevatts accumulator
        self.prevatts_probs = torch.cat([self.prevatts_probs, alphas_probs.unsqueeze(1)], 1)

        # compute summary
        values = ctx if values is None else values
        summary = self.summcomp(values, alphas)

        return alphas, summary, scores


class PhraseAttentionTeacher(Attention):       # for depth-first decoding
    """ Normal attention. Stores probs for the batch for use to supervise real phrase attention. """
    """ Assumes masking by termination of tree structure assuming single root (which is also start token) """
    def __init__(self, attcomp:AttComp=None, hard=True, **kw):
        score_norm = torch.nn.Softmax(-1)
        summcomp = SumSummComp()
        super(PhraseAttentionTeacher, self).__init__(attcomp=attcomp, summcomp=summcomp, score_norm=score_norm)
        self.prevatts_probs = None            # (batsize, declen_so_far, enclen)
        self.prevatt_ptr = None         # for every example, keeps a list of pointers to positions in prevatts
        self.prevatts_masks = None
        # structure: batsize x stackdepth x numsiblings
        # For every example, the stack contains groups of siblings.
        # Could have had just batsize x stackdepth, but need to remember siblings for sibling overlap penalty (see prevatt_siblings)
        self.record = False
        self.hard = hard

    def batch_reset(self):
        self.prevatts_probs, self.prevatt_ptr = None, None
        self.prevatts_masks = None

    def get_phraseatt_supervision(self, hard=True):
        """
        Propagates attentions in self.prevatts_probs from children to parents according to self.prevatt_ptr, this way
        converting softmax attentions generated here to a supervision signal usable for sigmoid attention.
        If hard, does argmax before propagating, else propagates probs.
        """
        return self.prevatts_probs, self.prevatts_masks

    def forward(self, qry, ctx, ctx_mask=None, values=None, prev_pushpop=None):
        """
        :param qry:     (batsize, dim)
        :param ctx:     (batsize, seqlen, dim)
        :param ctx_mask: (batsize, seqlen)
        :param values:  (batsize, seqlen, dim)
        :param prev_pushpop: (batsize,) - whether PREVIOUS token was a push or pop (or do-nothing) token:
                                -N ==> pop (N=how many to pop),
                                0 ==> nothing,
                                +N ==> push (N doesn't matter, always pushes one)
                ! push/pop happens AFTER the element
        :return:
        """
        alphas_probs, summary, scores = super(PhraseAttentionTeacher, self).forward(qry, ctx, ctx_mask=ctx_mask, values=values)

        if self.record is True:
            if self.prevatts_probs is None:   # there is no history
                # initialize prevatts (history)
                self.prevatts_probs = torch.zeros_like(alphas_probs).unsqueeze(1)
                # initialize prevatt masks
                self.prevatts_masks = torch.ones_like(alphas_probs).unsqueeze(1)
                if ctx_mask is not None:
                    self.prevatts_masks = self.prevatts_masks * ctx_mask.unsqueeze(1).float()
                # initialize prevatt_ptr
                self.prevatt_ptr = [[[]] for _ in range(len(prev_pushpop))]


            # update pointers to prevatt
            k = self.prevatts_probs.size(1) - 1       # index of the last produced attention alphas (that were used for prev token)
            for i in range(len(prev_pushpop)):      # iterate over all examples
                self.prevatt_ptr[i][-1].append(k)   # make last token a sibling of the children of the same parent before it (if any)
                if prev_pushpop[i].cpu().item() > 0:  # PUSH: previous token requires children --> make stack deeper by one level
                    self.prevatt_ptr[i].append([])
                elif prev_pushpop[i].cpu().item() < 0:    # POP: previous token was last in its row of siblings (and possibly terminates upwards levels too)
                    pp = prev_pushpop[i].cpu().item()
                    while pp < 0 and len(self.prevatt_ptr[i]) > 1:
                        siblings = self.prevatt_ptr[i].pop(-1)      # pop the list from stack
                        # add each of the sibling's attention probs to their parent and populate children's masks
                        parent = self.prevatt_ptr[i][-1][-1]
                        for sibling in siblings:
                            sibling_alphas = self.prevatts_probs[i, sibling]
                            self.prevatts_probs[i, parent] += sibling_alphas
                            self.prevatts_probs[i, parent].clamp_(0., 1.)
                        parent_alphas = self.prevatts_probs[i, parent]
                        for sibling in siblings:
                            self.prevatts_masks[i, sibling] = parent_alphas
                        pp += 1
                else:
                    pass

            # append current alpha probs to prevatts accumulator
            if self.hard:
                # _alphas_probs = torch.zeros_like(alphas_probs)\
                #                     .scatter_(1, torch.argmax(alphas_probs, 1, True), 1.)
                alphas_dist = torch.distributions.OneHotCategorical(probs=alphas_probs)
                _alphas_probs = alphas_dist.sample()
            else:
                _alphas_probs = alphas_probs
            self.prevatts_probs = torch.cat([self.prevatts_probs, _alphas_probs.unsqueeze(1)], 1)
            self.prevatts_masks = torch.cat([self.prevatts_masks, torch.zeros_like(_alphas_probs.unsqueeze(1))], 1)     # will be filled once siblings have been done

        return alphas_probs, summary, scores



def test_phrase_attention(lr=0):
    # simulate operation of attention
    ctx = torch.randn(2, 5, 4)
    qrys = torch.randn(2, 6, 4)
    ctx_mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    pushpop = torch.tensor([[1,0,1,0,-1,-1], [1,1,1,1,-4,0]])
    # pushpop = list(zip(*pushpop))

    m = PhraseAttention(hard=True)

    for i in range(qrys.size(1)):
        alphas, summary, scores = m(qrys[:, i], ctx, ctx_mask=ctx_mask, prev_pushpop=pushpop[:, i])

    overlap = m.get_sibling_overlap()
    pass


def test_phrase_attention_teacher(lr=0):
    # simulate operation of attention
    ctx = torch.randn(2, 5, 4)
    qrys = torch.randn(2, 6, 4)
    ctx_mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    pushpop = torch.tensor([[1,0,1,0,-1,-1], [1,1,1,1,-4,0]])
    # pushpop = list(zip(*pushpop))

    m = PhraseAttentionTeacher(hard=True)
    m.record = True

    for i in range(qrys.size(1)):
        alphas, summary, scores = m(qrys[:, i], ctx, ctx_mask=ctx_mask, prev_pushpop=pushpop[:, i])

    print(m.prevatts_probs[0])
    print(m.prevatts_masks[0])
    print(m.prevatts_probs[1])
    print(m.prevatts_masks[1])

    overlap = m.get_phraseatt_supervision(hard=True)
    pass
# endregion


# region components for phrase attention
class LSTMAttComp(AttComp):
    def __init__(self, qrydim=None, ctxdim=None, encdim=None, dropout=0., numlayers=1, bidir=False, **kw):
        super(LSTMAttComp, self).__init__(**kw)
        encdims = [encdim] * numlayers
        self.layers = q.LSTMEncoder(qrydim+ctxdim, *encdims, bidir=bidir, dropout_in=dropout)
        self.lin = torch.nn.Linear(encdim, 1)

    def forward(self, qry, ctx, ctx_mask=None):
        """
        :param qry:     (batsize, qrydim)
        :param ctx:     (batsize, seqlen, ctxdim)
        :param ctx_mask:    (batsize, seqlen)
        :return:
        """
        inp = torch.cat([ctx, qry.unsqueeze(1).repeat(1, ctx.size(1), 1)], 2)
        out = self.layers(inp, mask=ctx_mask)
        ret = self.lin(out).squeeze(-1)     # (batsize, seqlen)
        return ret


class LSTMSummComp(SummComp):
    def __init__(self, valdim=None, encdim=None, dropout=0., numlayers=1, **kw):
        super(LSTMSummComp, self).__init__(**kw)
        encdims = [encdim] * numlayers
        self.layers = q.LSTMCellEncoder(valdim, *encdims, bidir=False, dropout_in=dropout)

    def forward(self, values, alphas):
        _, out = self.layers(values, gate=alphas, ret_states=True)
        out = out[:, 0]

        skip = values * alphas.unsqueeze(-1)
        skip = skip.sum(1)

        out = out + skip
        return out


class PooledLSTMSummComp(SummComp):
    """
    Uses a bidirectional lstm encoder with skip connections to encode according to given mask, not updating the state if gate is zero.
    If valdim != encdim * 2, uses a linear projection on the values.
    After encoding, does max and mean pooling across time, weighted by the provided attention weights.
    Best use only with hard attention alphas.
    """
    def __init__(self, valdim=None, encdim=None, dropout=0., numlayers=1, **kw):
        super(PooledLSTMSummComp, self).__init__(**kw)
        encdims = [encdim] * numlayers
        self.layers = q.LSTMCellEncoder(valdim, *encdims, bidir=True, dropout_in=dropout)
        self.skip_adapt = torch.nn.Linear(valdim, encdim*2) if valdim != encdim * 2 else lambda x: x

    def forward(self, values, alphas):
        """
        :param values:  (batsize, seqlen, valdim)
        :param alphas:  (batsize, seqlen)
        :return:        (batsize, seqlen, encdim*2*2)
        """
        topouts, out = self.layers(values, gate=alphas, ret_states=True)
        skip_vals = self.skip_adapt(values)

        rnnouts = topouts + skip_vals
        rnnouts = rnnouts * alphas.unsqueeze(2)

        meanpool = rnnouts.sum(1) / (alphas.unsqueeze(2).sum(1) + 1e-6)
        maxpool = rnnouts#.clone()
        maxpool.masked_fill_((1 - alphas).byte().unsqueeze(2), -np.infty)
        maxpool = maxpool.max(1)[0]
        maxpool = q.inf2zero(maxpool)

        out = torch.cat([meanpool, maxpool], 1)     # (batsize, encdim * 2 * 2)
        return out
        #
        # out = out[:, 0]
        #
        # skip = values * alphas.unsqueeze(-1)
        # skip = skip.sum(1)
        #
        # out = out + skip
        # return out


def test_pooled_lstm_summ_comp(lr=0.):
    vals = torch.randn(2, 5, 8)
    vals.requires_grad = True
    alphas = (torch.rand(2, 5) > 0.8).float()
    print(alphas)
    m = PooledLSTMSummComp(valdim=8, encdim=4)
    out = m(vals, alphas)
    print(out.size())
    l = out.sum()
    l.backward()
    print(vals.grad)


class LSTMPhraseAttention(PhraseAttention):
    def __init__(self, qrydim=None, ctxdim=None, valdim=None, encdim=None, dropout=0., numlayers=1, hard=False, **kw):
        ctxdim = qrydim if ctxdim is None else ctxdim
        valdim = ctxdim if valdim is None else valdim
        encdim = ctxdim if encdim is None else encdim
        attcomp = FwdAttComp(qrydim=qrydim, ctxdim=ctxdim, encdim=encdim, dropout=dropout, numlayers=numlayers)
        summcomp = LSTMSummComp(valdim=valdim, encdim=encdim, dropout=dropout, numlayers=numlayers)
        super(LSTMPhraseAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, hard=hard, **kw)


class PooledLSTMPhraseAttention(PhraseAttention):
    def __init__(self, qrydim=None, ctxdim=None, valdim=None, encdim=None, dropout=0., numlayers=1, hard=False, **kw):
        ctxdim = qrydim if ctxdim is None else ctxdim
        valdim = ctxdim if valdim is None else valdim
        encdim = ctxdim if encdim is None else encdim
        attcomp = FwdAttComp(qrydim=qrydim, ctxdim=ctxdim, encdim=encdim, dropout=dropout, numlayers=numlayers)
        summcomp = PooledLSTMSummComp(valdim=valdim, encdim=encdim, dropout=dropout, numlayers=numlayers)
        super(PooledLSTMPhraseAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, hard=hard, **kw)


class PhraseAttentionDecoderCell(torch.nn.Module):     # Luong-style decoder cell
    """ Need to subclass this, implementing get_pushpop_from for specific vocabulary. Or specify mapping id2pushpop during construction. """
    def __init__(self, emb=None, core=None, att:PhraseAttention=None, merge:q.rnn.DecCellMerge=q.rnn.ConcatDecCellMerge(),
                 out=None, feed_att=False, return_alphas=False, return_scores=False, return_other=False,
                 dropout=0, id2pushpop=None, **kw):
        """
        Based on LuongCell, only change: support for prev_pushpop arg in forward --> passed to attention
        :param emb:
        :param core:
        :param att:
        :param merge:
        :param out:         if None, out_vec (after merge) is returned
        :param feed_att:
        :param h_hat_0:
        :param id2pushpop:    torch tensor mapping token ids to pushpop values
        :param kw:
        """
        super(PhraseAttentionDecoderCell, self).__init__(**kw)
        self.emb, self.core, self.att, self.merge, self.out = emb, core, att, merge, out
        self.feed_att = feed_att
        self._outvec_tm1 = None
        self.outvec_t0 = None
        self.return_alphas = return_alphas
        self.return_scores = return_scores
        self.return_other = return_other
        self._id2pushpop = id2pushpop               # THIS LINE IS ADDED
        self.dropout = torch.nn.Dropout(dropout)

    def batch_reset(self):
        self.outvec_t0 = None
        self._outvec_tm1 = None

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        assert (ctx is not None)

        embs = self.emb(x_t)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        if self.feed_att:
            if self._outvec_tm1 is None:
                assert (self.outvec_t0 is not None)   #"h_hat_0 must be set when feed_att=True"
                self._outvec_tm1 = self.outvec_t0
            core_inp = torch.cat([embs, self._outvec_tm1], 1)
        else:
            core_inp = embs

        prev_pushpop = self.get_pushpop_from(x_t)           # THIS LINE IS ADDED

        core_out = self.core(core_inp)

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx, prev_pushpop=prev_pushpop)   # THIS LINE IS CHANGED
        out_vec = self.merge(core_out, summaries, core_inp)
        out_vec = self.dropout(out_vec)
        self._outvec_tm1 = out_vec      # store outvec

        ret = tuple()
        if self.out is None:
            ret += (out_vec,)
        else:
            _out_vec = self.out(out_vec)
            ret += (_out_vec,)

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


def test_lstm_phrase_attention(lr=0):
    m = LSTMPhraseAttention(4)
    ctx = torch.randn(2, 5, 4)
    qrys = torch.randn(2, 6, 4)
    ctx_mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    pushpop = [[1,0,1,0,-1,-1],     # output of last step will be "masked"
               [1,1,1,1,-4,0]]      # output of last two steps will be "masked"
    pushpop = torch.tensor(pushpop)
    # pushpop = list(zip(*pushpop))

    for i in range(qrys.size(1)):
        alphas, summary, scores = m(qrys[:, i], ctx, ctx_mask=ctx_mask, prev_pushpop=pushpop[:, i])

    overlap = m.get_sibling_overlap()
    pass

# endregion

if __name__ == '__main__':
    # q.argprun(test_custom_f)
    # q.argprun(test_phrase_attention)
    # q.argprun(test_phrase_attention_teacher)
    # q.argprun(test_lstm_phrase_attention)
    # q.argprun(test_pooled_lstm_summ_comp)
    q.argprun(test_rel_attention)