import torch
import qelos as q
from abc import ABC, abstractmethod
from semparse import attention as att


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


class StackCellCombiner(torch.nn.Module, ABC):
    """
    Combines parent and child embeddings into subtree embedding.
    """
    @abstractmethod
    def forward(self, x, mask):     # x is (batsize, numargs, dim), mask (batsize, numargs)
                                    # x[:, 0] is parent embedding, following embeddings are children embeddings
        pass


class BasicCombiner(StackCellCombiner):
    """
    Combines parent and child embeddings into subtree embedding.
    y = W * [p, c] where p is parent embedding and c is average of child embeddings
    """
    def __init__(self, dim, **kw):
        super(BasicCombiner, self).__init__(**kw)
        self.W = torch.nn.Linear(dim*2, dim, bias=False)

    def forward(self, x, mask):         # (batsize, numchildren, dim), (batsize, numchildren) --> (batsize, dim)
        x_ = x * mask.float().unsqueeze(2)
        parents = x_[:, 0]
        children = x_[:, 1:].sum(1) / mask[:, 1:].float().sum(1).unsqueeze(1).clamp_min(1e-6)     # (batsize, dim)
        inp = torch.cat([parents, children], 1)
        ret = self.W(inp)
        return ret


class ForwardCombiner(StackCellCombiner):
    def __init__(self, dim, maxnumchildren=4, **kw):
        super(ForwardCombiner, self).__init__(**kw)
        pass


class GLUishCombiner(StackCellCombiner):
    pass        # TODO: implement
                # - assumes few children (<4/5)
                # - computes a gate and a transform


class StackCell(torch.nn.Module, q.Stateful):
    statevars = ["_outvec_tm1", "outvec_t0", "_saved_ctx", "_saved_ctx_mask", "_core_state_history", "_combiner_history", "_stack", "_t"]
    TERMINAL = 0
    NONTERMINAL = 1
    REDUCE = 2

    RET_NORMAL = ["ret_normal"]
    RET_ALPHAS = ["alphas"]
    RET_SCORES = ["scores"]
    RET_OTHER =  ["embs", "core_out", "summaries"]

    def __init__(self,
                 emb=None,          # embeds input tokens and determines stack action
                 tok2act=None,
                 core=None,         # performs update
                 combiner:StackCellCombiner=None,     # merges children and parent vectors into subtree vector
                 att=att.BasicAttention(),
                 merge:q.DecCellMerge=q.ConcatDecCellMerge(),
                 out=None,
                 feed_att=False,
                 dropout=0,
                 returns=None,
                 **kw):
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
        super(StackCell, self).__init__(**kw)
        self.emb, self.tok2act, self.core, self.att, self.merge, self.out, self.combiner \
            = emb, tok2act, core, att, merge, out, combiner
        self.feed_att = feed_att
        self._outvec_tm1 = None    # previous attention summary
        self.outvec_t0 = None
        self.returns = [self.RET_NORMAL] if returns is None else returns
        self.dropout = torch.nn.Dropout(dropout)
        self._t = 0
        self._saved_ctx, self._saved_ctx_mask = None, None
        self._core_state_history = None
        self._combiner_history = None   # for every timestep, a vector, either an embedding (self.emb) or combined embeddings (self.combiner)
        self._stack = []        # stack of frames. each frame is
                # (parent ts, [list of children ts])
                # RED -->

    def save_ctx(self, ctx, ctx_mask=None):
        self._saved_ctx, self._saved_ctx_mask = ctx, ctx_mask

    def batch_reset(self):
        self.outvec_t0 = None
        self._outvec_tm1 = None
        self._saved_ctx = None
        self._saved_ctx_mask = None
        self._core_state_history = None
        self._combiner_history = None
        self._stack = []
        self._t = 0

    def pre_core_update(self, x_t, embs):
        stackactions = self.tok2act(x_t)
        embs = self.update_combiner_history(embs, stackactions)
        self.load_core_states(stackactions)
        self.update_stack(stackactions)         # maintain stack
        return embs

    def update_combiner_history(self, embs, actions):
        """
        :param embs:        (batsize, embsize) float vectors embedding input tokens
        :param actions:     (batsize,) int ids of actions from tokens
        :returns            (batsize, embsize) float vectors of embeddings of structure output at previous timestep
        If t==0, returns embeddings given
        If t > 0, applies combiner for examples where RED was previous action and otherwise returns given embeddings
        If t > 0, also saves embedding vectors that it just produced for token/actions from previous time step
        ==> self._combiner_history[:, 0] is embedding of first token decoded by this system (in freerunning mode)
        """
        if self._t == 0:
            ret = embs
        else:
            # if previous action was REDUCE (==2), get children and parent, run combiner and output combined embedding
            # else, take embedding as given
            # then, concat embedding to history
            # return last embedding from history
            switcher = actions == 2
            actions_ = actions.detach().cpu().numpy()
            if torch.any(switcher).detach().cpu().item() == 1:
                frames = [[stack_e[-1][0]] + stack_e[-1][1] if (action == 2 and stack_e is not None)
                          else []
                          for (stack_e, action) in zip(self._stack, actions_)]
                comb_embs = self.do_combiner(frames)
                switcher = switcher.float().unsqueeze(1)
                ret = (1 - switcher) * embs + switcher * comb_embs
            else:
                ret = embs
            if self._combiner_history is None:
                self._combiner_history = ret.unsqueeze(1)
            else:
                self._combiner_history = torch.cat([self._combiner_history, ret.unsqueeze(1)], 1)
        return ret

    def do_combiner(self, frames):      # nested list (batsize, numargs+1) int ids for timesteps in combiner_history
        # convert frames to tensor
        maxlen = max([len(frame) for frame in frames])
        frames_ = [frame + [0]*(maxlen-len(frame)) for frame in frames]
        mask_ = [[1]*len(frame) + [0]*(maxlen-len(frame)) for frame in frames]
        frames_ = torch.tensor(frames_).to(self._combiner_history.device)   # (batsize, numargs+1)
        mask_ = torch.tensor(mask_).to(self._combiner_history.device)   # (batsize, numargs+1)
        frames_ = frames_.unsqueeze(2).repeat(1, 1, self._combiner_history.size(2))   # (batsize, numargs+1, embdim)
        # get arguments to combiner module
        args = torch.gather(self._combiner_history, 1, frames_)
        # run combiner module
        vecs = self.combiner(args, mask_)
        return vecs

    def load_core_states(self, actions):
        if self._t == 0:
            pass
        else:
            if torch.any(actions == self.REDUCE).detach().cpu().item() == 1:
                def get_state_element(state_e, indexes):    # (batsize, seqlen, ...), list of (batsize,)
                    if isinstance(state_e, torch.Tensor):   # (batsize, seqlen, ...) --> index along seqlen
                        indexes = [indexes_i if indexes_i > -1 else state_e.size(1)-1 for indexes_i in indexes]
                        indexes_ = torch.tensor(indexes).to(state_e.device)     # (batsize,)
                        indexes_ = indexes_.unsqueeze(1)
                        for size in state_e.size()[2:]:
                            reps = [1]*indexes_.dim() + [size]
                            indexes_ = indexes_.unsqueeze(-1).repeat(*reps)
                        ret = torch.gather(state_e, 1, indexes_).squeeze(1)
                        return ret
                    elif isinstance(state_e, list):
                        return [state_e[index] for state_e, index in zip(state_e, indexes)]
                    else:
                        raise Exception(f"Unsupported state type: {type(state_e)}")
                actions_ = actions.detach().cpu().numpy()
                idxs = [-1 if action != self.REDUCE or stack_e is None else stack_e[-1][0]
                        for (action, stack_e) in zip(actions_, self._stack)]
                loaded_states = {csi_k: get_state_element(csi_v, idxs) for (csi_k, csi_v) in self._core_state_history.items()}
                self.core.set_state(loaded_states)

    def update_stack(self, actions):        # updates stack based on actions from tokens from t-1
        actions_ = actions.detach().cpu().numpy()
        if self._t == 0:
            assert(self._stack == [])
            self._stack = [[] for _ in actions_]     # initialize stack
        else:
            for i, (action, stackitem) in enumerate(zip(actions_, self._stack)):
                if stackitem is not None:
                    if action == self.TERMINAL:
                        stackitem[-1][1].append(self._t - 1)        # new child to current parent
                    elif action == self.NONTERMINAL:
                        stackitem.append((self._t - 1, []))     # new parent --> new frame
                    elif action == self.REDUCE:
                        poppedframe = stackitem.pop(-1)         # current parent finished
                        if len(stackitem) > 0:
                            stackitem[-1][1].append(self._t - 1)        # new child is what the combiner made for the RED timestep (which was t-1)
                        else:
                            self._stack[i] = None


    def post_core_update(self):
        self.save_core_states()     # 0-th state is output state of first time step that produces first NT token
        self._t += 1

    def save_core_states(self):
        def map_state_element(state_element):
            if isinstance(state_element, torch.Tensor):
                return state_element.unsqueeze(1)
            elif isinstance(state_element, list):
                return [[state_element_i] for state_element_i in state_element]
            else:
                raise Exception(f"Unsupported state type: {type(state_element)}")

        def merge_state_elements(state_acc, state_elem):
            if isinstance(state_elem, torch.Tensor):
                return torch.cat([state_acc, state_elem], 1)
            elif isinstance(state_elem, list):
                return [state_acc_i + state_elem_i for (state_acc_i, state_elem_i) in zip(state_acc, state_elem)]
            else:
                raise Exception(f"Unsupported state type: {type(state_elem)}")

        core_states = self.core.get_state()
        if self._core_state_history is None:
            self._core_state_history = {csi_k: map_state_element(csi_v) for csi_k, csi_v in core_states.items()}
        else:
            self._core_state_history = {csi_k: merge_state_elements(self._core_state_history[csi_k], map_state_element(csi_v))
                                        for (csi_k, csi_v) in core_states.items()}

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        if ctx is None:
            ctx, ctx_mask = self._saved_ctx, self._saved_ctx_mask
        assert (ctx is not None)

        if self.out is not None and hasattr(self.out, "update"):
            self.out.update(x_t)

        embs = self.emb(x_t)        # embed input tokens
        if isinstance(self.emb, q.WordEmb):   # unpack if necessary
            embs, mask = embs

        embs = embs if not hasattr(self, "_debug_embs") else embs + self._debug_embs[:, self._t]

        embs = self.pre_core_update(x_t, embs)

        if self.feed_att:
            if self._outvec_tm1 is None:
                assert (self.outvec_t0 is not None)   #"h_hat_0 must be set when feed_att=True"
                self._outvec_tm1 = self.outvec_t0
            core_inp = torch.cat([embs, self._outvec_tm1], 1)     # append previous attention summary
        else:
            core_inp = embs

        core_out = self.core(core_inp)  # feed through rnn

        self.post_core_update()

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)  # do attention
        out_vec = self.merge(core_out, summaries, core_inp)
        out_vec = self.dropout(out_vec)
        self._outvec_tm1 = out_vec      # store outvec (this is how Luong, 2015 does it)

        if self.out is None:
            ret_normal = out_vec
        else:
            if isinstance(self.out, PointerGeneratorOut):
                _out_vec = self.out(out_vec, scores=scores)
            else:
                _out_vec = self.out(out_vec)
            ret_normal = _out_vec

        l = locals()
        ret = tuple([l[k] for k in sum(self.returns, [])])
        return ret[0] if len(ret) == 1 else ret


def test_training_stackcell(lr=0.001,
                            ):
    pass



if __name__ == '__main__':
    q.argprun(test_training_stackcell)