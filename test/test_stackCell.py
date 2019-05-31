from unittest import TestCase
import qelos as q
from semparse.stackcell import StackCell, BasicCombiner, StackCellCombiner
from semparse.attention import BasicAttention
import torch


class TestStackCell(TestCase):
    def test_it(self):
        D = "<MASK> [RED] NT(START) NT(a) T(b) NT(c) T(d) T(e) NT(f) T(g) T(h) T(i)"
        D = dict(zip(D.split(), range(len(D.split()))))
        tok2act = {k: (2 if k == "[RED]" else 1 if k[:2] == "NT" else 0) for k in D}


        class CustomCombiner(StackCellCombiner):
            def forward(self, _x, mask):
                ret = (_x * mask.unsqueeze(-1).float()).sum(1) / mask.float().sum(1).unsqueeze(-1).clamp_min(1e-6)
                ret = ret.detach()      # TODO: for grad debugging
                return ret

        class CustomWordLinout(q.WordLinout):
            def update(self, _):
                pass

        class Tok2Act(torch.nn.Module):
            def __init__(self, t2a, D):
                super(Tok2Act, self).__init__()
                self.D = D
                t2a_ = torch.zeros(max(D.values()) + 1).long()
                for k, v in t2a.items():
                    t2a_[D[k]] = v
                self.register_buffer("t2a", t2a_)

            def forward(self, _x):
                return self.t2a[_x]

        embdim = 4
        coredim = 5
        emb = q.WordEmb(embdim, worddic=D)
        core = q.LSTMCell(embdim, coredim, dropout_rec=.1)
        # combiner = BasicCombiner(embdim)
        combiner = CustomCombiner()
        att = BasicAttention()
        out = CustomWordLinout(coredim*2, worddic=D)
        tok2act = Tok2Act(tok2act, D)

        cell = StackCell(emb=emb, tok2act=tok2act, core=core, combiner=combiner, att=att, out=out)
        ctx = torch.randn(2, 6, coredim)
        cell.save_ctx(ctx)

        ex1 = "NT(START) NT(a) T(b) NT(c) T(d) T(e) [RED] NT(f) T(g) T(h) [RED] T(i) [RED]"
        ex2 = "NT(START) NT(a) NT(c) T(d) T(e) [RED] [RED]"
        x1 = [D[exi] for exi in ex1.split()] + [0]
        x2 = [D[exi] for exi in ex2.split()]
        x2 = x2 + [0]*(len(x1) - len(x2))
        x = torch.tensor([x1, x2])

        cell._debug_embs = torch.nn.Parameter(torch.zeros(2, len(x1), embdim))

        ys = []
        for i in range(len(x[0])):
            y = cell(x[:, i])
            ys.append(y)

        # print(cell._debug_embs)
        print(cell._debug_embs.size())
        l = ys[11][0].sum()
        l.backward()

        print(cell._debug_embs.grad)
        print(cell._stack)
