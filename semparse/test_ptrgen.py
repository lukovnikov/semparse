from unittest import TestCase
from semparse import rnn
import torch
import qelos as q


class TestPointerGeneratorOut(TestCase):
    def test_it(self):
        nlD = "cat dog fancy code fox bear sing aaa bb vvv ddd dsss".split()
        flD = "cat dog A B C".split()
        nlD = dict(zip(nlD, range(len(nlD))))
        flD = dict(zip(flD, range(len(flD))))
        nextflDid = max(flD.values()) + 1
        sourcemap = torch.zeros(len(nlD), dtype=torch.long)
        for k, v in nlD.items():
            if k not in flD:
                flD[k] = nextflDid
                nextflDid += 1
            sourcemap[v] = flD[k]

        class FakeOut(torch.nn.Module):
            def __init__(self):
                super(FakeOut, self).__init__()
                self.out = torch.zeros(4, len(flD))
                self.out.requires_grad = True

            def forward(self, *input, **kw):
                return self.out

        out = FakeOut()

        class FakeGate(torch.nn.Module):
            def __init__(self):
                super(FakeGate, self).__init__()
                self.out = torch.ones(4, 2) * .5
                self.out.requires_grad = True

            def forward(self, *input, **kw):
                return self.out

        gate = FakeGate()
        m = rnn.PointerGeneratorOut(out, sourcemap, gate)

        ctx_ids = torch.arange(0, 4, dtype=torch.long)
        ctx_ids = ctx_ids.unsqueeze(0).repeat(4, 1)
        m.ctx_ids = ctx_ids
        x = torch.randn(ctx_ids.size(0), 5)
        x.requires_grad = True
        scores = torch.zeros_like(ctx_ids).float()
        scores.requires_grad = True
        y = m(x, scores)
        yg = torch.zeros_like(y)
        yg.requires_grad = True
        y += yg

        l = q.loss.CELoss(mode="probs")
        g = torch.tensor([1, 2, 6, 6])
        loss = l(y, g)
        loss.backward()
        # print(x.grad)
        print(scores.grad)
        print(gate.out.grad)
        print(yg.grad)
        print(out.out.grad)

        oldyg = torch.tensor(yg.detach().numpy())

        optim = torch.optim.SGD([yg], lr=1.)
        optim.step()

        # print(yg - oldyg)
        #
        # print()
        #
        # pass

