import numpy as np
import torch
import qelos as q


def run(preepochs=0,
        epochs=1000):
    np.set_printoptions(precision=3, suppress=True)
    torch.random.manual_seed(1337)
    # random data
    x = torch.randn(50, 10)
    m = torch.nn.Sequential(
        torch.nn.Linear(10, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 6),
        torch.nn.Softmax()
    )
    g = torch.randint(0, 6, (50, 1))
    print(g[:10])
    # y = m(x)
    # print(y)
    # print(torch.gather(y, 1, g))

    optim = torch.optim.SGD(m.parameters(), lr=0.1)

    # pretrain to favour first class
    for i in range(preepochs):
        m.zero_grad()
        y = m(x)
        l = - torch.log(torch.gather(y, 1, g).sum(1).clamp_min(1e-6))
        l.sum().backward()
        optim.step()

    print(m(x)[:10].detach().cpu().numpy())

    # train with mml
    g2 = torch.randint(0, 6, (50, 2))
    g = torch.cat([g, g2], 1)
    for i in range(epochs):
        m.zero_grad()
        y = m(x)
        l = - torch.log(torch.gather(y, 1, g).clamp_min(1e-6))
        # l = - torch.log(torch.gather(y, 1, g).sum(1).clamp_min(1e-6))
        l.sum().backward()
        optim.step()

    print(g[:10])
    print(m(x)[:10].detach().cpu().numpy())

if __name__ == '__main__':
    run()