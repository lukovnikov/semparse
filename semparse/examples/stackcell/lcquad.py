import torch
import qelos as q
import numpy as np
import json
import os
import re


class Node(object):
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
        ysm.add(l.to_transitions())
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



def run(lr=0.001,
        ):
    xsm, ysm, teststart, tok2act = load_data()

    print("Some examples:")
    for i in range(10):
        print(xsm[i], ysm[i])

    print("Non-leaf tokens:")
    print({ysm.RD[k]: v for k, v in tok2act.items() if v > 0})


if __name__ == '__main__':
    q.argprun(run)