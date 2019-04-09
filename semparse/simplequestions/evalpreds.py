import re
import os
import numpy as np
from tqdm import tqdm


def run(predp="exp_bert_both_23/output.{}.txt",
        which="test",
        goldp="../../data/buboqa/data/processed_simplequestions_dataset/{}.txt"):
    predp = predp.format(which)
    goldp = goldp.format(which if which == "test" else "valid")
    predlines = open(predp, "r", encoding="utf8").readlines()
    goldlines = open(goldp, "r", encoding="utf8").readlines()
    entacc, relacc, allacc, total = 0, 0, 0, 0

    for predline, goldline in tqdm(zip(predlines, goldlines)):
        predent, predrel = predline.strip().split("\t")
        predent, predrel = predent.strip(), predrel.strip()
        goldsplits = goldline.strip().split("\t")
        goldent, goldrel = goldsplits[1].strip(), goldsplits[3].strip()
        entacc += float(predent == goldent)
        relacc += float(predrel == goldrel)
        allacc += float(predent == goldent and predrel == goldrel)
        # if predent != goldent or predrel != goldrel:
            # print(predline.strip(), goldline.strip())
            # print(predent, goldent, predrel, goldrel)
            # print()
        total += 1.

    print("{:.3}% total acc\n\t - {:.3}% ent acc\n\t - {:.3}% rel acc".format(
        allacc * 100 / total,
        entacc * 100 / total,
        relacc * 100 / total
    ))




if __name__ == '__main__':
    run()