import re
import os
import numpy as np
from tqdm import tqdm
import pickle as pkl


def run(predp="exp_bert_both_23",
        outfile="rerank_out.{}.txt",
        entcandfile="entcands.{}.pkl",
        which="test",
        goldp="../../data/buboqa/data/processed_simplequestions_dataset/{}.txt"):
    outp = os.path.join(predp, outfile.format(which))
    candp = os.path.join(predp, entcandfile.format(which))
    goldp = goldp.format(which if which == "test" else "valid")
    predlines = open(outp, "r", encoding="utf8").readlines()
    entcands = pkl.load(open(candp, "rb"))
    goldlines = open(goldp, "r", encoding="utf8").readlines()
    entacc, relacc, allacc, total = 0, 0, 0, 0
    entwrong = []
    relwrong = []
    bothwrong = []
    anywrong = []

    i = 0
    for predline, goldline in tqdm(zip(predlines, goldlines)):
        predent, predrel = predline.strip().split("\t")
        predent, predrel = predent.strip(), predrel.strip()
        goldsplits = goldline.strip().split("\t")
        goldent, goldrel = goldsplits[1].strip(), goldsplits[3].strip()
        entacc += float(predent == goldent)
        relacc += float(predrel == goldrel)
        allacc += float(predent == goldent and predrel == goldrel)
        anybad = True
        cands_i = set([ecie["entry"]["uri"] for ecie in entcands[i]])
        entincands = goldent in cands_i
        if predent != goldent:
            if predrel != goldrel:
                bothwrong.append((goldline, predline, "goldentincands:{}".format(entincands)))
            else:
                entwrong.append((goldline, predline, "goldentincands:{}".format(entincands)))
        elif predrel != goldrel:
            relwrong.append((goldline, predline, "goldentincands:{}".format(entincands)))
        else:
            anybad = False
        if anybad:
            anywrong.append((goldline, predline, "goldentincands:{}".format(entincands)))

        # if predent != goldent or predrel != goldrel:
            # print(predline.strip(), goldline.strip())
            # print(predent, goldent, predrel, goldrel)
            # print()
        total += 1.
        i += 1



    print("{:.3}% total acc\n\t - {:.3}% ent acc\n\t - {:.3}% rel acc".format(
        allacc * 100 / total,
        entacc * 100 / total,
        relacc * 100 / total
    ))

    print("anywrong: {:.3}%".format(len(anywrong) * 100 / total))
    print("{:.3}% both rel and ent wrong\n\t - {:.3}% only ent wrong\n\t - {:.3}% only rel wrong".format(
        len(bothwrong) * 100 / len(anywrong),
        len(entwrong) * 100 / len(anywrong),
        len(relwrong) * 100/ len(anywrong)))

    for errcat, errcatname in [(bothwrong, "bothwrong"), (entwrong, "entwrong"), (relwrong, "relwrong"), (anywrong, "anywrong"), (bothwrong+entwrong, "allentwrong")]:
        print(f"doing {errcatname}")
        errcat_total = 0
        errcat_gold_not_in_cand = 0
        for ec_goldline, ec_predline, goldinfo in errcat:
            errcat_total += 1
            if goldinfo == "goldentincands:False":
                errcat_gold_not_in_cand += 1
            else:
                assert(goldinfo == "goldentincands:True")
        print(f"gold ent not in cands (%): {errcat_gold_not_in_cand * 100 /errcat_total}")

    gentthere = {"bothwrong": 0, "entwrong": 0, "relwrong": 0}
    gentnotthere = {"bothwrong": 0, "entwrong": 0, "relwrong": 0}
    gentthere_total = 0
    gentnotthere_total = 0
    for errcat, errcatname in [(bothwrong, "bothwrong"), (entwrong, "entwrong"), (relwrong, "relwrong")]:
        for _, _, goldinfo in errcat:
            if goldinfo == "goldentincands:False":
                gentnotthere_total += 1
                gentnotthere[errcatname] += 1
            else:
                assert(goldinfo == "goldentincands:True")
                gentthere_total += 1
                gentthere[errcatname] += 1

    print("if gold entity is in cands:")
    print("\n".join(["{}: {:.3}".format(k, v * 100 / gentthere_total) for k, v in gentthere.items()]))
    print("if gold entity is NOT in cands:")
    print("\n".join(["{}: {:.3}".format(k, v*100/gentnotthere_total) for k, v in gentnotthere.items()]))






if __name__ == '__main__':
    run()