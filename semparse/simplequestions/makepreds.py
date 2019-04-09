import pickle as pkl
import numpy as np
import os
from tqdm import tqdm
import torch
import re

""" From candidates and predicate probabilities and connectivity, generate predictions and save """


def get_ent_stats(p="../../data/buboqa/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt",
                  outp="degrees.2M.pkl"):
    entre = re.compile("www\.freebase\.com/(.+)")
    degrees = {}
    with open(p) as f:
        for line in tqdm(f):
            line = line.strip().split()
            line = [
                "fb:" + ".".join(entre.match(line_e).group(1).split("/")) for line_e in line
            ]
            s, p = line[0], line[1]
            os = line[2:]
            for o in os:
                for x in [s, o]:
                    if x not in degrees:
                        degrees[x] = {"in": 0, "out": 0}
                if p not in degrees:
                    degrees[p] = {"deg": 0}
                degrees[s]["out"] += 1
                degrees[o]["in"] += 1
                degrees[p]["deg"] += 1
                # print(line)

    maxindegree, maxoutdegree, maxreldegree = 0, 0, 0
    for fbid, v in degrees.items():
        if not "deg" in v:
            maxindegree = max(maxindegree, v["in"])
            maxoutdegree = max(maxoutdegree, v["out"])
        else:
            maxreldegree = max(maxreldegree, v["deg"])
    for fbid, v in degrees.items():
        if not "deg" in v:
            v["in"] = v["in"]/maxindegree
            v["out"] = v["out"]/maxoutdegree
        else:
            v["deg"] = v["deg"]/maxreldegree
    with open(outp, "wb") as f:
        pkl.dump(degrees, f)


# TODO: exclude relations that have not been seen during training
def run(borderp="exp_bert_both_23",
        predp="exp_bert_both_23",
        dp="../../data/buboqa/data/bertified_dataset_new.npz",
        reachp="../../data/buboqa/indexes/reachability_2M.pkl",
        entinfo="degrees.2M.pkl",
        topk=50,
        which="test",
        outf="output.{}.txt",
        candp="candinfo.{}.pkl",
        goldp="../../data/buboqa/data/processed_simplequestions_dataset/{}.txt"):
    outf = outf.format(which)
    goldp = goldp.format(which if which == "test" else "valid")
    # region load data
    canp = os.path.join(borderp, "entcands.{}.pkl".format(which))
    relp = os.path.join(predp, "relpreds.{}.npy".format(which))

    candidates = pkl.load(open(canp, "rb"))
    relationprobs = np.load(relp)
    relationprobs = torch.softmax(torch.tensor(relationprobs) / 3, 1).numpy()
    data = np.load(dp)
    entreaches = pkl.load(open(reachp, "rb"))
    # relreaches = {}
    # for k, v in entreaches.items():
    #     for ve in v:
    #         if ve not in relreaches:
    #             relreaches[ve] = set()
    #         relreaches[ve].add(k)
    relD = data["relD"].item()
    # revrelD = {v: k for k, v in relD.items()}
    print("{} unique relation in all data".format(len(relD)))
    goldlines = open(goldp, "r", encoding="utf8").readlines()
    entdegrees = pkl.load(open(entinfo, "rb"))
    # endregion

    entacc, relacc, allacc, total = 0, 0, 0, 0
    print(relationprobs.shape)

    # check and transform loaded data
    predictions = []
    for i in tqdm(range(len(candidates))):
        cands_i = candidates[i]
        cands_i = cands_i[:min(len(cands_i), topk)]       # retain at most 50 candidates
        rels_i = zip(sorted(relD.items(), key=lambda x: x[1]),
                     list(relationprobs[i]))
        rels_i = sorted(rels_i, key=lambda x: x[1], reverse=True)
        # get best rel for every candidate entity
        for c in cands_i:
            canduri = c["entry"]["uri"]
            if canduri in entreaches:
                rels_from_c = entreaches[canduri]
                for rel in rels_i:
                    if rel[0][0] in rels_from_c:
                        c["rel"] = {
                            "relid": rel[0][0],
                            "relprob": rel[1]
                        }
                        break
            else:
                rels_from_c = set()
            if "rel" not in c:
                c["rel"] = {"relid": None, "relprob": 0}

            # enrich with degree information
            if canduri in entdegrees:
                c["entry"]["indegree"] = entdegrees[canduri]["in"]
                c["entry"]["outdegree"] = entdegrees[canduri]["out"]
            else:
                c["entry"]["indegree"] = 0
                c["entry"]["outdegree"] = 0
            if c["rel"]["relid"] in entdegrees:
                c["rel"]["degree"] = entdegrees[c["rel"]["relid"]]["deg"]
            else:
                c["rel"]["degree"] = 0

        # region RERANK
        for cand in cands_i:
            entsim = cand["similarity"]
            entnumrels = cand["entry"]["pop"]
            relprob = cand["rel"]["relprob"]
            entindegree = cand["entry"]["indegree"]
            entoutdegree = cand["entry"]["outdegree"]
            reldegree = cand["rel"]["degree"]

            features = np.asarray([entsim/100., relprob, entnumrels, entindegree, entoutdegree, reldegree])
            weights = np.asarray([1e6, 1, 0, 0, 0, 0])

            fs = (features * weights).sum()

            # fs = entsim
            cand["final_score"] = fs

        cands_i = sorted(cands_i, key=lambda x: x["final_score"], reverse=True)
        # endregion
        best_pair = cands_i[0]
        bestent_i = best_pair["entry"]["uri"]
        bestrel_i = best_pair["rel"]["relid"]
        # print(bestent_i, bestrel_i)
        predictions.append((bestent_i, bestrel_i))

        # region DEBUG
        goldline = goldlines[i]
        goldsplits = goldline.strip().split("\t")
        goldent, goldrel = goldsplits[1].strip(), goldsplits[3].strip()

        for cand in cands_i:
            ds = cand["final_score"]
            # if cand["entry"]["uri"] == goldent:
            # if cand["rel"]["relid"] == goldrel:
            #     ds = 1.
            # else:
            #     ds = 0.
            cand["debug_score"] = ds

        cands_i = sorted(cands_i, key=lambda x: x["debug_score"], reverse=True)
        # endregion

        # region EVAL
        best = cands_i[0]
        if best["debug_score"] > 0.:
            predent, predrel = best["entry"]["uri"], best["rel"]["relid"]
        else:
            predent, predrel = "none", "none"
        entacc += float(predent == goldent)
        relacc += float(predrel == goldrel)
        allacc += float(predent == goldent and predrel == goldrel)
        total += 1.
        # endregion

    print("EVAL results after DEBUG")
    print("{:.3}% total acc\n\t - {:.3}% ent acc\n\t - {:.3}% rel acc".format(
        allacc * 100 / total,
        entacc * 100 / total,
        relacc * 100 / total
    ))

    # print to output
    with open(os.path.join(borderp, outf), "w") as f:
        for prediction in predictions:
            f.write("{}\t{}\n".format(prediction[0], prediction[1]))

    pkl.dump(candidates, open(candp.format(which), "wb"))

    print("file written")






if __name__ == '__main__':
    # get_ent_stats()
    run()