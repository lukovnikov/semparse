import pickle as pkl
import numpy as np
import os
from tqdm import tqdm


""" From candidates and predicate probabilities and connectivity, generate predictions and save """

# TODO: exclude relations that have not been seen during training
def run(borderp="exp_bert_both_8",
        predp="exp_bert_both_8",
        dp="../../data/buboqa/data/bertified_dataset.npz",
        reachp="../../data/buboqa/indexes/reachability_2M.pkl",
        topk=50,
        outf="output.txt"):
    # region load data
    canp = os.path.join(borderp, "cands.test.pkl")
    relp = os.path.join(predp, "relpreds.npy")

    candidates = pkl.load(open(canp, "rb"))
    relationprobs = np.load(relp)
    data = np.load(dp)
    entreaches = pkl.load(open(reachp, "rb"))
    relreaches = {}
    for k, v in entreaches.items():
        for ve in v:
            if ve not in relreaches:
                relreaches[ve] = set()
            relreaches[ve].add(k)
    relD = data["relD"].item()
    revrelD = {v: k for k, v in relD.items()}
    print(data.keys())
    print("{} unique relation in all data".format(len(relD)))
    # endregion

    print(relationprobs.shape)

    # check and transform loaded data
    predictions = []
    for i in tqdm(range(len(candidates))):
        cands_i = candidates[i]
        allowedrels_i = set()
        for c in cands_i[:topk]:
            if c in entreaches:
                rels_from_c = entreaches[c]
            else:
                rels_from_c = set()
            allowedrels_i |= rels_from_c
        relprobs_i = relationprobs[i]
        relallow_i = np.zeros_like(relprobs_i)
        for allowedrel in allowedrels_i:
            if allowedrel in relD:
                relallow_i[relD[allowedrel]] = 1
        relprobs_i = relprobs_i + np.log(relallow_i)
        # best relation that is reachable from any of the topk candidates
        bestrel_i = np.argmax(relprobs_i)
        bestrel_i = revrelD[bestrel_i]
        # print(bestrel_i)
        # print(len(allowedrels_i))

        # getting best entity that has the chosen relation
        bestent_i = cands_i[0] if len(cands_i) > 0 else "https://www.youtube.com/watch?v=DLzxrzFCyOs"
        for c in cands_i[:topk]:
            if c in relreaches[bestrel_i]:
                bestent_i = c
                break
        # print(bestent_i, bestrel_i)
        predictions.append((bestent_i, bestrel_i))

    # print to output
    with open(os.path.join(borderp, outf), "w") as f:
        for prediction in predictions:
            f.write("{}\t{}\n".format(prediction[0], prediction[1]))

    print("file written")






if __name__ == '__main__':
    run()