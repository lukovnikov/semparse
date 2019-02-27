import json
import pickle as pkl
import numpy as np
import re
from unidecode import unidecode
from tabulate import tabulate
import qelos as q


def run(indexp="../../data/buboqa/indexes/",
        datap="../../data/buboqa/data/"):
    names = pkl.load(open(indexp + "names_2M.pkl", "rb"))
    # entities = pkl.load(open(indexp + "entity_2M.pkl", "rb"))
    print(len(names))


if __name__ == '__main__':
    q.argprun(run)