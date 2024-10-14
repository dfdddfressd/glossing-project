from collections import Counter, defaultdict
import os
import re
import sys
import argparse

from searching_string import readData

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold")
    parser.add_argument("--pred")
    args = parser.parse_args()
    return args

def getTag(gloss):
    elements = re.split("[.-]", gloss)
    elements = [ei for ei in elements if not ei.islower() and not ei.istitle()]
    return "-".join(elements)

if __name__ == "__main__":
    args = get_args()

    devSents = readData(args.gold)
    proposedSents = readData(args.pred)

    counts = Counter()
    
    for (dev, prop) in zip(devSents, proposedSents):
        sent, gloss, tr, _ = dev
        pSent, pGloss, pTr, _ = prop

        for (gi, pGI) in zip(gloss.split(), pGloss.split()):
            tagGI = getTag(gi)
            tagPGI = getTag(pGI)

            if tagGI != tagPGI:
                key = tuple(sorted( (tagGI, tagPGI) ))
                counts[key] += 1
            
    for pair, count in counts.most_common():
        if count > 0:
            print(pair, count)
