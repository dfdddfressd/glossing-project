from collections import Counter, defaultdict
import os
import re
import sys
import argparse
import json

from searching_string import readData, getTag

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold")
    parser.add_argument("--pred")
    parser.add_argument("--language")
    parser.add_argument("--cutoff", default=3, type=int)
    args = parser.parse_args()
    return args

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

    confTags = {}
    for pair, count in counts.most_common():
        if count > 0:
            print(pair, count)

        if count > args.cutoff:
            confTags[pair[0]] = pair[1]
            confTags[pair[1]] = pair[0]

    print("Confused tag dictionary:")
    print(confTags)

    with open(f"outputs/{args.language}/confusions.json", "w") as ofh:
        json.dump(confTags, ofh)
