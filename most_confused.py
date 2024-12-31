from collections import Counter, defaultdict
import os
import re
import sys
import argparse
import json

from searching_string import *

from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold")
    parser.add_argument("--pred")
    parser.add_argument("--language")
    parser.add_argument("--use_lm", action="store_true")
    parser.add_argument("--exclude_empty_tag", action="store_true")
    parser.add_argument("--cutoff", default=3, type=int)
    parser.add_argument("--replicates", default=1, type=int)
    parser.add_argument("--singleton_features", action="store_true")
    parser.add_argument("--root_match_only", action="store_true")
    args = parser.parse_args()
    return args

def mostConfused(devSents, proposedSents, rootMatchOnly=False):
    counts = Counter()

    for (dev, prop) in zip(devSents, proposedSents):
        sent, gloss, tr, _ = dev
        pSent, pGloss, pTr, _ = prop

        for (gi, pGI) in zip(gloss.split(), pGloss.split()):
            tagGI = getTag(gi)
            tagPGI = getTag(pGI)

            if rootMatchOnly:
                rootGI = getRoot(gi)
                rootPGI = getRoot(pGI)
                if rootGI != rootPGI:
                    counts[("ROOT", "ROOT")] += 1
                    continue
            
            if tagGI != tagPGI:
                key = tuple(sorted( (tagGI, tagPGI) ))
                counts[key] += 1

    confFeatures = Counter()
    for pair, count in counts.most_common():
        p0, p1 = pair
        features = set(re.split("[.-]", p0)).union(re.split("[.-]", p1))
        for feat in features:
            confFeatures[feat] += count

    return counts, confFeatures

def writePairInstructions(args, confTags, trainData, llm):
    for pair, count in confTags:
        for rep in range(args.replicates):
            confTagPrompt = makeConfusedTagPrompt(pair, trainData)
            if confTagPrompt != "":
                print("Asking LLM to disambiguate", pair)

                os.makedirs(confDir + f"/{rep}/", exist_ok=True)
                with open(confDir + f"/{rep}/{pair[0]}-{pair[1]}.txt", "w") as ofh:
                    ofh.write(confTagPrompt)

                if args.use_lm:
                    prompt_template = ChatPromptTemplate.from_template(
                        template="You are a helpful assistant. {prompt}"
                    )

                    chain = LLMChain(llm=llm, prompt=prompt_template)
                    response = chain.run(prompt=confTagPrompt) + "\n"
                    # response = response.replace("XX", pair[0]).replace("YY", pair[1])

                    with open("confused_tag_output_template.txt", "r") as templateFH:
                        template = templateFH.read()
                    responseDict = { "CONTENT" : response, "LANGUAGE" : args.language }
                    template = template.format(**responseDict)

                    os.makedirs(outDir + f"/{rep}", exist_ok=True)
                    with open(outDir + f"/{rep}/{pair[0]}-{pair[1]}.txt", "w") as ofh:
                        ofh.write(template)

def writeSingleInstructions(args, confTags, trainData, llm):
    for pair, count in confTags:
        f, feature = pair
        assert(f == "FEATURE")
        for rep in range(args.replicates):
            confTagPrompt = makeConfusedTagSingletonPrompt(feature, trainData)
            if confTagPrompt != "":
                print("Asking LLM to disambiguate", pair)

                os.makedirs(confDir + f"/{rep}/", exist_ok=True)
                with open(confDir + f"/{rep}/{pair[0]}-{pair[1]}.txt", "w") as ofh:
                    ofh.write(confTagPrompt)

                if args.use_lm:
                    prompt_template = ChatPromptTemplate.from_template(
                        template="You are a helpful assistant. {prompt}"
                    )

                    chain = LLMChain(llm=llm, prompt=prompt_template)
                    response = chain.run(prompt=confTagPrompt) + "\n"
                    # response = response.replace("XX", pair[0]).replace("YY", pair[1])

                    with open("confused_tag_output_template.txt", "r") as templateFH:
                        template = templateFH.read()
                    responseDict = { "CONTENT" : response }
                    template = template.format(**responseDict)

                    os.makedirs(outDir + f"/{rep}", exist_ok=True)
                    with open(outDir + f"/{rep}/{pair[0]}-{pair[1]}.txt", "w") as ofh:
                        ofh.write(template)

if __name__ == "__main__":
    args = get_args()

    if args.use_lm:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
    else:
        llm = None

    devSents = readData(args.gold)
    proposedSents = readData(args.pred)

    counts, confFeatures = mostConfused(devSents, proposedSents, rootMatchOnly=args.root_match_only)

    print("Confused tags:")
    confTags = []
    for pair, count in counts.most_common():
        print(pair, count)

        if count > args.cutoff:
            if args.exclude_empty_tag:
                if "" in pair:
                    continue

            confTags.append((pair, count))

    print()
    print("Total confusions:", sum(counts.values()))
    print("\n")

    print("Most confusing features:")
    for feat, count in confFeatures.most_common():
        print(feat, count)

    print("Confused tag list:")
    print(confTags)

    if args.language != None:
        confDir = f"prompts/{args.language}/confusions"
        os.makedirs(confDir, exist_ok=True)
        outDir = f"outputs/{args.language}/confusions"
        os.makedirs(outDir, exist_ok=True)

        if not args.singleton_features:
            with open(confDir + "/confusions.json", "w") as ofh:
                json.dump(confTags, ofh)

            trainData = Information(args.language, split="train")

            writePairInstructions(args, confTags, trainData, llm)
        else:
            confTags = []
            for feat, count in confFeatures.most_common():
                if count > args.cutoff:
                    if args.exclude_empty_tag:
                        if feat == "":
                            continue

                    confTags.append([("FEATURE", feat), count])
            
            with open(confDir + "/confusions.json", "w") as ofh:
                json.dump(confTags, ofh)

            trainData = Information(args.language, split="train")

            writeSingleInstructions(args, confTags, trainData, llm)
