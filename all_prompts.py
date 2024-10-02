import sys
from collections import Counter, defaultdict
import re
#it is easier to import functions from a module if it has
#no spaces in its filename
from searching_string import *

def completePrompt(group, wordToSentence, approxIndex, text):
    sentence, gloss, trans, _ = group
    prompts = []
    print(sentence)
    for word in sentence_split(sentence):
        instanceDict = {"WORD" : word, "LANGUAGE" : language}

        # I don't love the code below--- I think it would be neater
        # to use the same ".format" code to add the example sentences
        # to the template, since then we could control how the examples
        # appear and allow them to appear in the middle of the prompt
        # however I won't alter it for now
        
        filledPrompt = text.format(**instanceDict)
        filledPrompt += "\nHere are example sentences with the word " + word + ": " + findMatches(word, wordToSentence, approxIndex)
        prompts.append(filledPrompt)
        prompts.append("\n\n")
        
    return prompts


if __name__ == "__main__":
    #pull the language from the command line argument array
    language = sys.argv[1]

    path = f"2023glossingST/data/{language}/ddo-train-track1-uncovered"
    sentences = readData(path)
    wordToSentence = makeIndex(sentences)
    subToWord = makeApproxIndex(wordToSentence)
    
    #these are the examples we want to run the LLM on
    devPath = f"2023glossingST/data/{language}/ddo-dev-track1-covered"
    devSentences = readData(devPath)

    with open("originalfile.txt", "r") as originalfh:
        text = "".join(originalfh.readlines())
        
    for index, group in enumerate(devSentences):
        #uncomment this line to make the program slower
        #wordToSentence = makeIndex(sentences)
        prompt = completePrompt(group, wordToSentence, subToWord, text)
        output = f"prompts/{language}/{index}"
        with open(output, "w", encoding="utf-8") as ofh:
            for line in prompt:
                ofh.write(line)
