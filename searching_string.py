import sys
import os
from collections import Counter, defaultdict
import re
import random

def sentence_split(prompt):
    condition = re.compile(r"\b[\w'-’]+\b")    
    words = condition.findall(prompt)
    return words

def readData(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.read()
        lines = lines.replace(r"\t", "")
        lines = lines.replace(r"\m", "")
        lines = lines.replace(r"\g", "")
        lines = lines.replace(r"\l", "")
        prompts = lines.split('\n')

    sentences = []
    current = []

    for line in prompts:
        current.append(line)
        if len(current) == 4:
            sentences.append(current)
            current = []

    return sentences

def lcsubstring(s1, s2):
    m = len(s1)
    n = len(s2)
    lcsuff = [[0] * (n + 1) for i in range(m + 1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcsuff[i][j] = 0
            elif s1[i-1] == s2[j-1]:
                lcsuff[i][j] = lcsuff[i-1][j-1] + 1
                result = max(result, lcsuff[i][j])
            else:
                lcsuff[i][j] = 0
    return result

def makeIndex(sentences):
    wordToSentence = defaultdict(list)
    for group in sentences:
        og_sentence, gloss, trans, _ = group
        for word in sentence_split(og_sentence):
            wordToSentence[word].append((og_sentence, gloss, trans))

    return wordToSentence

def lcsMatch(queryWord, approxIndex, wordToSentence, length):
    rankedExamples = set()

    #for every word that shares a 4-length substring with the query
    #compute its lcs
    for start in range(len(queryWord) - length):
        sstr = queryWord[start : start + length]
        for word in approxIndex.get(sstr, []):
            #the word shouldn't be an approx match, because it's already
            #an exact match so we have all these already
            if word != queryWord:
                lcs_length = lcsubstring(queryWord, word)
                rankedExamples.add( (lcs_length, word) )

    #turn set into list, and sort by lcs
    rankedExamples = list(rankedExamples)
    rankedExamples.sort(key=lambda xx: xx[0])
    
    #take only the examples with the longest lcs
    #(this is not necessarily the correct behavior linguistically...)
    sents = []
    if rankedExamples:
        bestRank = rankedExamples[-1][0]
        for (rank, word) in rankedExamples:
            if rank == bestRank:
                sents += wordToSentence[word]

    random.shuffle(sents)
    return sents
        
def findMatches(userInput, wordToSentence, approxIndex):
    userInput = userInput.lower()
    queryWord = userInput
    exactExamples = wordToSentence.get(queryWord, [])
    
    output = ("\n\nExact Matches")
    for example in exactExamples[:3]:
        og_sentence, gloss, translation = example
        output += ("\nSentence:" + og_sentence)
        output += ("\nGloss:" + gloss)
        output +=("\nTranslation:" + translation)

    threshold = 4
    appxExamples = lcsMatch(queryWord, approxIndex, wordToSentence, threshold)
    
    output += ("\n\nApproximate Matches:")
    for example in appxExamples[:3]:
        og_sentence, gloss, translation = example
        output += ("\nSentence:" + og_sentence)
        output += ("\nGloss:" + gloss)
        output +=("\nTranslation:" + translation)
    
    return(str(output))

def makeApproxIndex(wordToSentence, length=4):
    subToWord = defaultdict(list)
    
    for word in wordToSentence:
        for start in range(len(word) - length):
            sstr = word[start : start + length]
            subToWord[sstr].append(word)

    return subToWord

def readLanguage(language, glossDir="2023glossingST-main", split="train"):
    path = f"{glossDir}/data/{language}/"
    files = os.listdir(path)
    langcode = files[0].split("-")[0]
    path = f"{glossDir}/data/{language}/{langcode}-{split}-track1-uncovered"
    sentences = readData(path)
    wordToSentence = makeIndex(sentences)
    subToWord = makeApproxIndex(wordToSentence)
    return (language, sentences, wordToSentence, subToWord)

def createPrompt(word, filePath, langInfo, trans="",
                 promptTemplate="originalfile.txt"):
    (language, sentences, wordToSentence, subToWord) = langInfo
    
    instanceDict = {"WORD" : word, "LANGUAGE" : language, "TRANSLATION": trans, "EXAMPLES" : findMatches(word, wordToSentence, subToWord)}

    with open(promptTemplate, "r") as originalfh:
        text = "".join(originalfh.readlines())
        
    filledPrompt = text.format(**instanceDict)
    text = filledPrompt

    with open(filePath, "w+", encoding="utf-8") as file:
        file.write(text)

    return text
        
if __name__ == "__main__":
    #pull the language from the command line argument array
    language = sys.argv[1]
    word = x
    word = word.lower()

    #the line below hard-codes the file path
    #(we could make it configurable with a bit more effort)
    #you used the dev data, but we should use the training data
    #we need the glosses (ie, the uncovered set)
    #these files don't have a .txt suffix
    path = f"2023glossingST-main/data/{language}/test" #ddo-train-track1-uncovered"
    sentences = readData(path)
    wordToSentence = makeIndex(sentences)
    subToWord = makeApproxIndex(wordToSentence)
    
    instanceDict = {"WORD" : word, "LANGUAGE" : language, "EXAMPLES" : findMatches(word, wordToSentence, subToWord)}

    with open("originalfile.txt", "r") as originalfh:
        text = "".join(originalfh.readlines())
        
    filledPrompt = text.format(**instanceDict)
    text = filledPrompt 

    with open(filePath, "w+", encoding="utf-8") as file:
        file.write(text)
    
