import sys
import os
from collections import Counter, defaultdict
import re
import random
import json

from nltk.stem import WordNetLemmatizer as wnl

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

def lcsubseq(s1, s2):
    #https://stackoverflow.com/questions/48651891/longest-common-subsequence-in-python
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)

    cs = matrix[-1][-1]

    return len(cs)

def makeIndex(sentences):
    wordToSentence = defaultdict(list)
    for group in sentences:
        og_sentence, gloss, trans, _ = group
        for word in sentence_split(og_sentence):
            wordToSentence[word].append((og_sentence, gloss, trans))

    return wordToSentence

def findGlossElement(word, lemmas, gloss, object_words):
    for ind, gi in enumerate(gloss):
        elements = re.split("[.-]", gi)
        if any([xx in [word,] + lemmas for xx in elements]):
            return (object_words[ind], gi)

    return None, None

def lemmatize(word, lemmatizer):
    verb_lemmas = lemmatizer._morphy(word, "v")
    lemma_x = lemmatizer.lemmatize(word)
    if lemma_x not in verb_lemmas:
        return verb_lemmas + [lemma_x]
    else:
        return verb_lemmas

def makeMetalanguageIndex(sentences):
    wordToSents = defaultdict(list)
    wordToGlosses = defaultdict(Counter)

    lemmatizer = wnl()

    for group in sentences:
        og_sentence, gloss, trans, _ = group
        gloss = gloss.split()
        object_words = [xx.lower().strip(".,\"'") for xx in og_sentence.split()]
        if len(gloss) != len(object_words):
            print(gloss, object_words)
        assert(len(gloss) == len(object_words))
        
        for index, word in enumerate(sentence_split(trans)):
            word = word.lower()
            lemmas = lemmatize(word, lemmatizer)
            trans_word, gloss_item = findGlossElement(word, lemmas, gloss, object_words)
            wordToSents[word].append((og_sentence, gloss, trans))

            if trans_word != None:
                wordToGlosses[word][(trans_word, gloss_item)] += 1

    return wordToSents, wordToGlosses

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

def getTag(gloss):
    elements = re.split("([.-])", gloss)

    def validFeat(glSym):
        return (not glSym.islower() and
                not glSym.istitle() and
                any([ch.isalnum() for ch in glSym]))
    
    keep = [validFeat(ei) for ei in elements]
    xKeep = keep[:]
    for ind in range(len(keep)):
        if keep[ind] and ind + 1 < len(keep):
            xKeep[ind + 1] = True

    xElts = [ei for (ei, ki) in zip(elements, xKeep) if ki]
    return "".join(xElts)

def frequentTags(word, wordToSentence):
    exactExamples = wordToSentence.get(word, [])
    tagCounts = Counter()
    featureCounts = Counter()

    for example in exactExamples:
        og_sentence, gloss, translation = example
        ind = sentence_split(og_sentence).index(word)
        gloss = gloss.split()[ind]
        tagCounts[gloss] += 1
        for feature in getTag(gloss).split("-"):
            featureCounts[feature] += 1

    return tagCounts, featureCounts

def fitsPattern(object_word, glosses):
    if len(glosses) < 2:
        return False #no evidence about how to translate
    
    object_word = object_word.lower()
    glItems = [xx for xx, ct in glosses.most_common(8)]
    
    lcsObToGl = []
    lcsGlToGl = []
    
    for (word, gloss) in glItems:
        lcs = lcsubseq(object_word, word)
        lcsObToGl.append(lcs)

    for ii, (wordI, glossI) in enumerate(glItems):
        for jj, (wordJ, glossJ) in enumerate(glItems):
            if jj < ii:
                lcs = lcsubseq(wordI, wordJ)
                lcsGlToGl.append(lcs)

    meanGlToGl = sum(lcsGlToGl) / len(lcsGlToGl)
    meanObToGl = sum(lcsObToGl) / len(lcsObToGl)

    # print(f"Does {object_word} fit into {glItems}?")
    # print(f"{lcsObToGl} vs {lcsGlToGl}? {meanObToGl} {meanGlToGl}")

    return meanObToGl + 1 > meanGlToGl

def filteredMetalanguageWords(object_word, trans, metaInfo):
    wordToSent, wordToGloss = metaInfo
    transWords = [xx.lower() for xx in sentence_split(trans)]

    result = ""

    for ti in set(transWords):
        glosses = wordToGloss[ti]
        if glosses:
            if fitsPattern(object_word, glosses):
                best = glosses.most_common(5)
                if best:
                    fBest = [f"{word} ({gloss})" for ((word, gloss), count) in best]
                    result += f"Words for \"{ti}\" include: {', '.join(fBest)}\n"

    return result
                    
def findMetalanguageWords(trans, metaInfo):
    wordToSent, wordToGloss = metaInfo
    transWords = [xx.lower() for xx in sentence_split(trans)]

    result = ""

    for ti in set(transWords):
        glosses = wordToGloss[ti]
        best = glosses.most_common(5)
        if best:
            fBest = [f"{word} ({gloss})" for ((word, gloss), count) in best]
            result += f"Some translations of \"{ti}\" include: {', '.join(fBest)}\n"

    return result

def makeApproxIndex(wordToSentence, length=4):
    subToWord = defaultdict(list)
    
    for word in wordToSentence:
        for start in range(len(word) - length):
            sstr = word[start : start + length]
            subToWord[sstr].append(word)

    return subToWord

def makeWordToTagIndex(sentences):
    index = defaultdict(set)
    for sentence in sentences:
        og_sentence, gloss, translation, _ = sentence
        words = sentence_split(og_sentence)
        glosses = gloss.split()
        for (wi, gi) in zip(words, glosses):
            index[wi].add(getTag(gi))

    return index

def getExampleWithTag(word, wordToSentence, tag):
    exactExamples = wordToSentence.get(word, [])
    random.shuffle(exactExamples)
    for example in exactExamples:
        og_sentence, gloss, translation = example
        ind = sentence_split(og_sentence).index(word)
        gloss = gloss.split()[ind]
        #print("gloss item for", word, "is", gloss)
        if getTag(gloss) == tag:
            return example        

def makeConfusedTagBlock(confusedTags, freqTags, langInfo, promptTemplate="confused_tag_template.txt"):
    t2 = None

    #print("checking for tag confusions at", freqTags.most_common())
    
    #find a tag which might be relevant here, and is confused with a partner t2
    for t1 in freqTags:
        #print(t1, getTag(t1), confusedTags, getTag(t1) in confusedTags)
        if getTag(t1) in confusedTags:
            t1 = getTag(t1)
            t2 = confusedTags[getTag(t1)]
            break

    #if we can't find one, nothing to do
    if t2 is None:
        return ""

    #print("found another tag", t1, t2)

    #find single words which occur with both t1 and t2
    wordExamples = []
    for word, tags in langInfo.wordToTags.items():
        if t1 in tags and t2 in tags:
            wordExamples.append(word)

    #print("words which have both", t1, "and", t2, wordExamples)
            
    #for now, only get examples if we have real contrastive ones... but consider later
    if not wordExamples:
        return ""

    random.shuffle(wordExamples)
    
    #get examples of each type
    exes = []
    for word in wordExamples:
        e1 = getExampleWithTag(word, langInfo.wordToSentence, t1)
        e2 = getExampleWithTag(word, langInfo.wordToSentence, t2)
        exes.append((e1, e2, word))
        
    #subselect the examples so we only have 3 max
    exes = exes[:3]

    formattedExes = ""
    for (e1, e2, word) in exes:
        formattedExes += f"Examples of {word} with both tags:\n"
        for ei in (e1, e2):
            og_sentence, gloss, translation = ei            
            formattedExes += ("\nSentence:" + og_sentence)
            formattedExes += ("\nGloss:" + gloss)
            formattedExes +=("\nTranslation:" + translation)
        formattedExes += "\n\n"

    with open(promptTemplate, "r") as originalfh:
        text = "".join(originalfh.readlines())

    if t1 == "":
        t1 = '""'
    if t2 == "":
        t2 = '""'
        
    instanceDict = {
        "LANGUAGE" : langInfo.language,
        "TAG1" : t1,
        "TAG2" : t2,
        "EXAMPLES" : formattedExes
        }
        
    filledPrompt = text.format(**instanceDict)
    return filledPrompt    

def createPrompt(word, filePath, langInfo, trans="",
                 promptTemplate="originalfile.txt"):
    
    freqTags, freqFeats = frequentTags(word, langInfo.wordToSentence)
    if len(freqTags) == 0:
        formattedTags = "Unknown"
        formattedFeats = "Unknown"
    else:
        formattedTags = ", ".join([tag for (tag, count) in freqTags.most_common(5)])
        formattedFeats = ", ".join([feat for (feat, count) in freqFeats.most_common(5)])

    confusedTags = langInfo.confusedTags
    confusedTagBlock = makeConfusedTagBlock(confusedTags, freqTags, langInfo)
    
    instanceDict = {"WORD" : word, "LANGUAGE" : langInfo.language, "TRANSLATION": trans,
                    "EXAMPLES" : findMatches(word, langInfo.wordToSentence, langInfo.subToWord),
                    "TRANSLATION_EXAMPLES" : filteredMetalanguageWords(word, trans, langInfo.metaIndices),
                    "FREQUENT_TAGS" : formattedTags,
                    "FREQUENT_FEATS" : formattedFeats,
                    "CONFUSED_TAG_BLOCK" : confusedTagBlock,
                    }

    with open(promptTemplate, "r") as originalfh:
        text = "".join(originalfh.readlines())
        
    filledPrompt = text.format(**instanceDict)
    text = filledPrompt

    with open(filePath, "w+", encoding="utf-8") as file:
        file.write(text)

    return text

class Information:
    def __init__(self, language, glossDir="2023glossingST-main", outputDir="outputs", split="train"):
        path = f"{glossDir}/data/{language}/"
        files = os.listdir(path)
        langcode = files[0].split("-")[0]
        path = f"{glossDir}/data/{language}/{langcode}-{split}-track1-uncovered"
        self.language = language
        self.sentences = readData(path)
        self.wordToSentence = makeIndex(self.sentences)
        self.subToWord = makeApproxIndex(self.wordToSentence)
        self.wordToTags = makeWordToTagIndex(self.sentences)
        self.metaIndices = makeMetalanguageIndex(self.sentences)

        #attempt to read confused tags
        path = f"{outputDir}/{language}/confusions.json"
        try:
            with open(path) as ifh:
                self.confusedTags = json.load(ifh)
            print("Loaded the following tag confusions:")
            print(self.confusedTags)
        except FileNotFoundError:
            self.confusedTags = {}
            
        self.split = split

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
    
    instanceDict = {"WORD" : word, 
                    "LANGUAGE" : language, 
                    "EXAMPLES" : findMatches(word, langInfo.wordToSentence, langInfo.subToWord)
                    }

    with open("originalfile.txt", "r") as originalfh:
        text = "".join(originalfh.readlines())
        
    filledPrompt = text.format(**instanceDict)
    text = filledPrompt 

    with open(filePath, "w+", encoding="utf-8") as file:
        file.write(text)    
