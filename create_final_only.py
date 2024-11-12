from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from collections import Counter, defaultdict
import os
import re
import json

from nltk.stem import WordNetLemmatizer as wnl
import nltk

from json_scanner import JSONScanner
from searching_string import *
from sentence_split import *
from glosser import glossSentence
from glosser import createFinal

def oracleGlosses(dicts, gold):
    gold = gold.split()
    sentenceGlosses = []
    
    for di, gi in zip(dicts, gold):
        best = -1
        result = None

        for possGloss in di["glosses"]:
            elD = possGloss.split("-")
            elG = gi.split("-")
            inter = set(elD).intersection(elG)
            if len(inter) > best:
                best = len(inter)
                result = possGloss

        sentenceGlosses.append(".".join(result.split()))

    return sentenceGlosses

def llmGlossPrompt(filePath, dicts, sent, translation, promptTemplate="final_template.txt"):
    split = sentence_split(sent)
    assert(len(split) == len(dicts))
    lines = dicts[0]["glosses"]
    gloss_per_word = [ [] for li in lines]
    for di in dicts:
        for ind, gi in enumerate(di["glosses"]):
            gloss_per_word[ind].append(gi)

    formatted_gloss_per_word = "\n".join(["\t".join(line) for line in gloss_per_word])
    output_list_template = '[' + ", ".join(['"_"' for word in split]) + ']'

    formatted_sent_gloss = []
    for word, di in zip(split, dicts):
        row = f"{word}:"
        for ind, gi in enumerate(di["glosses"]):
            row += f"\t({ind + 1}) {gi}"
        formatted_sent_gloss.append(row)

    formatted_sent_gloss = "\n".join(formatted_sent_gloss)
        
    substitutions = { "TRANSLATION" : translation,
                      "OUTPUT_LIST" : output_list_template,
                      "GLOSS_PER_WORD" : formatted_gloss_per_word,
                      "SENTENCE" : "\t".join(split),
                      "SENTENCE_AND_GLOSS" : formatted_sent_gloss,
                     }

    with open(promptTemplate, "r" ) as originalfh:
        text = "".join(originalfh.readlines())

    filledPrompt = text.format(**substitutions)

    with open(filePath, "w+", encoding="utf-8") as ofh:
        ofh.write(filledPrompt)

    return filledPrompt

def intOr1(string):
    try:
        return int(string)
    except ValueError:
        return 1

def llmGlosses(llm, language, index, dicts, sent, translation, noLM=False, convert_from_numeric=False):
    promptFile = f"prompts/{language}/{index}/final.txt"
    prompt = llmGlossPrompt(promptFile, dicts, sent, translation)

    prompt_template = ChatPromptTemplate.from_template(
        template="You are a helpful assistant. {prompt}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    if noLM:
        split = sentence_split(sent)
        response = '[' + ", ".join(['"_"' for word in split]) + ']'
    else:
        response = chain.run(prompt=prompt) + "\n"
        
    outFile = f"outputs/{language}/{index}/final.txt"
    with open(outFile, "w", encoding="utf-8") as ofh:
              ofh.write(response)

    dec = JSONScanner(response)
    gloss = dec.scan()
    if convert_from_numeric:
        print("got gloss", gloss, len(gloss), "for", len(dicts), "words")
        gloss = [di["glosses"][intOr1(gi) - 1] for (di, gi) in zip(dicts, gloss)]
    
    gloss = [".".join(gi.split()) for gi in gloss]

    print(f"Disambiguation {index} completed.")
    
    return gloss

def createFinalTranscripts(language, devSents, finalSelection, repairs=None):
    total = ""       

    for index, group in enumerate(devSents.sentences):
        sentenceGlosses = []
        try:
            x = createFinal(f"outputs/{language}/{index}/output.txt")
            ##print("The return value is: ", x)
            ##print(index, group)
            sentence, goldGloss, translation, dummyTwo = group
            if finalSelection == "cheat":
                sentenceGlosses = oracleGlosses(x, goldGloss)
            elif finalSelection == "first":
                for myDict in x:
                    glosses = myDict["glosses"]
                    sentenceGlosses.append(".".join(glosses[0].split()))

            elif finalSelection == "first-conf":
                for myDict in x:
                    glosses = myDict["glosses"]
                    confLex = myDict["confidence_word"]
                    confFeats = myDict["confidence_features"]
                    sentenceGlosses.append(".".join(glosses[0].split()) +
                                           f" ({confLex},{confFeats})")

            elif finalSelection == "llm":
                sentenceGlosses = llmGlosses(llm, language, index, x, sentence, translation, noLM=False, convert_from_numeric=True)
            elif finalSelection == "repair":
                sentenceGlosses = repairedGlosses(language, index, x, repairs)
            else:
                assert(0), f"Unknown final selection method {finalSelection}"                    
        except:
            raise

        y = " "
        output = (f"\\t{sentence}\n")
        output += (f"\\g {y.join(sentenceGlosses)}\n")
        output += ((f"\\l{translation}\n\n"))
        total += output
    
    with open("final.txt", "w", encoding="utf-8") as file:
        file.write(total)

def findUncertainWords(sentGroup, langInfo):
    sent, gloss, trans, _ = sentGroup
    uncertain = []

    for index, word in enumerate(sentence_split(sent)):
        freqTags, freqFeats = frequentTags(word, langInfo.wordToSentence)
        if len(freqTags) == 0:
            approxMatches = lcsMatch(word, subToWord, langInfo.wordToSentence, length=4)
            if len(approxMatches) == 0:
                uncertain.append(index)

    return uncertain

def missingWordPrompt(sentGroup, dicts, promptTemplate="uncertain_word_template.txt"):
    with open(promptTemplate, "r") as originalfh:
        text = "".join(originalfh.readlines())

    sent, goldGloss, trans, _ = sentGroup

    transWords = sentence_split(trans)
    posTags = nltk.pos_tag(transWords)

    def contentful(word, tag):
        #noun, verb, adverb, adjective
        if any([tag.startswith(letter) for letter in "NVRJ"]):
            lemmas = lemmatize(word, wnl())
            if any([lemma in ["be", "have"] for lemma in lemmas]):
                return False

            return True

        return False

    contentWords = [wi for (wi, tag) in posTags if contentful(wi, tag)]

    resultFormat = [f"{{ \"word\" : \"{wi}\", \"status\" : \"_\" }}"
                    for wi in contentWords]
    resultFormat = "[\n" + ",\n".join(resultFormat) + "\n]\n"

    sentenceGlosses = []
    for myDict in dicts:
        glosses = myDict["glosses"]
        sentenceGlosses.append(".".join(glosses[0].split()))

    instanceDict = {
        "SENT" : sent,
        "GLOSS" : "\t".join(sentenceGlosses),
        "TRANS" : trans,
        "RESULT" : resultFormat,
        }

    filledPrompt = text.format(**instanceDict)
    return filledPrompt

def repairPrompt(sentGroup, dicts, uncertain, missing, promptTemplate="repair_word_prompt.txt"):
    lemmatizer = wnl()
    sent, goldGloss, trans, _ = sentGroup
    split = sentence_split(sent)
    if len(split) != len(dicts):
        print("Output length is", len(dicts), "but sent length is", len(split), "for",
              sent)
        for ind, di in enumerate(dicts):
            print(di)
    assert(len(split) == len(dicts))
    options_per_word = [ [] for si in split]
    ourGloss = []
    for ind, di in enumerate(dicts):
        glosses = di["glosses"]
        gloss = (".".join(glosses[0].split()))
        ourGloss.append(gloss)
        options_per_word[ind].append(gloss)

    missingLemmas = []
    for word in missing:
        lemmas = lemmatize(word, lemmatizer)
        best = min(lemmas, key=len)
        missingLemmas.append(best)

    for ind in uncertain:
        for mi in missingLemmas:
            options_per_word[ind].append(mi)

    formatted_gloss_per_word = []
    for pos, (word, di, options) in enumerate(zip(split, dicts, options_per_word)):
        if len(options) > 1:
            candidates = "[ " + ", ".join([f"\"{opt}\"" for opt in options]) + "]"
            json = f"{{ \"word\" : \"{word}\", \"word_pos\" : {pos}, \"candidates\" : {candidates}, \"select\" : _, }}"
            formatted_gloss_per_word.append(json)

    formatted_gloss_per_word = "[\n" + ",\n".join(formatted_gloss_per_word) + "\n]"

    substitutions = {
        "SENTENCE" : sent,
        "GLOSS" : "\t".join(ourGloss),
        "TRANSLATION" : trans,
        "GLOSS_PER_WORD" : formatted_gloss_per_word,
                     }

    with open(promptTemplate, "r" ) as originalfh:
        text = "".join(originalfh.readlines())

    filledPrompt = text.format(**substitutions)

    return filledPrompt

def repairTranscripts(devSents, langInfo, noLM=False):
    repairSent = set()

    for index, sent in enumerate(devSents):
        dicts = createFinal(f"outputs/{language}/{index}/output.txt")
        uncertain = findUncertainWords(sent, langInfo)

        prompt = missingWordPrompt(sent, dicts)
        outFile = f"prompts/{language}/{index}/missing.txt"
        with open(outFile, "w", encoding="utf-8") as ofh:
            ofh.write(prompt)

        if not noLM:
            prompt_template = ChatPromptTemplate.from_template(
                template="You are a helpful assistant. {prompt}"
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = chain.run(llm=llm, prompt=prompt)
            outFile = f"outputs/{language}/{index}/missing.txt"
            with open(outFile, "w", encoding="utf-8") as ofh:
                ofh.write(response)

            dec = JSONScanner(response)
            missingWords = dec.scan()
        else:
            #load the missing file if there is one
            mFile = f"outputs/{language}/{index}/missing.txt"
            try:
                with open(mFile, encoding="utf-8") as ifh:
                    response = ifh.read()

                dec = JSONScanner(response)
                missingWords = dec.scan()
            except FileNotFoundError:
                missingWords = []
            except StopIteration:
                print("Cannot read", mFile)
                raise

        missing = set()
        for wDict in missingWords:
            if wDict["status"] == "check":
                missing.add(wDict["word"].lower())

        #print("missing words:", missing)

        #don't bother trying to repair the gloss unless we have a missing meaning
        #and a gloss element for which the direct evidence was poor
        if len(uncertain) > 0 and len(missing) > 0:
            repairSent.add(index)
            prompt = repairPrompt(sent, dicts, uncertain, missing)
            outFile = f"prompts/{language}/{index}/repair.txt"
            with open(outFile, "w", encoding="utf-8") as ofh:
                ofh.write(prompt)

            if not noLM:
                prompt_template = ChatPromptTemplate.from_template(
                    template="You are a helpful assistant. {prompt}"
                )
                chain = LLMChain(llm=llm, prompt=prompt_template)
                response = chain.run(llm=llm, prompt=prompt)
                outFile = f"outputs/{language}/{index}/repair.txt"
                with open(outFile, "w", encoding="utf-8") as ofh:
                    ofh.write(response)

    return repairSent

def repairedGlosses(language, index, dicts, repairSent):
    #if no repairs are possible for this sentence, don't bother looking for an llm choice
    if index in repairSent:
        repairFile = f"outputs/{language}/{index}/repair.txt"
        with open(repairFile, encoding="utf-8") as ifh:
            response = ifh.read()

        dec = JSONScanner(response)
        repairChoices = dec.scan()
    else:
        repairChoices = []

    print("repairing sentence", index, "with", repairChoices)

    sentenceGlosses = []
    for ind, myDict in enumerate(dicts):
        glosses = myDict["glosses"]
        gloss = ".".join(glosses[0].split())
        for ri in repairChoices:
            if ri["word_pos"] == ind:
                repair = ri["select"]
                candidates = ri["candidates"]
                print("checking candidates for", ri, repair)

                if repair > 0:
                    print("replacing", gloss, "with", candidates[repair])
                    gloss = candidates[repair]

        sentenceGlosses.append(gloss)

    return sentenceGlosses

if __name__ == "__main__":
    #pull the language from the command line argument array
    language = sys.argv[1]

    finalSelection = "first"

    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
    #llm = Ollama(model="llama2")
    
    langInfo = Information(language)
    devInfo = Information(language, split="debug")

    repairs = repairTranscripts(devSents, langInfo, noLM=True)
    createFinalTranscripts(language, devInfo.sentences, finalSelection, repairs)
