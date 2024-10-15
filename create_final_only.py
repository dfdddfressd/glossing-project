from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from collections import Counter, defaultdict
import os
import re
import json
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

def createFinalTranscripts(language, devSents, finalSelection):
    total = ""       

    for index, group in enumerate(devSents):
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

if __name__ == "__main__":
    #pull the language from the command line argument array
    language = sys.argv[1]

    finalSelection = "first"

    if finalSelection == "llm":
        #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
        #llm = Ollama(model="llama2")
    
    langInfo = readLanguage(language)
    (_, devSents, _, _, _, _) = readLanguage(language, split="debug")

    createFinalTranscripts(language, devSents, finalSelection)
