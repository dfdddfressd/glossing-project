from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from collections import Counter, defaultdict
import os
import sys
from searching_string import *
from sentence_split import *
import os
import json
from json_scanner import JSONScanner

def glossSentence(sentence, langInfo, llm, prompt_template,
                  promptPath, outputPath, templatePath,
                  instructions=None,
                  verbose=False, noLM=False):
    response = ""
    count = 0
    source_sentence, _, translation, _ = sentence
    translation = translation.strip()
    words = sentence_split(source_sentence)

    if verbose:
        print("Glossing:", words)
    
    for index, x in enumerate(words):
        count+=1
        promptFile = f"{promptPath}/prompt{count}.txt"
        prompt = createPrompt(x, promptFile, langInfo, (source_sentence, index, translation), instructions=instructions)

        if verbose:
            print("Running prompt:", prompt)

        if noLM:
            response += """{ "word" : "None", "glosses" : [ "None", "None", "None" ] }"""
        else:
            chain = LLMChain(llm=llm, prompt=prompt_template)

            user_prompt = prompt

            response += chain.run(prompt=user_prompt) + "\n"
            if verbose:
                print("Got response:", response)

    outFile = f"{outputPath}/output.txt"
    with open(outFile, "w", encoding="utf-8") as file:
        file.write(response)

    print("Gloss completed.")

def fixQuotes(jsonStr):
    """Try to render a string as valid JSON by ensuring all words are quoted."""
    quoteRE = re.compile(r'"word":\s*([^"][^",]*),')
    res = re.sub(quoteRE, r'"word": "\g<1>",', jsonStr)
    return res
    
def createFinal(filename):
    with open(filename, encoding="utf8") as jsonfh:
        everything = jsonfh.read()
    
    output = []
    dec = JSONScanner(everything)

    everything = fixQuotes(everything)
    
    while True:
        try:
            structure = dec.scan()
            output.append(structure)
        except StopIteration as e:
            #print(filename, e)
            break
    return(output)
