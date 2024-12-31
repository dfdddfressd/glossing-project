from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from collections import Counter, defaultdict
import os
import re
from searching_string import *
from sentence_split import *
from glosser import glossSentence
from glosser import createFinal
from create_final_only import createFinalTranscripts
import os

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language")
    parser.add_argument("--split", default="debug")
    parser.add_argument("--use_lm", action="store_true")
    parser.add_argument("--instructions", default="0", help="Number of the instruction dir, or best, or range (0:10).")
    args = parser.parse_args()
    return args

def glossAllSentences(info, llm, prompt_template, use_lm, instructions, subdir=False):
    for count, sentence in enumerate(info.sentences):
        if subdir:
            #make an instruction-set-specific subdirectory
            promptPath = f"prompts/{info.language}/i{instructions}/{count}"
            outputPath = f"outputs/{info.language}/i{instructions}/{count}"
        else:
            promptPath = f"prompts/{info.language}/{count}"
            outputPath = f"outputs/{info.language}/{count}"

        os.makedirs(promptPath, exist_ok=True)
        os.makedirs(outputPath, exist_ok=True)

        print(f"...{count}")

        glossSentence(sentence, langInfo,
                      llm=llm,
                      prompt_template=prompt_template,
                      promptPath=promptPath,
                      outputPath=outputPath,
                      templatePath="originalfile.txt",
                      instructions=instructions,
                      noLM=not args.use_lm)

if __name__ == "__main__":
    #pull the language from the command line argument array
    args = get_args()

    if args.use_lm:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        prompt_template = ChatPromptTemplate.from_template(
            template="You are a helpful assistant. {prompt}"
        )
    else:
        llm = None
        prompt_template = None

    language = args.language

    langInfo = Information(language)
    devInfo = Information(language, split=args.split)

    if ":" in args.instructions:
        instrRange = range(int(args.instructions[:args.instructions.index(":")]),
                           int(args.instructions[args.instructions.index(":") + 1:]))
        for instr in instrRange:
            glossAllSentences(devInfo, llm, prompt_template, args.use_lm, instructions=instr, subdir=True)
            createFinalTranscripts(language, devInfo, finalSelection="first", output_name=f"final_i{instr}.txt", instructions=instr)
    else:
        glossAllSentences(devInfo, llm, prompt_template, args.use_lm, instructions=args.instructions)
        createFinalTranscripts(language, devInfo, finalSelection="first")
