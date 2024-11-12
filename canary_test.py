from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from collections import Counter, defaultdict
import os
import re
import json

from searching_string import *
from sentence_split import *

if __name__ == "__main__":
    nTests = 4
    nPrefix = 6
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    for language in ["Tsez"]:
        langInfo = readLanguage(language)
        (_, devSents, _, _, _, _) = readLanguage(language, split="test")
        tested = 0
        for sent in devSents:
            original, _, _, _ = sent
            words = sentence_split(original)

            if len(words) < nPrefix:
                continue

            tested += 1
            sentPrefix = " ".join(words[:nPrefix])
            suffix = " ".join(words[nPrefix:])

            substitutions = {
                "LANGUAGE" : language,
                "SENTENCE" : sentPrefix
                }
            with open("canary_prompt.txt", "r") as ifh:
                prompt = ifh.read()

            prompt = prompt.format(**substitutions)

            print(prompt)

            prompt_template = ChatPromptTemplate.from_template(
                template="You are a helpful assistant. {prompt}"
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = chain.run(prompt=prompt)
            print(response)
            print("Correct response:", suffix)
            print()
            print()
            
            if tested == nTests:
                break
