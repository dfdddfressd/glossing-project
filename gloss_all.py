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

if __name__ == "__main__":
    #os.environ["OPENAI_API_KEY"] = ""
    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.25)
    #llm = Ollama(model="llama2")
    #llm = None
    prompt_template = ChatPromptTemplate.from_template(
        template="You are a helpful assistant. {prompt}"
    )

    #pull the language from the command line argument array
    language = sys.argv[1]

    langInfo = readLanguage(language)
    (_, devSents, _, _, _, _) = readLanguage(language, split="debug")
    
    for count, sentence in enumerate(devSents):
        promptPath = f"prompts/{language}/{count}"
        outputPath = f"outputs/{language}/{count}"
        os.makedirs(promptPath, exist_ok=True)
        os.makedirs(outputPath, exist_ok=True)
        
        glossSentence(sentence, langInfo,
                      llm=llm,
                      prompt_template=prompt_template,
                      promptPath=promptPath,
                      outputPath=outputPath,
                      templatePath="originalfile.txt",
                      noLM=False)

    createFinalTranscripts(language, devSents, finalSelection="first")
