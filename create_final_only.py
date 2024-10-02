from collections import Counter, defaultdict
import os
import re
from searching_string import *
from sentence_split import *
from glosser import glossSentence
from glosser import createFinal
import os

if __name__ == "__main__":
    #pull the language from the command line argument array
    language = sys.argv[1]

    langInfo = readLanguage(language)
    (_, devSents, _, _) = readLanguage(language, split="debug")

    total = ""       
    for index, group in enumerate(devSents):
        sentenceGlosses = []
        try:
            x = createFinal(f"outputs/Tsez/{index}/output.txt")
            ##print("The return value is: ", x)
            ##print(index, group)
            sentence, dummy, translation, dummyTwo = group
            for myDict in x:
                glosses = myDict["glosses"]
                sentenceGlosses.append(".".join(glosses[0].split()))
        except:
            raise
            
        y = " "
        output = (f"\\t{sentence}\n")
        output += (f"\\g {y.join(sentenceGlosses)}\n")
        output += ((f"\\l{translation}\n\n"))
        total += output
    
    with open("final.txt", "w", encoding="utf-8") as file:
        file.write(total)
            

