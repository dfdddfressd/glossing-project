import re

def sentence_split(prompt):
    condition = re.compile(r"\b[\w'-’]+\b")    
    words = condition.findall(prompt)
    return words


prompt = "qweffewq' weqfojqwen weqfwff_woeq' weqffoij'wefq,"
words = sentence_split(prompt)
print(words)





