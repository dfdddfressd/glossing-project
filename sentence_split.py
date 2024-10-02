import re

def sentence_split(prompt):
    condition = re.compile(r"\b[\w'-’]+\b")    
    words = condition.findall(prompt)
    return words




