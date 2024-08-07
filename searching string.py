from collections import Counter, defaultdict
import re

def sentence_split(prompt):
    condition = re.compile(r"\b[\w'-’]+\b")    
    words = condition.findall(prompt)
    return words

with open("ddo-dev-track2-covered.txt", "r", encoding="utf-8") as file:
    lines = file.read()
    lines = lines.replace(r"\t", "")
    lines = lines.replace(r"\m", "")
    lines = lines.replace(r"\g", "")
    lines = lines.replace(r"\l", "")
    prompts = lines.split('\n')
file.close()


sentences = []
current = []

for line in prompts:
   current.append(line)
   if len(current) == 4:
     sentences.append(current)
     current = []
     

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

def findMatches(userInput):
    global sentences
    userInput = userInput.lower()
    wordToSentence = defaultdict(list)
    for group in sentences:
        og_sentence = group[0]
        translation = group[3]
        for word in sentence_split(og_sentence):
            wordToSentence[word].append((og_sentence, translation))
        
    queryWord = userInput
    exactExamples = wordToSentence.get(queryWord, [])
    
    output = ("\n\nExact Matches")
    for example in exactExamples[:3]:
        og_sentence, translation = example
        output += ("\nSentence:" + og_sentence)
        output +=("\nTranslation:" + translation)
    
    threshold = 4
    appxExamples = []

    for word, sentences in wordToSentence.items():
        lcs_length = lcsubstring(queryWord, word)
        if lcs_length >= threshold:
            appxExamples.extend(sentences)

    unique_examples = list(dict.fromkeys(appxExamples))
    
    output += ("\n\nApproximate Matches:")
    for example in appxExamples[:3]:
        og_sentence, translation = example
        output += ("\nSentence:" + og_sentence)
        output += ("\nTranslation:" + translation)
    
    
    return(str(output))



language = "Tsez"
word = "esirno"
word = word.lower()

instanceDict = {"WORD" : word, "LANGUAGE" : language}

with open("originalfile.txt", "r") as originalfh:
    text = originalfh.readlines()
filledPrompt = [text.format(**instanceDict) for text in text]    
for text in filledPrompt:
    text = text + "\nHere are example sentences with the word " + word + ": " + findMatches(word)

with open("prompts.txt", "w+", encoding="utf-8") as file:
    file.write(text)
    
file.close()
    