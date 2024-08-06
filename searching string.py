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


def find_top_k_stems(arr, user_input, k=3):
    n = len(arr)
    
    if user_input in arr:
        return [(user_input, 1)]
        
    
    substrings = {}
    substrings_in_sentences = defaultdict(set)

    for s in arr:
        m = len(user_input)
        l = len(s)
        
        for i in range(m):
            for j in range(i + 1, m + 1):
                stem = user_input[i:j]
                if stem in substrings:
                    continue
                count = sum(stem in string for string in arr)
                if count > 0:
                    substrings[stem] = count
                    for sentence in arr:
                        if stem in sentence:
                            substrings_in_sentences[stem].add(sentence)

    sorted_substrings = sorted(substrings.keys(), key=len, reverse=True)
    top_k_stems = sorted_substrings[:k]
    result = [(stem, substrings[stem], list(substrings_in_sentences[stem])) for stem in top_k_stems]
    return result

def findMatches(userInput):
    """best_matches = find_top_k_stems(prompts, userInput, k=3)"""

    wordToSentence = defaultdict(list)
    for group in sentences:
        og_sentence = group[0]
        translation = group[3]
        for word in sentence_split(og_sentence):
            wordToSentence[word].append((og_sentence, translation))
        
    queryWord = userInput
    examples = wordToSentence.get(queryWord, [])

    output = ("the sentences that contain " + queryWord +  " are:")
    for example in examples[:3]:
        og_sentence, translation = example
        output += ("\nSentence:" + og_sentence)
        output +=("\nTranslation:" + translation)
    
    
    """else:
        for stem, count, sentences  in best_matches:
            output = (f"'{stem}' appears in {count} sentence(s)")
            for sentence in sentences:
                output +=(f"- {sentence}")"""
    return(str(output))



language = "Tsez"
word = "Hemełƛ’o"

instanceDict = {"WORD" : word, "LANGUAGE" : language}

with open("originalfile.txt", "r") as originalfh:
    text = originalfh.readlines()
filledPrompt = [text.format(**instanceDict) for text in text]    
for text in filledPrompt:
    text = text + "\nHere are example sentences with the word " + word + ": " + findMatches(word)


with open("prompts.txt", "w+", encoding="utf-8") as file:
    file.write(text)
    
file.close()
    