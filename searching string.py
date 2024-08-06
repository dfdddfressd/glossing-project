from collections import Counter, defaultdict
import re

def sentence_split(prompt):
    condition = re.compile(r"\b[\w'-’]+\b")    
    words = condition.findall(prompt)
    return words

prompts = ["Hemełƛ’o sida nukarer qalamä teł xanez užä ruqˤno zawruni barubi rukayn.", "ʕAt’idä nesiq kinaw raqru łinałäy esin.", "Ražbadinez idu barun, xexbin yołƛin, žawab teƛno ečruni žek’a.", "Ža xabar äsirun neła, akäłru baħrä biłe, maħor mec boƛik’no.", "Ža uži esarus aħozir ixiw sayɣatno bodin, egirno xizor."]


"""prompt = "qweffewq' weqfojqwen weqfwff_woeq' weqffoij'wefq,"
words = sentence_split(prompt)
print(words)"""



"""def word_frequency(list):
    word_counts = Counter()
    word_to_sentences = defaultdict(set)
    
    for sentence in list:
        words = re.findall(r"\b\w+(?:'\w+)?\b|\b'\w+\b|\b\w+'\b", sentence.lower())
        word_counts.update(words)
        for word in words:
            word_to_sentences[word].add(sentence)
    
    return word_counts, word_to_sentences

frequency, word_to_sentences = word_frequency(prompts)

for word, count in frequency.items():
    sentences = word_to_sentences[word]
    sentences_str = ', '.join(map(str, sentences)) if sentences else 'no sentence'
    print(f"'{word}': {count} (Found in sentence(s): {sentences_str})")"""
    



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

"""user_input = input("Enter a sentence or phrase to match: ")"""

def findMatches(userInput):
    best_matches = find_top_k_stems(prompts, userInput, k=3)

    if best_matches[0][0] == userInput and best_matches[0][1] == 1:
        for sentence in best_matches[0][2]:
            output = (f"- {sentence}")
    else:
        for stem, count, sentences  in best_matches:
            output = (f"'{stem}' appears in {count} sentence(s)")
            for sentence in sentences:
                output +=(f"- {sentence}")
    return(output)



prompt = "ʕAt’idä nesiq kinaw raqru łinałäy esin."
words = sentence_split(prompt)
length = len(words)
language = "Tsez"

"""for i in words:
    word = i"""

word = "werwfe"
instanceDict = {"WORD" : word, "LANGUAGE" : language}

with open("originalfile.txt", "r") as originalfh:
    text = originalfh.readlines()
filledPrompt = [text.format(**instanceDict) for text in text]    

for text in filledPrompt:
    text = text + "\nHere are example sentences: " + findMatches(word)
    
    
with open("prompts.txt", "w+", encoding="utf-8") as file:
    file.write(text)
    
file.close()
    