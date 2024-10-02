from collections import defaultdict
import re

# Function to compute the longest common substring length
def longest_common_substring(s1, s2):
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

# Function to split a sentence into words
def sentence_split(prompt):
    condition = re.compile(r"\b[\w'-’]+\b")    
    words = condition.findall(prompt)
    return words

# Read the file and replace the escape sequences
with open("ddo-dev-track2-covered.txt", "r", encoding="utf-8") as file:
    lines = file.read()
    lines = lines.replace(r"\t", "")
    lines = lines.replace(r"\m", "")
    lines = lines.replace(r"\g ", "")
    lines = lines.replace(r"\l", "")
    prompts = lines.split('\n')

sentences = []
current = []
for line in prompts:
    current.append(line)
    if len(current) == 4:
        sentences.append(current)
        current = []

# Create a dictionary to map words to their sentences and translations
wordToSentence = defaultdict(list)
for group in sentences:
    raw_sentence = group[0]  # Assuming the raw language sentence is the first line in each group
    translation = group[3]   # Assuming the translation is the fourth line in each group
    for word in sentence_split(raw_sentence):
        wordToSentence[word].append((raw_sentence, translation))

# Query word and retrieve sentences with approximate matches
queryWord = "esirno"
threshold = 4  # Define a threshold for similarity, adjust as needed
examples = []

for word, sentences in wordToSentence.items():
    lcs_length = longest_common_substring(queryWord, word)
    if lcs_length >= threshold:
        examples.extend(sentences)

# Remove duplicates and limit to the first 3 examples
unique_examples = list(dict.fromkeys(examples))

print("The sentences that contain approximate matches to", queryWord, "are:")
for example in unique_examples[:3]:
    raw_sentence, translation = example
    print("Raw sentence:", raw_sentence)
    print("Translation:", translation)
    print()
    
    