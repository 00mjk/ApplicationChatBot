import tensorflow
import  random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patteren"]:
        
        # this will return a list with all 
        # the words in them as a token 
        words_list = nltk.word_tokenize(pattern)

        words.extend(words_list)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
