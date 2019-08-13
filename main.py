import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)


try:
    # used to get the data from a saved file so the model
    # will not need training everytime we run the code
    with open("data.pickle","rb")as f:
        words,labels,training,output=pickle.load(f)
except:

    ### Data preprocessing ###

    words = []
    labels = []
    docs_I = []
    docs_J = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            
            # this will return a list with all 
            # the words in them as a token 
            words_list = nltk.word_tokenize(pattern)

            words.extend(words_list)
            docs_I.append(words_list)
            docs_J.append(intent["tag"])

        #makes sure all the tags are added to the lable list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    # removing the dublicuts form the list
    # removing question mark
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    #creats an empty list that has the size of the number of lables
    out_empty = [0 for _ in range(len(labels))]

    #loops through the list of tags with a counter X 
    for x, doc in enumerate(docs_I):
        bag = []

        #stemming the words 
        words_list = [stemmer.stem(w) for w in doc]

        #loops through all the words in the json file
        for w in words:

            #  checking if the words does exist in the 
            #  pattern we are looping in
            if w in words_list:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]

        # places "1" at  the number of the counter(x) which is the location of the tag
        # example: tags list [names,shop,hours]
        # the number where the tage is found [0,0,1] (found in the hours tag) 
        output_row[labels.index(docs_J[x])] = 1

        #now we have list 
        training.append(bag)
        output.append(output_row)

    # takes the list and changes them into an array 
    # which can be fed to our modle
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle","wb")as f:
        pickle.dump((words,labels,training,output),f)


### creating the model ###

#resetting the old data  
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])

# hidding layer with 8
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

# output  layer 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# DNN is the type of  network
model = tflearn.DNN(net)

try:
    model.laod("model.tflearn")
except:
    # fitting the data passing it and the 2000 is the number of times the model will see the data
    model.fit(training,output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def translate(input, words):

    # creats an array with the size of the word
    # and places zeros as the number of words 
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(input)
    s_words = [stemmer.stem(word.lower())for word in s_words]

    # checking which tag matches the number and replacing it with 1
    for x in s_words:
        for i,w in enumerate(words):
            if w == x:
                bag[i] = 1

    # replacing the list  
    return numpy.array(bag)

def chat():
    print("The Bot is ready")
    while True:
        talk = input("Person: ")
        
        # leaves the loop if the word is quit 
        if talk.lower() == "quit":
            break
        
        # returns a list of predictions on how lucky it could be this tag
        results = model.predict([translate(talk, words)])

        # it will give us the index of the largest number
        results_index = numpy.argmax(results)

        # this will get the predicted tag for the input
        tag = labels[results_index]
        
        if results[results_index] < 0.7:
            for reply in data["intents"]:
                if reply["tag"] == tag:
                    responses = reply["responses"]
        else:
            print("Please enter another question.")

        print(random.choice(responses))

chat()