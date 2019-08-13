import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json


with open("intents.json") as file:
    data = json.load(file)


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

# fitting the data passing it and the 2000 is the number of times the model will see the data
model.fit(training,output,n_epoch=2000, batch_size=8, show_metric=True)
model.save("model.tflearn")