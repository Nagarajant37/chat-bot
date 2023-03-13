import nltk
nltk.download('punkt')#divides a text into a list of sentences
nltk.download('wordnet')
import pickle
import json#data storage file 
import numpy as np
from tensorflow import keras
from keras.models import Sequential #model create
from keras.layers import Dense,Dropout,Activation # nerual layers input and Output
import random #number Creation
from nltk.stem import WordNetLemmatizer ##same meaning words
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"s",'()','-','_','%']

data_file=open('intents.json').read()
intents=json.loads(data_file,strict=False)
##Data converstion
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)#split a sentence into words
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]##word Processing Remove unwanted and repated words
words=sorted(list(set(words)))#ordering the words in numbers in an ascending order
classes=sorted(list(set(classes)))#ordering the classes in numbers in an ascending order
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

##print("words",words)
##print("doc", documents)
##print("cls", classes)


##training data
training=[]
output_empty=[0]*len(classes)#label

for doc in documents:
    print("11",doc)
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]#label proceesing 
    
    for word in words:
        print("22",word)
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)# string new list
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1])  

print("trrrraaain",training)

###Model
##model=Sequential()#The sequential API allows you to create models layer-by-layer for most problems
##model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))#Dense layer is the regular deeply connected neural network layer
##model.add(Dropout(0.5))#Dropout is a technique used to prevent a model from overfitting. 
##model.add(Dense(64,activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(len(y_train[0]),activation='softmax'))
##
##adam=keras.optimizers.Adam(0.001)
##model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
##weights=model.fit(np.array(X_train),np.array(y_train),epochs=300,batch_size=10,verbose=1)    
##model.save('mymodel.h5',weights)
