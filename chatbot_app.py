from flask import Flask, render_template, request
import json
import pickle
import time
import random
import nltk
import numpy as np
from gtts import gTTS
import os
import playsound
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
from keras.models import load_model
model = load_model('mymodel.h5')
import json
import random
intents1 = open('intents.json').read()
intents=json.loads(intents1,strict=False)
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

app.static_folder = 'static'

@app.route('/')
def login():
   return render_template("home.html")
@app.route('/home',methods=['POST'])
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return "created by rathinam student "

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]

    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json,text):
    
    if len(return_list)==0:
        tag='noanswer'
    else:    
        tag=return_list[0]['intent']


    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result
def response(text):
    return_list=predict_class(text,model)
    response=get_response(return_list,intents,text)
    return response
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    resp=response(userText)
##    language='en'
##    myob= gTTS(text=str(resp),lang=language,slow=False)
##    myob.save("wel.mp3")
##    os.system("mpg321 wel.mp3")
##    playsound.playsound("wel.mp3",True)
##    os.remove("wel.mp3")
##    print(resp)
    return resp
    
if __name__ == "__main__":
    app.run()

