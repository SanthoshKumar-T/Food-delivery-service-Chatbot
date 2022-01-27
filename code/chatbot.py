import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open(r'D:\G_DRIVE\Deeplearning\intents.json') as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model(r'C:\Users\DELL\ml_dl_practise\chatbot_model')

    # load tokenizer object
    with open(r'C:\Users\DELL\ml_dl_practise\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(r'C:\Users\DELL\ml_dl_practise\label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
        #Making predictions
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        #Returning response
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))


print(Fore.YELLOW + "Start messaging with the AI bot (type quit to stop)!" + Style.RESET_ALL)
chat()