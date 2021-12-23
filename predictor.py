import tensorflow
import numpy as np
import streamlit as st
import os
from tensorflow import keras
import re
import string
# import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
# from nltk.tokenize import word_tokenize
import pickle

GDFILE = "1UwWhrEjVNQtPx8W11yqQPNMrY0J1jfkL" #CNN All MODEL
# GDFILE = "18flRD4XATu-pXYr1a9ElLqDCLjdAUk90" #BILSTM All MODEL
PKFILE = "15WvsUDR7YDkgTmZjqc2mrQAN8SICebg_" #BILSTM All TOKENIZER


def load_model():
    filepath = "model/model.h5"
    picklepath = "model/tokenizer.pickle"
    if not os.path.exists('model'):
        	os.mkdir('model')
	
    if not os.path.exists(filepath):
    	from google_drive_downloader import GoogleDriveDownloader as gdd
    	gdd.download_file_from_google_drive(file_id=GDFILE, dest_path=filepath)

    if not os.path.exists(picklepath):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        gdd.download_file_from_google_drive(file_id=PKFILE, dest_path=picklepath)
        
    model = keras.models.load_model(filepath)
    
    return model

def predict(model, input):
    print(input)
    # class_names = ['INFJ', 'ENTP', 'INTP', 'ENTJ', 'INFP', 'ENFJ', 'INTJ', 'ENFP', 'ISTP', 'ISFJ', 'ESFP', 'ESTP','ISFP','ESFJ','ISTJ', 'ESTJ']
    class_names = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'] #for BiLSTM_All

    # for wiki-news
    # max_sentence_length = 5500 
    # tokenizer = Tokenizer()
    # seq = tokenizer.texts_to_sequences(input)
    # padded = sequence.pad_sequences(seq, maxlen=max_sentence_length)
    # pred = model.predict(padded)
    # prediction = format(class_names[np.argmax(pred)])

    # for glove
    max_sentence_length = 764 
    with open('model/tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        seq = loaded_tokenizer.texts_to_sequences([input])
    
    padded = sequence.pad_sequences(seq, maxlen=max_sentence_length)
    pred = model.predict(padded)
    prediction = format(class_names[np.argmax(pred[0])])

    return prediction