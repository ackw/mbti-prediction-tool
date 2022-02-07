
# imports ----------------------------------------------------------------------

import streamlit as st
import pandas as pd

# predict
import tensorflow
import numpy as np
import streamlit as st
import os
from tensorflow import keras
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle

# visualization
import plotly.express as px

# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
s = SentimentIntensityAnalyzer()

# predictor initialisation ----------------------------------------------------------------------

GDFILE = "1UwWhrEjVNQtPx8W11yqQPNMrY0J1jfkL" #CNN All MODEL
TKFILE = "15WvsUDR7YDkgTmZjqc2mrQAN8SICebg_" #TOKENIZER

filepath = "model/model.h5"
picklepath = "model/tokenizer.pickle"

def load_model():
    # if not os.path.exists('model'):
    #     os.mkdir('model')
    
    # download model
    if not os.path.exists(filepath):
    	from google_drive_downloader import GoogleDriveDownloader as gdd
    	gdd.download_file_from_google_drive(file_id=GDFILE, dest_path=filepath)

    # download tokenizer
    # if not os.path.exists(picklepath):
    #     from google_drive_downloader import GoogleDriveDownloader as gdd
    #     gdd.download_file_from_google_drive(file_id=TKFILE, dest_path=picklepath)
        
    model = keras.models.load_model(filepath)
    
    return model

def predict(model, input):
    print("User input : ", input)
    class_names = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'] #for BiLSTM_All

    # for glove
    max_sentence_length = 764 
    with open(picklepath, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        seq = loaded_tokenizer.texts_to_sequences([input])
    
    padded = sequence.pad_sequences(seq, maxlen=max_sentence_length)
    pred = model.predict(padded)
    prediction = format(class_names[np.argmax(pred[0])])

    return prediction

# initialisation ----------------------------------------------------------------------

st.set_page_config(
    page_title="Myer-Brigging this up...",
    page_icon="🔮"
)

# load tensorflow model
with st.spinner("Loading the genie..."):
    model = load_model()

# datasets
df = pd.read_csv("datasets/mbti_1_cleaned_all.csv")
df2 = pd.read_csv("datasets/multitask_mbti.csv")

# sidebar ------------------------------------------------------------------------------
sections = ['MBTI Prediction Tool', 'Data Visualization', 'Sentiment Analysis']
selected_sect = st.sidebar.selectbox("Choose a feature:", sections)
print(selected_sect)

st.sidebar.markdown("***")
st.sidebar.caption("What do they mean?")

with st.sidebar.expander("16 MBTI Types"):
    st.write('**Analysts**: INTJ, INTP, ENTJ, ENTP')
    st.write('**Diplomats**: INFJ, INFP, ENFJ, ENFP')
    st.write('**Sentinels**: ISTJ, ISFJ, ESTJ, ESFJ')
    st.write('**Explorers**: ISTP, ISFP, ESTP, ESFP')

with st.sidebar.expander("4 Dimensions"):
    st.write('**IE**: Introvert, Extrovert')
    st.write('**NS**: Intuition, Sensing')
    st.write('**TF**: Thinking, Feeling')
    st.write('**JP**: Judging, Perceiving')

# playground ------------------------------------------------------------------------------

# d1 = pd.DataFrame(df.IE.value_counts().reset_index().values, columns=["axis", "count"])
# d1 = d1.sort_index(axis = 0, ascending=True)

# d2 = pd.DataFrame(df.NS.value_counts().reset_index().values, columns=["axis", "count"])
# d2 = d2.sort_index(axis = 0, ascending=True)

# d3 = pd.DataFrame(df.TF.value_counts().reset_index().values, columns=["axis", "count"])
# d3 = d3.sort_index(axis = 0, ascending=True)

# d4 = pd.DataFrame(df.JP.value_counts().reset_index().values, columns=["axis", "count"])
# d4 = d4.sort_index(axis = 0, ascending=True)

# df3 = pd.concat([d1, d2, d3, d4], ignore_index=True)
# df3


# Section one: Prediction --------------------------------------------------------------
if selected_sect == 'MBTI Prediction Tool':
    st.title("MBTI Prediction Tool")
    st.subheader("Let's guess your MBTI...")
    user_input = st.text_input(value="", label= "Enter some text here" , help = "Type something here, then press the Enter!")

    # btn = st.button("Guess")

    # prediction
    if (user_input != ""):
        with st.spinner("Hmmm..."):
            prediction = predict(model, user_input)

        st.write("🔮 The genie thinks you're an ", prediction)
        print("Predicted : ", prediction)


# Section two: Visualisation ------------------------------------------------------------
elif selected_sect == 'Data Visualization':
    st.title("Data Visualization")
    st.subheader("What fuels the genie?")
    
    # main plot
    df3 = pd.DataFrame({
        "axis": ["IE", "NS", "TF", "JP"],
        "first": [6675, 7477, 4693, 5240],
        "second": [1999, 1197, 3981, 3434]
    })

   
    fig = px.bar(df3, x="axis", y=["first", "second"], height=400,color_discrete_sequence=px.colors.qualitative.Pastel2)
    # fig = px.bar(df3, x="axis", y=["first", "second"], barmode='group', height=400) # non-stacked (grouped plot)

    st.plotly_chart(fig)

    # expanders
    with st.expander("The Data! ✌️", expanded=False):
        # df = df.sample(frac=1).reset_index(drop=True)
        st.dataframe(df)

    with st.expander("Sentiment Analysis", expanded=True):
        col1, col2 = st.columns([3, 1.5])
        with col1:
            st.bar_chart(df2.sentiment.value_counts())

        with col2:
            st.write(df2.sentiment.value_counts())

    with st.expander("MBTI Types", expanded=True):
        temp_df=df[['label','IE','NS','TF','JP']] # every col except text
        columns = temp_df.columns.tolist()
        selected_col = st.selectbox("Choose a column to display:", columns)

        if selected_col:
            selected_df = df[selected_col]

            col1, col2 = st.columns([3.5, 1])
            with col1:
                st.bar_chart(selected_df.value_counts())

            with col2:
                st.write(selected_df.value_counts())

# Section three: Sentiment Analysis ------------------------------------------------------------

elif selected_sect == 'Sentiment Analysis':
    st.title("Sentiment Analysis Tool")
    st.subheader("Let's guess your sentiment...")
    user_input = st.text_input(value="", label= "Enter some text here:" , help = "Type something here, then press the 'Guess' button!")
    score = s.polarity_scores(user_input)

    if score['compound']>0:             
        st.write("Positive😃")

    elif score['compound']==0:    
        st.write("Neutral😶")

    elif score['compound']<0:    
        st.write("Negative😢")


# REF --------------------------------------------------------------------------

# graph colors: https://plotly.com/python/discrete-color/
# nice example: https://gist.github.com/dataprofessor/bfd5908a197a7e8a6bdf0206cc166cdc

# ARCHIVE --------------------------------------------------------------------------
# st.subheader("Text Bloop")
# # if st.checkbox("Show dataset with selected columns"):
# # get the list of columns
# columns = df.columns.tolist()
# st.write("#### Select the columns to display:")
# selected_cols = st.multiselect("", columns)
# if len(selected_cols) > 0:
#     selected_df = df[selected_cols]
#     st.dataframe(selected_df.value_counts())