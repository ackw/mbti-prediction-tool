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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# sentiment analysis & subjectivity
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
s = SentimentIntensityAnalyzer()
from textblob import TextBlob

# Initialisation ----------------------------------------------------------------------

def predict(model, input):
    print("User input : ", input)
    class_names = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

    # GloVe
    with open(picklepath, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        seq = loaded_tokenizer.texts_to_sequences([input])
    
    padded = sequence.pad_sequences(seq, maxlen=max_sentence_length)
    pred = model.predict(padded)
    prediction = format(class_names[np.argmax(pred[0])])

    return prediction


st.set_page_config(
    page_title="Myer-Brigging this up...",
    page_icon="🔮",
    layout="centered"
)

# Datasets
df = pd.read_csv("datasets/mbti_1_cleaned_all.csv")
df2 = pd.read_csv("datasets/data.csv")

# Sidebar ------------------------------------------------------------------------------

st.sidebar.subheader("Choose a Feature")
sections = ['MBTI Prediction Tool', 'Data Visualization']
selected_sect = st.sidebar.selectbox("Predict or Visualize:", sections)

# Section one: Prediction --------------------------------------------------------------

if selected_sect == 'MBTI Prediction Tool':
    st.title("MBTI Prediction Tool")
    st.subheader("Let's guess your MBTI...")
    user_input = st.text_input(value="", label= "Enter some text here" , help = "Type something here, then press the Enter!")

    # simple text cleaning
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]','', user_input)
    user_input = re.sub(r"\d", '', user_input)

    print(user_input)

    # sidebar
    mdls = ['CNN + PersonalityCafe Data', 'CNN + Reddit Data']
    selected_mod = st.radio(label = 'Choose a Model for Prediction', options = mdls)

    # load models
    # 1. Model trained with Kaggle Data
    if selected_mod =='CNN + PersonalityCafe Data':
        filepath = "model/model_kaggle.h5"
        picklepath = "model/tokenizer_kaggle.pickle"
        max_sentence_length = 764 

        with st.spinner("Loading the genie..."):
            model = keras.models.load_model(filepath)
    
    # 2. Model trained with Reddit Data
    else:
        filepath = "model/model_reddit.h5"
        picklepath = "model/tokenizer_reddit.pickle"
        max_sentence_length = 881 
        
        with st.spinner("Loading the genie..."):
            model = keras.models.load_model(filepath)

    # MBTI Prediction Tool
    if (user_input != ""):
        if len(user_input) < 10:
            st.error("Invalid text! Enter text with more than 10 letters")
        else:
            # MBTI
            with st.spinner("Hmmm..."):
                prediction = predict(model, user_input)

            print("Predicted : ", prediction)

            # Subjectivity
            sc = TextBlob(user_input).sentiment.subjectivity
            if sc > 0.5:
                subjectivity = 'Opinionated'

            elif sc < 0.5:
                subjectivity = 'Subjective'

            else:
                subjectivity = 'Somewhat Subjective and Objecive'

            # Sentiment
            score = s.polarity_scores(user_input)
            if score['compound']>0:             
                st.subheader("Hmm..")
                st.write("The genie thinks you're an ", prediction, "+ the words are ", subjectivity, "and Positive😃")

            elif score['compound']==0:    
                st.subheader("Hmm..")
                st.write("The genie thinks you're an ", prediction, "+ the words are ", subjectivity, "and Neutral😶")

            elif score['compound']<0:    
                st.subheader("Hmm..")
                st.write("The genie thinks you're an ", prediction, "+ the words are ", subjectivity, "and Negative😢")
            

# Section two: Visualisation ------------------------------------------------------------

elif selected_sect == 'Data Visualization':
    st.title("Data Visualization with PersonalityCafe Data")

    # sidebar
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

    # Selection Dropdown
    sections = ['Text Analysis', 'Personality Types']
    selected_viz = st.selectbox("Choose a Visualization:", sections)

    # Selection one
    if selected_viz == 'Personality Types':
        st.subheader("4 Dimensions")
        # Donut Charts
        fig1 = {
            "data": [
                {"values": [6675, 1999], "labels": ["I","E"], "domain": {"x": [0.2, 0.5], "y": [0.5, .95]}, 
                "hoverinfo":"label+percent", "hole": .4,"type": "pie"},
                {"values": [7477, 1197], "labels": ["N","S"], "domain": {"x": [0.51, 0.8], "y": [0.5, .95]},
                "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [4693, 3981], "labels": ["T","F"], "domain": {"x": [0.2, 0.5], "y": [0, 0.45]},
                "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [5240, 3434], "labels": ["J","P"], "domain": {"x": [0.51, 0.8], "y": [0, 0.45]},
                "hoverinfo":"label+percent", "hole": .4, "type": "pie"}],  
            "layout": {"piecolorway": px.colors.qualitative.Pastel2}
        }

        st.plotly_chart(fig1) 

        st.subheader("16 Personality Types")
        # MBTI value counts
        df3 = pd.DataFrame({
        "mbti": ["INFP", "INFJ", "INTP", "INTJ", "ENTP", "ENFP", "ISTP", "ISFP", "ENTJ", "ISTJ", "ENFJ", "ISFJ", "ESTP", "ESFP", "ESFJ", "ESTJ"],
        "value": [1831, 1470, 1304, 1091, 685, 675, 337, 271, 231, 205, 190, 166, 89, 48, 42, 39],
        })

        fig2 = px.bar(df3, x="mbti", y="value", height=400, color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig2.update_layout(legend_title_text='', showlegend=False)
        st.plotly_chart(fig2)

    # Selection two
    elif selected_viz == 'Text Analysis':
        st.subheader("Sentiment & Subjectivity Analysis")
        col1, col2 = st.columns(2)
        # Sentiment
        with col1:
            df3 = pd.DataFrame({
            "sentiment": ["Positive", "Negative", "Neutral"],
            "value": [7530, 1127, 17],
            })
        
            fig3 = px.bar(df3, x="sentiment", y="value", height=400, width=400, color_discrete_sequence=px.colors.qualitative.Pastel1)
            fig3.update_layout(legend_title_text='', showlegend=False)
            st.plotly_chart(fig3)

        # Subjectivity
        with col2:
            df3 = pd.DataFrame({
            "subjectivity": ["Subjective", "Objective", "Neutral"],
            "value": [7285, 1388, 1],
            })

            fig4 = px.bar(df3, x="subjectivity", y="value", height=400, width=400, color_discrete_sequence=px.colors.qualitative.Plotly)
            fig4.update_layout(legend_title_text='', showlegend=False)
            st.plotly_chart(fig4)

        # Top 30 words
        st.subheader("Top 30 Words Used")
        df_wc = pd.read_csv("datasets/wordcloud.csv")
        df_wc = df_wc.iloc[:30]

        fig5 = px.bar(df_wc, x="Word", y="WordCount", height=400, color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig5.update_layout(legend_title_text='', showlegend=False)
        st.plotly_chart(fig5)

        if st.checkbox("Show wordcloud"):
            # Word Cloud
            df_wc = pd.read_csv("datasets/wordcloud.csv")

            words = " ".join(df_wc.Word)
            cloud = WordCloud(width=800, height=400, max_words=200, background_color='White', colormap='tab10').generate(words)

            plt.imshow(cloud, interpolation='gaussian')
            plt.axis("off")
            st.pyplot(plt)