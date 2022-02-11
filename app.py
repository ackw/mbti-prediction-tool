
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
    page_icon="🔮",
    layout="centered"
)

# load tensorflow model
with st.spinner("Loading the genie..."):
    model = load_model()

# datasets
df = pd.read_csv("datasets/mbti_1_cleaned_all.csv")
df2 = pd.read_csv("datasets/data.csv")

# sidebar ------------------------------------------------------------------------------
sections = ['MBTI Prediction Tool', 'Data Visualization']
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

# Section one: Prediction --------------------------------------------------------------
if selected_sect == 'MBTI Prediction Tool':
    st.title("MBTI Prediction Tool")
    st.subheader("Let's guess your MBTI...")
    user_input = st.text_input(value="", label= "Enter some text here" , help = "Type something here, then press the Enter!")
    user_input = user_input.lower()
    print(user_input)
    # btn = st.button("Guess")
    
    # Tool
    if (user_input != ""):
        if len(user_input) < 10 or user_input.isnumeric()==True or user_input.isalnum()==True:
            st.error("Invalid text! Enter text with more than 10 letters with no numbers.")
        else:
            # MBTI
            with st.spinner("Hmmm..."):
                prediction = predict(model, user_input)
        
            # Subjectivity
            sc = TextBlob(user_input).sentiment.subjectivity
            if sc > 0.5:
                subjectivity = 'Highly Opinionated'

            elif sc < 0.5:
                subjectivity = 'Not really Opinionated'
            else:
                subjectivity = 'Somewhat Subjective and Objecive'

            # Sentiment
            score = s.polarity_scores(user_input)
            if score['compound']>0:             
                st.write("The genie thinks you're an ", prediction, "+ the words are ", subjectivity, "and Positive😃")

            elif score['compound']==0:    
                st.write("The genie thinks you're an ", prediction, "+ the words are ", subjectivity, "and Neutral😶")

            elif score['compound']<0:    
                st.write("The genie thinks you're an ", prediction, "+ the words are ", subjectivity, "and Negative😢")

            print("Predicted : ", prediction)


# Section two: Visualisation ------------------------------------------------------------
elif selected_sect == 'Data Visualization':
    st.title("Data Visualization")
    # st.subheader("What fuels the genie?")

    # selection
    sections = ['Text Analysis', 'Personality Types']
    selected_viz = st.selectbox("Choose a Visualization:", sections)

    if selected_viz == 'Personality Types':

        st.subheader("4 Dimensions")
        # donuts
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
            "layout": {
                # "title":'MBTI Traits - Donuts',
                "piecolorway":px.colors.qualitative.Pastel2
            }
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
            # fig = px.bar(df3, x="axis", y=["first", "second"], barmode='group', height=400) # non-stacked (grouped plot)
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

        st.subheader("Top 30 Words Used")
        # value counts bar chart - top 30 words
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
        
# REF --------------------------------------------------------------------------

# graph colors: https://plotly.com/python/discrete-color/
# nice example: https://gist.github.com/dataprofessor/bfd5908a197a7e8a6bdf0206cc166cdc

# with st.expander("Sentiment Analysis"):
#     col1, col2 = st.columns([3, 1.5])
#     with col1:
#         st.bar_chart(df2.sentiment.value_counts())

#     with col2:
#         st.write(df2.sentiment.value_counts())

# with st.expander("Subjectivity Analysis"):
#     col1, col2 = st.columns([3, 1.5])
#     with col1:
#         st.bar_chart(df2.subjectivity.value_counts())

#     with col2:
#         st.write(df2.subjectivity.value_counts())

# with st.expander("The Data! ✌️", expanded=False):
#     # df = df.sample(frac=1).reset_index(drop=True)
#     st.dataframe(df)

# with st.expander("MBTI Types"):
#     col1, col2 = st.columns([3, 1.5])
#     with col1:
#         st.bar_chart(df2.mbti.value_counts())

#     with col2:
#         st.write(df2.mbti.value_counts())

# with st.expander("MBTI Types"):
#     temp_df=df[['label','IE','NS','TF','JP']] # every col except text
#     temp_df.rename(columns={'label': 'MBTI'}, inplace=True)
#     columns = temp_df.columns.tolist()
#     selected_col = st.selectbox("Choose a column to display:", columns)

#     if selected_col:
#         selected_df = temp_df[selected_col]

#         col1, col2 = st.columns([3.5, 1])
#         with col1:
#             st.bar_chart(selected_df.value_counts())

#         with col2:
#             st.write(selected_df.value_counts())



# col1, col2 = st.columns([1.5, 3])
#         # dimensions value counts
#         with col1:
#             df3 = pd.DataFrame({
#             "axis": ["IE", "NS", "TF", "JP"],
#             "trait 1": [6675, 7477, 4693, 5240],
#             "trait 2": [1999, 1197, 3981, 3434]
#             })

#             fig = px.bar(df3, x="axis", y=["trait 1", "trait 2"], height=400, width=300, color_discrete_sequence=px.colors.qualitative.Pastel2)
#             # fig = px.bar(df3, x="axis", y=["first", "second"], barmode='group', height=400) # non-stacked (grouped plot)
#             fig.update_layout(legend_title_text='', showlegend=False)
#             st.plotly_chart(fig)

#         # MBTI value counts
#         with col2:
#             df3 = pd.DataFrame({
#             "mbti": ["INFP", "INFJ", "INTP", "INTJ", "ENTP", "ENFP", "ISTP", "ISFP", "ENTJ", "ISTJ", "ENFJ", "ISFJ", "ESTP", "ESFP", "ESFJ", "ESTJ"],
#             "value": [1831, 1470, 1304, 1091, 685, 675, 337, 271, 231, 205, 190, 166, 89, 48, 42, 39],
#             })

#             fig = px.bar(df3, x="mbti", y="value", height=400, width=500, color_discrete_sequence=px.colors.qualitative.Pastel2)
#             fig.update_layout(legend_title_text='', showlegend=False)
#             st.plotly_chart(fig)