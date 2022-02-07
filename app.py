import streamlit as st
import pandas as pd
import altair as alt
from predictor import predict, load_model

st.set_page_config(
    page_title="Myer-Brigging this up...",
    page_icon="🔮"
)

# load tensorflow model upon page launch
with st.spinner("Loading the genie..."):
    model = load_model()

# datasets
df = pd.read_csv("datasets/mbti_1_cleaned_all.csv")
df2 = pd.read_csv("datasets/multitask_mbti.csv")

st.title("MBTI Prediction Tool")


# section one: prediction --------------------------------------------------------------
st.subheader("Let's guess your MBTI...")
user_input = st.text_input(value="", label= "Enter some text here:" , help = "Type something here, then press the 'Guess' button!")
btn = st.button("Guess")

# commented out function
if (user_input != "") or btn:
    with st.spinner("Hmmm..."):
        prediction = predict(model, user_input)
    print(prediction)
    
    st.write("The genie thinks you're an ", prediction, "🔮")


# section two: Visualisation ------------------------------------------------------------
st.subheader("What fuels the genie?")

if st.checkbox("Show!"):
    with st.expander("The Data! ✌️", expanded=False):
        # df = df.sample(frac=1).reset_index(drop=True)
        st.dataframe(df)

    with st.expander("Sentiment Analysis", expanded=False):
        col1, col2 = st.columns([3, 1.5])
        with col1:
            st.bar_chart(df2.sentiment.value_counts())

        with col2:
            st.write(df2.sentiment.value_counts())

    with st.expander("MBTI Types", expanded=False):
        temp_df=df[['label','IE','NS','TF','JP']] # every col except text
        columns = temp_df.columns.tolist()
        selected_col = st.selectbox("Select the column to display:", columns)

        if selected_col:
            selected_df = df[selected_col]
            # st.bar_chart(selected_df.value_counts())

            col1, col2 = st.columns([3.5, 1])
            with col1:
                st.bar_chart(selected_df.value_counts())

            with col2:
                st.write(selected_df.value_counts())
        

# sidebar ------------------------------------------------------------------------------
st.sidebar.markdown('## MBTI Types')
st.sidebar.markdown('**Analysts :** [INTJ, INTP, ENTJ, ENTP]')
st.sidebar.markdown('**Diplomats :** [INFJ, INFP, ENFJ, ENFP]')
st.sidebar.markdown('**Sentinels :** [ISTJ, ISFJ, ESTJ, ESFJ]')
st.sidebar.markdown('**Explorers :** [ISTP, ISFP, ESTP, ESFP]')

st.sidebar.markdown('## What each letter mean')
st.sidebar.markdown('I = Introvert, E = Extrovert')
st.sidebar.markdown('N = Intuition, S = Sensing')
st.sidebar.markdown('T = Thinking, F = Feeling')
st.sidebar.markdown('J = Judging, P = Perceiving')

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