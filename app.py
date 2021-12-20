import streamlit as st
import pandas as pd
from predictor import predict, load_model

st.set_page_config(
    page_title="Myer-Brigging this up...",
    page_icon="🔮"
)

st.write("""# MBTI Prediction Tool""")
# refer to : https://docs.streamlit.io/library/api-reference
# refer to : https://docs.streamlit.io/en/stable/api.html#magic-commands

with st.spinner("Loading the genie..."):
	model = load_model()

st.write('*The Data*! :sunglasses:')
df = pd.read_csv("https://drive.google.com/uc?id=1NWGrm705AS3xOWrDFfETswf2saoPCkyu")
df = df.sample(frac=1).reset_index(drop=True)
df
# st.dataframe(df) 

# st.write('*SenticNet* Results! :beach_with_umbrella:')
# df = pd.read_csv("https://drive.google.com/uc?id=1uDjEO077joBTeVgwwq7bDht0LJb8NST4")
# df = df.sample(frac=1).reset_index(drop=True)
# df = df.head(1000)
# df

# if st.checkbox("Show dataset with selected columns"):
#     # get the list of columns
#     columns = df.columns.tolist()
#     st.write("#### Select the columns to display:")
#     selected_cols = st.multiselect("", columns)
#     if len(selected_cols) > 0:
#         selected_df = df[selected_cols]
#         st.dataframe(selected_df.value_counts())

# st.write(df.label.value_counts())

st.subheader("Let's guess your MBTI...")
user_input = st.text_area(label= "Enter some text here:" , help = "Type something here, then click anywhere outside the box!")
btn = st.button("Guess!")

if (user_input != "") or btn:
    with st.spinner("Hmmm..."):
        prediction = predict(model, user_input)
    print(prediction)
    st.write("The genie thinks you're an ", prediction, "🔮")
    # st.balloons()

# btn = st.button("Party!")
# if btn:
#     st.balloons()

# nice example: https://gist.github.com/dataprofessor/bfd5908a197a7e8a6bdf0206cc166cdc