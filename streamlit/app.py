from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

# Sidebar menu
with st.sidebar:
    selected = option_menu("Model Options", ["YOLO", 'YOLO Stacked Model', 'Detectron'], 
      menu_icon="none", 
      default_index=1,
      styles = {
        "container": {"padding-top": "350px", "background-color": "#fafafa"},
        "menu-icon": {"text-align": "left"},
        "nav-link": {"text-align": "left"},
      }
    )
    selected

# Title
st.title(":red[_Welcome to CalVision_ - test]")

# Subheader
st.subheader("Instructions")

# Instructions Text
st.markdown("**Welcome!**\nIn order to submit your photo for parsing, please make sure to follow these simple instructions.")
st.markdown(":one: Position your check in the center of your camera")
st.markdown(":two: Make sure you have good lighting and the **_whole_** check is visible")
st.markdown(":three: Confirm that your check is in focus and there are no other objects in the photo")
st.markdown(":four: You are ready to submit!!")



# Uploading Image
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)


# Submit Button
submit = st.button("Submit Your Photo", type="primary")