from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from PIL import Image  # for opening images


# --------------------------------------------
# SETUP SESSION STATE
# This makes sure we have a place to store:
# - last prediction result
# - last model metrics
#
# Session state = temporary memory Streamlit uses
# --------------------------------------------

# If we do not yet have a "prediction" saved, create it
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None  # nothing stored yet

# Same for metrics
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None     # nothing stored yet
    
# --------------------------------------------
# DUMMY MODEL FUNCTIONS
# These are placeholders so the interface works.
# Later, your teammates can replace the insides
# with real YOLO / YOLO Stacked / Detectron code.
# --------------------------------------------

def run_yolo(image):
    """Fake YOLO model. Returns a fake label and metrics."""
    prediction = "YOLO: check detected, amount = $123.45"
    metrics = {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.90
    }
    return prediction, metrics

def run_yolo_stacked(image):
    """Fake YOLO Stacked model."""
    prediction = "YOLO Stacked: check detected, amount = $130.00"
    metrics = {
        "accuracy": 0.97,
        "precision": 0.95,
        "recall": 0.92
    }
    return prediction, metrics

def run_detectron(image):
    """Fake Detectron model."""
    prediction = "Detectron: check detected, amount = $118.00"
    metrics = {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.88
    }
    return prediction, metrics
 # -----------------------------------------------------------------------------

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
    # Page switcher
    # This lets the user switch between the PREDICT page and VIEW METRICS page
    page = st.radio("Page", ["Predict", "View Metrics"])

    # Reset Button
    # When the user clicks this, we clear all stored values and reload the app
    if st.button("Reset", type="secondary"):
        st.session_state.clear()  # Removes saved predictions/metrics
        st.rerun()                # Reloads the app (new name in Streamlit)

# Title
st.title(":red[_Welcome to CalVision_]")

# Subheader
st.subheader("Instructions")

# Instructions Text
st.markdown("**Welcome!**\nIn order to submit your photo for parsing, please make sure to follow these simple instructions.")
st.markdown(":one: Position your check in the center of your camera")
st.markdown(":two: Make sure you have good lighting and the **_whole_** check is visible")
st.markdown(":three: Confirm that your check is in focus and there are no other objects in the photo")
st.markdown(":four: You are ready to submit!!")



# --------------------------------------------
# PREDICT PAGE CONTENT
# We only want to show the upload box and buttons
# when the user has selected the "Predict" page
# --------------------------------------------
if page == "Predict":
    st.subheader("Upload your check")

    # File uploader
    # Only allow image files: jpg, jpeg, png
    uploaded_file = st.file_uploader(
        "Upload a picture of your check",
        type=["jpg", "jpeg", "png"]
    )

    # This variable will hold the opened image
    image = None

    # If the user uploaded a file, open it as an image and show it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # open the image
        st.image(image, caption="Uploaded check", use_container_width=True)

    # Simple submit button for now
    submit = st.button("Submit Your Photo", type="primary")

    if submit:
        if image is None:
            # User clicked submit but did not upload an image
            st.warning("Please upload an image first.")
        else:
            # --------------------------------------------
            # CALL THE CORRECT MODEL FUNCTION
            # based on what the user chose in the sidebar
            # --------------------------------------------
            if selected == "YOLO":
                prediction, metrics = run_yolo(image)
            elif selected == "YOLO Stacked Model":
                prediction, metrics = run_yolo_stacked(image)
            elif selected == "Detectron":
                prediction, metrics = run_detectron(image)
            else:
                prediction, metrics = None, None

            # --------------------------------------------
            # SAVE RESULTS IN SESSION STATE
            # so that the View Metrics page can show them
            # --------------------------------------------
            st.session_state["prediction"] = prediction
            st.session_state["metrics"] = metrics

            # --------------------------------------------
            # Give feedback to the user
            # --------------------------------------------
            st.success("Photo submitted and prediction saved!")
            st.write("Prediction (quick view):")
            st.write(prediction)



# --------------------------------------------
# VIEW METRICS PAGE
# Shows the last prediction and its metrics
# --------------------------------------------
if page == "View Metrics":
    st.subheader("Model Prediction and Metrics")

    # If no prediction has been made yet, tell the user what to do
    if st.session_state["prediction"] is None:
        st.info("No predictions yet. Go to the 'Predict' page, upload an image, and click 'Submit Your Photo'.")
    else:
        # Show the prediction text
        st.markdown("### Prediction")
        st.write(st.session_state["prediction"])

        # Show the metrics in a small table
        st.markdown("### Metrics")
        metrics = st.session_state["metrics"]

        # If metrics is a dictionary, turn it into a table
        if isinstance(metrics, dict):
            metrics_df = pd.DataFrame(
                list(metrics.items()),
                columns=["Metric", "Value"]
            )
            st.table(metrics_df)
        else:
            # If it's something else, just print it
            st.write(metrics)
