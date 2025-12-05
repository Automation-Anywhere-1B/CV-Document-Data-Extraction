import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import io
import sys
import numpy as np
from pathlib import Path
from Inference.predict_yolo import run_yolo_inference
from Inference.predict_detectron import run_detectron_inference

st.set_page_config(page_title="CalVision", layout="wide")

# =========================================================
#   Load YOLO model (cached, runs once)
# =========================================================
YOLO_WEIGHTS_PATH = (
    Path("Yolo_Model") / "runs" / "detect" / "train" / "weights" / "best.pt"
)



# Detect if path exists (debug help)
if not YOLO_WEIGHTS_PATH.exists():
    st.error(f"❌ YOLO weights not found at: {YOLO_WEIGHTS_PATH}")
#else:
    #st.sidebar.success(f"Using YOLO model:\n{YOLO_WEIGHTS_PATH}")

# =========================================================
# Sidebar Navigation
# =========================================================
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "YOLO Model", "YOLO Stacked", "Detectron"],
        menu_icon="cast",
        default_index=0,
    )

# =========================================================
# HOME PAGE — Full Workflow
# =========================================================
if selected == "Home":
    st.title("📄 Welcome to CalVision")
    st.subheader("Document Vision Model Comparison Tool")

    st.markdown(
        """
        ### 📌 Instructions  
        
        **Welcome!** In order to submit your photo, please make sure to follow these simple instructions:

        :one: Position your check in the center of your camera

        :two: Make sure you have good lighting and the **_whole_** check is visible

        :three: Confirm that your check is in **_focus_** and there are no other objects in the photo

        :four: You are ready to submit!

        :five: Upload your document, then click **Run All Models** to compare YOLO, YOLO Stacked, and Detectron2.
        """
    )

    # Upload
    uploaded_file = st.file_uploader(
        "Upload your document image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save to session state
        st.session_state["uploaded_image"] = image
        st.session_state["uploaded_bytes"] = uploaded_file.getvalue()

    # Run all models
    if uploaded_file and st.button("Run All Models", type="primary"):
        st.session_state["results"] = {}

        # ---------------------------------------------------
        # YOLO Model
        # ---------------------------------------------------
        with st.spinner("Running YOLO Model..."):
            annotated, detections = run_yolo_inference(
                image_bytes=st.session_state["uploaded_bytes"],
                weights_path=str(YOLO_WEIGHTS_PATH),
            )

            st.session_state["results"]["yolo"] = {
                "image": annotated,
                "detections": detections
            }

        # ---------------------------------------------------
        # YOLO Stacked (placeholder)
        # ---------------------------------------------------
        with st.spinner("Running YOLO Stacked..."):
            st.session_state["results"]["stacked"] = {
                "image": image,
                "detections": [],
            }

        # ---------------------------------------------------
        # Detectron2 (placeholder)
        # ---------------------------------------------------
        with st.spinner("Running Detectron2..."):
            det_annotated, det_detections = run_detectron_inference(
                st.session_state["uploaded_bytes"]
            )
            st.session_state["results"]["detectron"] = {
                "image": det_annotated,
                "detections": det_detections,
            }

        st.success("All models finished running!")

    # Display results
    if "results" in st.session_state:
        st.header("📊 Model Comparison")

        tabs = st.tabs(["YOLO", "YOLO Stacked", "Detectron2"])

        # YOLO tab
        with tabs[0]:
            st.subheader("YOLO Results")
            st.image(
                st.session_state["results"]["yolo"]["image"], use_column_width=True
            )
            st.json(st.session_state["results"]["yolo"]["detections"])

        # Stacked
        with tabs[1]:
            st.subheader("YOLO Stacked Results")
            st.image(
                st.session_state["results"]["stacked"]["image"], use_column_width=True
            )
            st.json(st.session_state["results"]["stacked"]["detections"])

        # Detectron
        with tabs[2]:
            st.subheader("Detectron2 Results")
            st.image(
                st.session_state["results"]["detectron"]["image"],
                use_column_width=True,
            )
            st.json(st.session_state["results"]["detectron"]["detections"])

# =========================================================
# YOLO Standalone Page
# =========================================================
if selected == "YOLO Model":
    st.title("YOLO Model Metrics")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Show uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run YOLO inference
        if st.button("Run YOLO Inference", type="primary"):
            with st.spinner("Running YOLO Model..."):
                annotated, detections = run_yolo_inference(
                    image_bytes=uploaded_file.getvalue(),
                    weights_path=str(YOLO_WEIGHTS_PATH)
                )

            st.subheader("YOLO Output")
            st.image(annotated, caption="Predicted Image", use_column_width=True)

            st.subheader("Detections")
            st.json(detections)

    st.title("Overall Metrics")
    st.image("Yolo_Model/runs/detect/train/confusion_matrix_normalized.png", caption="Confusion Matrix (Normalized)", width=600)
    st.image("Yolo_Model/runs/detect/train/BoxF1_curve.png", caption="F1 Curve", width=600)
    st.image("Yolo_Model/runs/detect/train/BoxR_curve.png", caption="Recall Confidence Curve", width=600)


# =========================================================
# YOLO Stacked Standalone Page (placeholder)
# =========================================================
if selected == "YOLO Stacked":
    st.title("YOLO Stacked Model Metrics")
    st.info("Stacked model integration coming soon!")

# =========================================================
# Detectron Page (placeholder)
# =========================================================
# if selected == "Detectron":
#     st.title("Detectron2 Model")
#     st.info("Detectron2 inference coming soon!")
if selected == "Detectron":
    st.title("Detectron2 Model Metrics")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()

        if st.button("Run Detectron2 Inference", type="primary"):
            with st.spinner("Running Detectron2..."):
                annotated, detections = run_detectron_inference(image_bytes)

            st.subheader("Detectron2 Output")
            st.image(annotated, caption="Predicted Image", use_column_width=True)

            st.subheader("Detections")
            st.json(detections)
    else:
        st.info("Please upload an image to run Detectron2.")

    st.title("Overall Metrics")
    st.image("Detectron2/output_saved/iou_distribution.png", caption="IOU Distribution", width=500)
    st.image("Detectron2/output_saved/loss_curve.png", caption="Loss Curve", width=500)
    st.image("Detectron2/output_saved/pr_curves.png", caption="Precision Curves", width=500)