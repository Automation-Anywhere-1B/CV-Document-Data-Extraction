import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import io
import numpy as np
from pathlib import Path
from Inference.predict_yolo import run_yolo_inference

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
        Upload your document, then click **Run All Models** to compare YOLO, YOLO Stacked, and Detectron2.
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
            st.session_state["results"]["detectron"] = {
                "image": image,
                "detections": [],
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
    st.title("YOLO Model")

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

# =========================================================
# YOLO Stacked Standalone Page (placeholder)
# =========================================================
if selected == "YOLO Stacked":
    st.title("YOLO Stacked Model")
    st.info("Stacked model integration coming soon!")

# =========================================================
# Detectron Page (placeholder)
# =========================================================
if selected == "Detectron":
    st.title("Detectron2 Model")
    st.info("Detectron2 inference coming soon!")
