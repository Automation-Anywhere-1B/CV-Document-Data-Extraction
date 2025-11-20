from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pandas as pd

# Title
st.title(":red[_Welcome to CalVision_]")

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
