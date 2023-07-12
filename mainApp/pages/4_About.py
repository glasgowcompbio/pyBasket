import streamlit as st
from common import add_logo, sideBar

add_logo()
sideBar()
st.header("About")

st.write("This App has been developed by Marina Flores Payan as part of the MSc Bioinformatics project at the University of Glasgow.")

st.write(
    """
    The web interface of the pyBasket pipeline has been implemented based on the open source Python's Streamlit library. 
    The back-end computations and the interpretable Machine Learning models have been developed using several Python's libraries: shap, scikit-learn,
     lime, etc.
     """
)