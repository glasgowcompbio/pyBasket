import streamlit as st
from common import add_logo,sideBar


add_logo()
sideBar()
st.write("## About this App")
st.markdown(
    """
    This is a beta version of the App I am building for my MSc project.
    ### More to come
    """
)