import streamlit as st
from common import add_logo, sideBar

add_logo()
sideBar()
st.header("About")
url_streamlit = "https://streamlit.io"
repository = "https://github.com/marinafloresp/MScDiss"
url_pybasket = "https://glasgowcompbio.github.io/pyBasket/"
lime = "https://github.com/marcotcr/lime"
shap = "https://shap.readthedocs.io/en/latest/index.html"
altair = "https://altair-viz.github.io/index.html"
st.write("The EMERY app has been developed by Marina Flores Payan as the final project for the MSc in Bioinformatics at the University of Glasgow.")
st.write("For more information about pyBasket, visit the pyBasket [page](%s)" % url_pybasket)

st.write("##### Web framework")
st.write("EMERY app [repository](%s). --- For full scripts and code underlying the pyBasket app." % repository)
st.write(
    "The web interface of EMERY has been implemented based on the open source Python's Streamlit library ([link](%s))" % url_streamlit)
st.write("The back-end computations and the interpretable Machine Learning models have been developed using several Python's libraries: [shap](%shap), scikit-learn, [lime](%lime), [altair](%altair), etc.")


