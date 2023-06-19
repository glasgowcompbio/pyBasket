import streamlit as st
from processing import readPickle, Results, add_logo, defaultPlot_leg, Analysis, hideRows
from mpld3 import plugins
from mpld3.utils import get_id
import mpld3
import collections
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
from interpretation import Prototypes, DEA, FI
from lime import lime_tabular
from sklearn.cluster import KMeans
import sklearn
import sklearn.ensemble

add_logo()
hide_rows = hideRows()
st.header("Features importance")
if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = Analysis(data)
    feature_inter = FI(data)
    tab1, tab2 = st.tabs(["Global Importance","Individual predictions (LIME)"])
    with tab1:
        st.subheader("Transcripts with the highest importance")
        feature_inter.plotImportance()
        st.subheader("Permutation based")
        feature_inter.permutationImportance()
    with tab2:
        st.subheader("LIME for individual predictions")
        feature_inter.limeInterpretation(20)

