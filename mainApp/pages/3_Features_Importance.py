import streamlit as st
from processing import readPickle, Results, add_logo, defaultPlot_leg, Analysis, hideRows
from mpld3 import plugins
from mpld3.utils import get_id
import mpld3
import collections
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
from interpretation import Prototypes, DEA
from importance import FI
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
    explainer, values = feature_inter.SHAP()
    tab1, tab2 = st.tabs(["Global Importance","Individual predictions"])
    with tab1:
        global_model = st.selectbox("Select a method", ["RF feature importance", "SHAP", "Permutation"], key="model1")
        if global_model == "RF feature importance":
            st.subheader("Transcripts with the highest importance")
            feature_inter.plotImportance()
        elif global_model == "SHAP":
            st.subheader("SHAP values")
            fig = feature_inter.SHAP_summary(explainer,values)
            st.pyplot(fig)
        elif global_model == "Permutation":
            None
            #st.subheader("Permutation based")
            #feature_inter.permutationImportance()
    with tab2:
        st.subheader("Local methods")
        local_model = st.selectbox("Select a method", ["LIME", "SHAP"], key="model2")
        col21, col22 = st.columns((2,2))
        with col21:
            cluster = st.selectbox("Select a cluster", ["None"]+data.setClusters(),key="cluster")
            #responsive = st.selectbox("Select response to treatment", ["Responsive (1)", "Non-responsive (0)"],key="response")
        with col22:
            basket = st.selectbox("Select a tissue/basket", ["None"]+data.setBaskets(),key="basket")
        transc, size = feature_inter.displaySamples(cluster,basket)
        st.info("###### Samples in selection: {}".format(size))
        if size >1:
            responses = st.radio("",
                                 ['All', 'Only responsive samples', "Only non-responsive samples"],
                                 key="responsive", horizontal=True)
            col23, col24 = st.columns((2,1))
            with col23:
                transc = feature_inter.filterSamples(transc, responses)
                sample = st.selectbox("Select a sample", transc, key="sample")
            with col24:
                n_features = st.number_input('Number of features', value=5)
            if local_model == "LIME":
                st.subheader("LIME for individual predictions")
                RawD = st.checkbox("Show raw data", key="raw-data-LIME")
                feature_inter.limeInterpretation(sample,n_features,RawD)
            elif local_model == "SHAP":
                st.write("  ")
                st.write("##### SHAP explanation for sample: {}".format(sample))
                RawD = st.checkbox("Show raw data", key="raw-data-LIME")
                feature_inter.SHAP_forces(sample,explainer,values,n_features, RawD)
