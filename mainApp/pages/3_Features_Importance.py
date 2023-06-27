import streamlit as st
from processing import readPickle, Results, defaultPlot_leg, Analysis
from importance import FI
from common import add_logo, hideRows, saveTable, savePlot,sideBar



add_logo()
sideBar()
hide_rows = hideRows()
st.header("Features importance")
if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = Analysis(data)
    feature_inter = FI(data)
    explainer, values = feature_inter.SHAP()
    tab1, tab2, tab3 = st.tabs(["Overview","Global Methods","Local Methods"])
    with tab1:
        st.write(" ")
        st.subheader("Features importance overview")
        st.write('Further exploration of the most important features (transcripts) for predicting AAC response')
        global_model = st.selectbox("Select a method", ["RF feature importance", "SHAP", "Permutation FI"], key="model1")
        if global_model == "RF feature importance":
            st.subheader("Most important features (Random Forest)")
            st.write("Top 25 most important features calculated from Random Forest.")
            RawD = st.checkbox("Show raw data", key="rd-RF")
            feature_inter.plotImportance(RawD)
        elif global_model == "SHAP":
            st.subheader("SHAP values")
            RawD = st.checkbox("Show raw data", key="rd-SHAP")
            if RawD:
                raw_df = feature_inter.SHAP_results(values)
                saveTable(raw_df, "SHAP")
                st.dataframe(raw_df, use_container_width=True)
            else:
                fig = feature_inter.SHAP_summary(explainer, values)
                savePlot(fig, "Global_SHAP")
                st.pyplot(fig)

        elif global_model == "Permutation FI":
            st.subheader("Permutation feature importance")
            st.write('The importance of a feature is measured by calculating the increase in the model’s prediction'
                     'error when re-shuffling each predictor. How much impact have these features in the model’s prediction for AAC response.')
            feature_inter.permutationImportance()
    with tab2:
        st.write(" ")
        st.subheader("Global methods")
        st.write(" ")
        st.write('Global methods are used to interpret the average behaviour of a Machine Learning model'
                 'and how it makes predictions as a whole. ')

    with tab3:
        st.subheader("Local methods")
        if "basket" in st.session_state:
            basket = st.session_state["basket"]
        if "cluster" in st.session_state:
            cluster = st.session_state["cluster"]
        local_model = st.selectbox("Select a method", ["LIME", "SHAP"], key="model2")
        group_samples = st.radio("",['Use samples in interaction', 'Select only samples in cluster', 'Select only samples in tissue/basket'],
                             key="samples", horizontal=True)

        if group_samples == 'Use samples in selection':
            basket, cluster = basket, cluster
        elif group_samples == 'Select only samples in cluster':
            cluster= st.selectbox("Select a cluster", data.setClusters(), key="only_cluster")
            basket = "None"
        elif group_samples == 'Select only samples in tissue/basket':
            basket = st.selectbox("Select a basket/tissue", data.setBaskets(), key="only_basket")
            cluster = "None"
        transc, size = feature_inter.displaySamples(cluster,basket)
        st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket, size))
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
                st.write("  ")
                st.write("##### Forces plot")
                st.write("  ")
                RawD = st.checkbox("Show raw data", key="raw-data-SHAP")
                feature_inter.SHAP_forces(sample,explainer,values,n_features, RawD)
                st.write("  ")
                st.write("##### Bar plot")
                st.write("  ")
                RawD_bar = st.checkbox("Show raw data", key="raw-data-SHAP_bar")
                feature_inter.SHAP_bar_indiv(sample, explainer, values, n_features, RawD_bar)
