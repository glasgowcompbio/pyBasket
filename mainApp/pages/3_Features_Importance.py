import streamlit as st
from processing import readPickle, Results, defaultPlot_leg, Analysis
from importance import FI, Global
from common import add_logo, hideRows, saveTable, savePlot,sideBar
from streamlit_option_menu import option_menu

add_logo()
sideBar()
hide_rows = hideRows()
st.header("Features importance")

st.write("---")
menu = option_menu(None, ["Overview", "Global methods", "Local methods"],
    icons=["bi bi-bar-chart", "bi bi-globe", "bi bi-pin-map"],
    menu_icon="cast", default_index=0, orientation="horizontal")

if "data" in st.session_state:
    data = st.session_state["data"]
    if "basket" in st.session_state:
        basket = st.session_state["basket"]
    if "cluster" in st.session_state:
        cluster = st.session_state["cluster"]
    analysis_data = Analysis(data)
    feature_inter = FI(data)
    explainer, values = feature_inter.SHAP()
    if menu == "Overview":
        st.write(" ")
        st.subheader("Overview")
        st.write('Further exploration of the most important features (transcripts) for predicting AAC response')
        col11, col12 = st.columns((3,2))
        with col11:
            global_model = st.selectbox("Select a method", ["RF feature importance", "SHAP", "Permutation FI"], key="model1")
        with col12:
            num_feat = st.number_input('Select top number of features to display (<30 is encouraged)', value=10)

        if global_model == "RF feature importance":
            st.subheader("Most important features (Random Forest)")
            st.write("Top {} most important features calculated from Random Forest.".format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-RF")
            feature_inter.plotImportance(RawD,num_feat)
        elif global_model == "SHAP":
            st.subheader("SHAP values")
            RawD = st.checkbox("Show raw data", key="rd-SHAP")
            if RawD:
                raw_df = feature_inter.SHAP_results(values)
                saveTable(raw_df, "SHAP")
                st.dataframe(raw_df, use_container_width=True)
            else:
                fig = feature_inter.SHAP_summary(values, num_feat)
                savePlot(fig, "Global_SHAP")
                st.pyplot(fig)

        elif global_model == "Permutation FI":
            st.subheader("Permutation feature importance")
            st.write('The importance of a feature is measured by calculating the increase in the model’s prediction'
                     'error when re-shuffling each predictor. How much impact have these features in the model’s prediction for AAC response.')
            feature_inter.permutationImportance(num_feat)
    elif menu == "Global methods":
        st.write(" ")
        st.subheader("Global methods")
        st.write(" ")
        st.write('Global methods are used to interpret the average behaviour of a Machine Learning model '
                 'and how it makes predictions as a whole. ')


        st.write("#### Accumulated Local Effects")
        st.write(" ")
        st.write("Accumulated Local Effects describe how features influences the prediction made by the ML "
                 "model on average.")
        global_ALE = Global(data)
        gsamples = st.radio("", ['Use samples in the selected interaction', 'Select only samples in cluster',
                                 'Select only samples in tissue/basket'],
                            key="global_samples", horizontal=True)
        transcripts = global_ALE.transcripts
        feature = st.selectbox("Select a transcript/feature", transcripts, key="global_feature")
        if gsamples == 'Use samples in the selected interaction':
            basket, cluster = basket, cluster
            option = "interaction"
            response = st.checkbox("Split by Responsive vs Non-responsive samples")
            if response:
                global_ALE.global_ALE_resp(feature)
            else:
                global_ALE.global_ALE_single(feature, cluster,basket, option)
        else:
            if gsamples == 'Select only samples in cluster':
                groups = st.multiselect(
                    'Please select a cluster, up to 2 cluster to compare or all clusters', ["All"]+data.clusters_names, max_selections=2)
                option = "clusters"
            elif gsamples == 'Select only samples in tissue/basket':
                groups = st.multiselect(
                    'Please select a tissue, up to 2 tissues to compare or all tissues', ["All"]+data.baskets_names, max_selections=2)
                option = "baskets"
            if len(groups)<1 and "All" not in groups:
                st.write("")
            elif len(groups)<2 and "All" not in groups:
                global_ALE.global_ALE_single(feature, groups[0],None,option)
            elif "All" in groups:
                global_ALE.global_ALE(feature)
            else:
                global_ALE.global_ALE_mult(feature, groups[0], groups[1], option)
    elif menu == "Local methods":
        st.subheader("Local methods")
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
