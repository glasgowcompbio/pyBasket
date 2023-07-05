import streamlit as st
from processing import readPickle, Results, defaultPlot_leg, Analysis
from importance import FI, Global
from common import add_logo, hideRows, saveTable, savePlot,sideBar,openGeneCard
from streamlit_option_menu import option_menu
import webbrowser

add_logo()
sideBar()
hide_rows = hideRows()
st.header("Features importance")

st.write("---")
menu = option_menu(None, ["Overview", "Global methods", "Local methods", "Features interaction"],
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
        st.write('Feature Importance is a technique to describe how relevant an input feature is and its effect on the model '
                 'being used to predict an outcome. Here, the feature importance is calculated using several Model-Agnostic '
                 'methods to find the transcripts that have the most important impact on predicting the AAC response of a sample. '
                 )
        col11, col12 = st.columns((3,2))
        with col11:
            global_model = st.selectbox("Select a method", ["RF feature importance", "SHAP", "Permutation Based"], key="model1")
        with col12:
            num_feat = st.number_input('Select top number of features to display (<30 is encouraged)', value=10)

        if global_model == "RF feature importance":
            st.subheader("Random Forest")
            st.write("Ramdom Forest is a common ML model that combines the output of multiple decision trees to reach a single result. It has been used "
                     "as part of the pyBasket pipeline to select the 500 most important features. Below is shown the top {} most important features calculated "
                     "from Random Forest, ordered by descending importance.".format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-RF")
            feature_inter.plotImportance(RawD,num_feat)
        elif global_model == "SHAP":
            st.subheader("SHAP values")
            st.write("SHAP (SHapley Additive exPlanations) is a technique that explains the prediction of an observation by "
                         "computing the contribution of each feature to the prediction. It is based on Shapley values from game theory,"
                         " as it uses fair allocation results from cooperative game to allocate credit for a model's output among its input features."
                         "Below, the top {} features that most contribute to predict the AAC response are shown".format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-SHAP")
            if RawD:
                raw_df = feature_inter.SHAP_results(values)
                saveTable(raw_df, "SHAP")
                st.dataframe(raw_df, use_container_width=True)
            else:
                fig = feature_inter.SHAP_summary(values, num_feat)
                savePlot(fig, "Global_SHAP")
                st.pyplot(fig)

        elif global_model == "Permutation Based":

            st.subheader("Permutation based feature importance")
            st.write('Permutation based feature importance is a MA method that measures'
                     ' the importance of a feature by calculating the increase in the model’s prediction'
                     'error when re-shuffling each predictor. Below is shown how much impact have the top {} features have in the model’s prediction for AAC response.'.format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-PBI")
            feature_inter.permutationImportance(num_feat,RawD)
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
            try:
                if response:
                    global_ALE.global_ALE_resp(feature)
                else:
                    global_ALE.global_ALE_single(feature, cluster,basket, option)
            except:
                st.warning("Not enough samples in the selected basket*cluster interaction. Please try a different combination.")
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
        st.subheader("Local MA methods")
        st.write(" ")
        st.write("Local Model-Agnostic interpretation methods aim to explain individual predictions made by a Machine Learning model.")
        col31, col32 = st.columns((2, 2))
        with col31:
            local_model = st.selectbox("Select a local interpretable method", ["LIME", "SHAP"], key="model2")
        st.write("---")
        col33, col34, col35 = st.columns((2,2,2))
        with col33:
            group_samples = st.radio("", ['Use samples in interaction', 'Select only samples in cluster',
                                      'Select only samples in tissue/basket'],
                                 key="samples")
        if group_samples == 'Use samples in selection':
            basket, cluster = basket, cluster
        elif group_samples == 'Select only samples in cluster':
            with col35:
                cluster= st.selectbox("Select a cluster", data.setClusters(), key="only_cluster")
            basket = "None"
        elif group_samples == 'Select only samples in tissue/basket':
            with col35:
                basket = st.selectbox("Select a basket/tissue", data.setBaskets(), key="only_basket")
            cluster = "None"
        transc, size = feature_inter.displaySamples(cluster,basket)
        st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket, size))
        if size >1:
            with col34:
                responses = st.radio("",
                                 ['All', 'Only responsive samples', "Only non-responsive samples"],
                                 key="responsive")
            with col35:
                n_features = st.number_input('Number of features', value=5)
            transc = feature_inter.filterSamples(transc, responses)
            sample = st.selectbox("Select a sample", transc, key="sample")
            if local_model == "LIME":
                st.subheader("LIME for individual predictions")
                st.write(" ")
                st.write("Local interpretable model-agnostic explanations (LIME) is a technique to explain individual predictions "
                         "made by a ML model. For this, an explanation is obtained by approximating the main model with a more interpretable one."
                         " The interpretable model is trained on small perturbations of the original observation to provide a good local approximation."
                         "For a selected sample, LIME results will show the features that have the strongest (positive or negative) impact on the "
                         "sample prediction."
                         )
                RawD = st.checkbox("Show raw data", key="raw-data-LIME")
                feature_inter.limeInterpretation(sample,n_features,RawD)
            elif local_model == "SHAP":
                st.write("  ")
                st.subheader("SHAP")
                st.write(" ")
                st.write("SHAP (SHapley Additive exPlanations) is a technique that explains the prediction of an observation by "
                         "computing the contribution of each feature to the prediction. It is based on Shapley values from game theory,"
                         " as it uses fair allocation results from cooperative game to allocate credit for a model's output among its input features."
                         )

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
    elif menu == "Features interaction":
        st.subheader("Features interaction")

        st.write("Sometimes is useful to investigate whether features interact between them.")
        feature_inter.SHAP_interact(explainer)