import streamlit as st
from importance import FI, Global
from common import add_logo, hideRows, saveTable, savePlot,sideBar,openGeneCard,searchTranscripts
from streamlit_option_menu import option_menu

add_logo()
sideBar()
hide_rows = hideRows()
st.header("Features importance")

st.write("---")
menu = option_menu(None, ["Overview", "Global MA methods", "Local MA methods"],
    icons=["bi bi-bar-chart", "bi bi-globe", "bi bi-pin-map"],
    menu_icon="cast", default_index=0, orientation="horizontal")

if "data" in st.session_state:
    data = st.session_state["data"]
    if "basket" in st.session_state:
        basket = st.session_state["basket"]
    if "cluster" in st.session_state:
        cluster = st.session_state["cluster"]
    analysis_data = st.session_state["Analysis"]
    feature_inter = FI(data)
    explainer, values = feature_inter.SHAP()
    if menu == "Overview":
        st.write(" ")
        st.subheader("Overview")
        st.write('Feature Importance is a technique to describe how relevant an input feature is and its effect on the model '
                 'being used to predict an outcome. Here, the feature importance is calculated using several Model-Agnostic (MA) '
                 'methods to find the transcripts that are mostly driving the prediction of the AAC response for each sample or samples. '
                 )
        col11, col12 = st.columns((3,2))
        with col11:
            global_model = st.selectbox("Select a method", ["RF feature importance", "SHAP", "Permutation Based"], key="model1")
        with col12:
            num_feat = st.number_input('Select top number of features to display (<30 is encouraged)', value=10)
        st.write("---")
        if global_model == "RF feature importance":
            st.subheader("Random Forest")
            st.write("Ramdom Forest is a common ML model that combines the output of multiple decision trees to reach a single result. It has been used "
                     "as part of the pyBasket pipeline to select the 500 most important features. Features' importance has been ranked based on the decrease in the impurity measure. "
                     "Below is shown the top {} features that are most important, i.e. their inclusion in a Random Forest will lead to a significant decrease in impurity. "
                     "Features are ranked based on their importance, higher values indicate greater importance. ".format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-RF")
            feature_inter.plotImportance(RawD,num_feat)
            st.caption("The x-axis represents the feature's importance (decrease in impurity measure).  The y-axis represents the features with the highest importance. ")
        elif global_model == "SHAP":
            st.subheader("SHAP values")
            st.write("SHAP (SHapley Additive exPlanations) is a technique that explains the prediction of an observation by "
                         "computing the contribution of each feature to the prediction. It is based on Shapley values from game theory,"
                         " as it uses fair allocation results from cooperative game to quantify the contribution that each feature makes to the model's prediction."
                         "Below, the top {} features that most influence the prediction of the AAC response to the drug are shown".format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-SHAP")
            if RawD:
                raw_df = feature_inter.SHAP_results(values)
                saveTable(raw_df, "SHAP")
                st.dataframe(raw_df, use_container_width=True)
            else:
                fig = feature_inter.SHAP_summary(values, num_feat)
                savePlot(fig, "Global_SHAP")
                st.pyplot(fig)
                st.caption("The x-axis represents the magnitude of the feature's impact on the model's output."
                           " The y-axis shows the features with the highest impact ordered in decreasing order."
                           "Each point is a Shapley value of an instance per feature/transcript and the colour represents the feature's value.")
        elif global_model == "Permutation Based":
            st.subheader("Permutation based feature importance")
            st.write('Permutation based feature importance is a MA method that measures'
                     ' the importance of a feature by calculating the increase in the model’s prediction '
                     'error when re-shuffling each predictor. '
                     ''
                     'Below is shown how much impact have the top {} features have in the model’s prediction for AAC response.'.format(num_feat))
            RawD = st.checkbox("Show raw data", key="rd-PBI")
            feature_inter.permutationImportance(num_feat,RawD)
            st.caption("The x-axis represents the feature's importance for the model's prediction. The y-axis represents the features ordered in decreasing order"
                       " of importance.")
    elif menu == "Global MA methods":
        st.write(" ")
        st.subheader("Global MA methods")
        st.write(" ")
        st.write('Global Model-Agnostic (MA) methods are used to describe the average behaviour of a Machine Learning model. Here, the transcripts that have the highest'
                 ' impact on the AAC response prediction for a group of cell lines on average can be explored.  ')
        method = st.selectbox("Select a global method", ['ALE', 'PDP (SHAP)'], key ='global')
        st.write("---")
        if method == 'ALE':
            st.write("#### Accumulated Local Effects")
            st.write(" ")
            st.write("Accumulated Local Effects (ALE) describe how a feature/transcript influences the prediction made by the ML "
                     "model on average. Here, this method has been implemented so that the behaviour of a feature can be explored for all samples or "
                     "for a group of samples with specified conditions, such as only samples that fall into a selected basket*cluster interaction, a specific cluster or basket/trial."
                     "In addition, the impact of a feature in two different groups of samples, i.e. two clusters, can also be compared. ")
            st.write(
                "The resulting ALE plot shows how the model's predictions change as the feature's values move across different bins.")
            global_ALE = Global(data)
            gsamples = st.radio("", ['All samples','Use samples in the selected interaction', 'Select only samples in cluster',
                                     'Select only samples in tissue/basket'],
                                key="global_samples", horizontal=True)
            transcripts = global_ALE.transcripts
            feature = searchTranscripts(transcripts)
            if gsamples == 'All samples':
                response = st.checkbox("Split by Responsive vs Non-responsive samples")
                if response:
                    global_ALE.global_ALE_resp(feature)
                else:
                    global_ALE.global_ALE(feature)
            elif gsamples == 'Use samples in the selected interaction':
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
                        'Please select a cluster or up to 2 clusters to compare.', data.clusters_names, max_selections=2)
                    option = "clusters"
                elif gsamples == 'Select only samples in tissue/basket':
                    groups = st.multiselect(
                        'Please select a tissue or up to 2 tissues to compare.',data.baskets_names, max_selections=2)
                    option = "baskets"
                try:
                    if len(groups)<2:
                        global_ALE.global_ALE_single(feature, groups[0],None,option)
                    else:
                        global_ALE.global_ALE_mult(feature, groups[0], groups[1], option)
                except:
                    st.warning(
                        "Please select at least a group.")

        elif method == 'PDP (SHAP)':
            st.write("#### Partial Dependence Plot (PDP) by SHAP")
            st.write("The partial dependence plot (PDP) calculated by SHAP shows "
                     "the marginal effect that one or two features, that might interact with each other,"
                     " have on the predictions made by the ML model (the predicted AAC response to the drug).  "
                     )
            features = feature_inter.SHAP_results(values)
            transcript2 = st.selectbox("Select feature/transcript", features['Transcript'], key="transcript2")
            st.caption("Values ordered by decreasing importance by SHAP")
            st.write("#### SHAP dependence plot")
            feature_inter.SHAP_dependence(values, transcript2)
    elif menu == "Local MA methods":
        st.subheader("Local MA methods")
        st.write(" ")
        st.write("Local Model-Agnostic (MA) interpretation methods aim to explain individual predictions made by a Machine Learning model. The transcripts that have the highest impact"
                 " on the prediction of the AAC response of individual cell lines can be explored. ")
        col31, col32 = st.columns((2, 2))
        with col31:
            local_model = st.selectbox("Select a local interpretable method", ["LIME", "SHAP"], key="model2")
        col33, col34, col35 = st.columns((2,2,2))
        with col33:
            group_samples = st.radio("", ['Use samples in interaction', 'Select only samples in cluster',
                                      'Select only samples in tissue/basket'],
                                 key="samples")
        if group_samples == 'Use samples in selection':
            basket, cluster = basket, cluster
        elif group_samples == 'Select only samples in cluster':
            with col35:
                cluster= st.selectbox("Select a cluster", data.clusters_names, key="only_cluster")
            basket = "None"
        elif group_samples == 'Select only samples in tissue/basket':
            with col35:
                basket = st.selectbox("Select a basket/tissue", data.baskets_names, key="only_basket")
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
            st.write("---")
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
                limedf = feature_inter.limeInterpretation(sample,n_features,RawD)
                st.caption("The x-axis represents the feature's importance on the model's prediction. The y-axis represents the features with highest importance in decreasing order."
                    "Green values: positive effect on prediction. Red values: negative effect on prediction. ")
                st.write(" ")
                st.write("##### Further information about the features displayed: ")
                feature = searchTranscripts(limedf['Feature'].tolist())
            elif local_model == "SHAP":
                st.write("  ")
                st.subheader("SHAP")
                st.write(" ")
                st.write("SHAP (SHapley Additive exPlanations) is a technique that explains the prediction of an observation by "
                         "computing the contribution of each feature to the prediction. It is based on Shapley values from game theory,"
                         " as it uses fair allocation results from cooperative game to allocate credit for a model's output among its input features."
                         )
                st.write("  ")
                st.write("##### Model's prediction")
                pred, true_value = feature_inter.SHAP_pred(sample)
                st.write("**Model's predicted AAC response is:** {} ".format(pred))
                st.write("**True AAC response is:** {} ".format(true_value))
                st.write(" ")
                st.write("##### Decision plot for {}".format(sample))
                st.write(
                    "The SHAP decision plot shows the relationship between the {} most influential features/transcripts and the model's prediction for the "
                    "individual cell line {}. "
                    "They are a linear representation of SHAP values.".format(n_features, sample))
                st.write("")
                RawD = st.checkbox("Show raw data", key="raw-data-SHAP-dec")
                transcripts = feature_inter.SHAP_decision(sample, explainer, values, n_features, RawD)
                st.caption(
                    "The x-axis represents the model's AAC prediction. The y-axis represents the model's features. "
                    "The grey vertical line represents the base value. Coloured line is the prediction and how each feature impacts it."
                    " Bracket values are the real features values for the chosen sample.")
                st.write("##### Further information about the features displayed: ")
                feature = searchTranscripts(transcripts)

