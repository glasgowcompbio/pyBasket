import matplotlib.pyplot as plt
import streamlit as st
from processing import readPickle, Results, Analysis, dim_PCA, heatMap
from common import add_logo, hideRows, savePlot,sideBar, openGeneCard
from interpretation import Prototypes, Kmedoids,DEA
from streamlit_option_menu import option_menu

hide_rows = hideRows()
add_logo()
sideBar()
st.header("Data exploration")
st.write("---")
menu = option_menu(None, ["Samples information", "pyBasket results","Statistics"],
    icons=["bi bi-clipboard-data", "bi bi-graph-up"],
    menu_icon="cast", default_index=0, orientation="horizontal")

if "data" in st.session_state:
    data = st.session_state["data"]
    heatmap = heatMap(data)
    if menu == "Samples information":
        st.subheader("Number of samples")
        st.write(
            "The number of samples is shown by cluster or basket/tissue. The number of responsive and non-responsive"
            " samples within groups can also be explored.")
        st.write(" ")
        col11, col12 = st.columns((1, 1))
        with col11:
            option_tab1 = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'),
                                       key="option-tab1")
        with col12:
            st.write(" ")
            RD = st.checkbox("Group by responsiveness", key="responsive")
            RawD = st.checkbox("Show raw data", key="raw-data")
        if option_tab1 == "Clusters":
            data.displayNums("cluster_number", "Cluster number", RD, RawD, "Samples per cluster")
        elif option_tab1 == 'Baskets/Tissues':
            data.displayNums("tissues", "Tissue", RD, RawD, "Samples per tissue")
        st.write("---")
        st.subheader("AAC response")
        st.write(
            "The Area Above the Curve (AAC) is a measure used to analyse drug response and quantify the effect of"
            " a drug over a period of time. In the context of he GDSC dataset, the AAC is the measure of the cell"
            " or cell line overall survival in response to the drug: the larger the AAC, the more resistance to the"
            " drug is shown."
            " "
            "The AAC values per cluster o basket/tissue is shown, as well as within these for Responsive and Non-"
            "responsive samples.")
        st.write(" ")
        col21, col22 = st.columns((1, 1))
        with col21:
            option_tab2 = st.selectbox("Select how to group samples", ('None', 'Clusters', 'Baskets/Tissues'),
                                       key="option-tab2")
        with col22:
            st.write(" ")
            st.write(" ")
            RawD_AAC = st.checkbox("Show raw data", key="raw-data-AAC")
        if option_tab2 == 'None':
            with col21:
                option_tab3 = st.selectbox("Show subgroups", ('None', 'Clusters', 'Baskets/Tissues'),
                                           key="option-tab3")
            if option_tab3 == "None":
                data.non_group_plot(None, RawD_AAC)
            elif option_tab3 == "Clusters":
                data.non_group_plot("cluster_number", RawD_AAC)
            elif option_tab3 == 'Baskets/Tissues':
                data.non_group_plot("tissues", RawD_AAC)
        else:
            with col22:
                RD_AAC = st.checkbox("Group by responsiveness", key="responsive-AAC")
            if option_tab2 == "Clusters":
                data.displayAAC("cluster_number", "Cluster number", RD_AAC, RawD_AAC, "AAC response per cluster")
            elif option_tab2 == 'Baskets/Tissues':
                data.displayAAC("tissues", "Tissue", RD_AAC, RawD_AAC, "AAC response per tissue")
    elif menu == "pyBasket results":
        heatmap.barInferredProb(data)
    elif menu == "Statistics":
        tab21, tab22, tab23 = st.tabs(["Dimensionality reduction", "Prototypes", "Differential expression"])
        with tab21:
            analysis_data = Analysis(data)
            pca = dim_PCA(data)
            st.subheader("Dimensionality reduction")
            st.write("The goal of dimensionality reduction techniques is to project high-dimensionality data to a lower "
                     "dimensional subspace while preserving the essence of the data and the maximum amount of information.")
            st.write("Principal Component Analysis (PCA) is a dimensionality reduction method that enables the visualisation"
                     " of high-dimensional data. The results for PCA on the data can be explored for the data grouped by "
                     "clusters, baskets/tissues or responsiveness.")
            st.write(" ")
            col31, col32 = st.columns((2,2))
            with col31:
                technique = st.selectbox("Choose a Dimensionality Reduction technique", ('PCA', 't-SNE'), key="technique")
                option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues', 'Responsive'), key="PCA")
                RawD = st.checkbox("Show raw data", key="raw-data-PCA")
            with col32:
                st.write(" ")
                st.write(" ")
                pca.infoPCA(option)
            if option == "Clusters":
                choices = analysis_data.patient_df["cluster_number"].unique()
                pca.PCA_analysis("cluster_number", RawD)
            elif option == "Responsive":
                pca.PCA_analysis("responsive", RawD)
                choices = analysis_data.patient_df["responsive"].unique()
            elif option == 'Baskets/Tissues':
                pca.PCA_analysis("tissues", RawD)
                choices = analysis_data.patient_df["tissues"].unique()
        with tab22:
            st.subheader("Prototypes")
            st.write("The prototypical sample of each cluster, basket/tissue or pattern of response has been calculated using"
                     " KMedoids. KMedoids is a partitioning technique that finds the sample (medoid or prototype)"
                    " that is the closest to the rest of samples in the same group.")
            st.write(" ")
            option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'), key="Prototypes")
            prototype = Prototypes(data)
            prototype.findPrototypes(option)
            #Kmedoids()

        with tab23:
            st.subheader("Differential expression")
            st.write(" ")
            st.write("The goal of Differential Expression Analysis (DEA) is to discover whether the expression "
                     "level of a feature (gene or transcript) is quantitatively different between experimental "
                     "groups or conditions.")
            st.write("T-test for the means of each feature in two independent groups or conditions is calculated."
                     "The null hypothesis is that the feature has identical average values across conditions.")
            st.write(" ")
            col51, col52 = st.columns((2, 2))
            with col51:
                option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues', 'Responsive'), key="DEA")
                if option == 'Clusters':
                    subgroups = data.setClusters()
                    feature = "cluster_number"
                elif option == 'Baskets/Tissues':
                    subgroups = data.setBaskets()
                    feature = "tissues"
                else:
                    subgroups = ['0','1']
                    feature = "responsive"
                groups = st.multiselect(
                    'Please select up to 2 Groups/Baskets to compare', subgroups, max_selections=2)
            if len(groups)<2:
                st.write("")
            else:
                #st.write("Groups {} and {} have been chosen for Differential Expression Analysis".format(groups[0],groups[1]))
                with col51:
                    pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value = 0.05)
                    logthresh = st.number_input('log2 FC threshold for significance (1 by default)', value=1.0)
                dea = DEA(data)
                dea.diffAnalysis_simple(groups[0],groups[1],feature,pthresh,logthresh)
                results = dea.showResults(feature)
                with col52:
                    st.write(" ")
                    st.write(" ")
                    dea.infoTest(groups[0],groups[1],option,pthresh,logthresh)
                st.subheader("Single Transcript DEA")
                st.write("")
                col53, col54 = st.columns((2,4))
                with col53:
                    transcript = st.selectbox("Select transcript", results["Feature"], key="transcript")
                    dea.infoTranscript(transcript)
                    st.write(" ")
                    st.write("Click button to search for feature {} in GeneCards database.".format(transcript))
                    st.button('Open GeneCards',on_click=openGeneCard, args=(transcript,))
                base = dea.boxplot(groups[0],groups[1],feature,transcript)
                with col54:
                    savePlot(base , "DEA" + transcript)
                    st.altair_chart(base, theme="streamlit", use_container_width=True)















