import streamlit as st
from analysis import Analysis,heatMap
from common import add_logo, hideRows, savePlot,sideBar, openGeneCard,searchTranscripts
from interpretation import Prototypes, DEA
from streamlit_option_menu import option_menu

hide_rows = hideRows()
add_logo()
sideBar()
st.header("Data exploration")
st.write("---")
menu = option_menu(None, ["Samples information", "pyBasket results","Statistics"],
    icons=["bi bi-clipboard-data", "bi bi-basket","bi bi-graph-up"],
    menu_icon="cast", default_index=0, orientation="horizontal")

if "data" in st.session_state:
    data = st.session_state["data"]
    heatmap = heatMap(st.session_state["saved data"],st.session_state["File Name"])
    analysis = st.session_state["Analysis"]
    #Subpage showing information about the samples: Number and AAC response
    if menu == "Samples information":
        st.subheader("Number of samples")
        st.write(
            "The number of samples can be shown either by cluster or basket/tissue. The number of Responsive and Non-responsive"
            " samples within groups can also be explored. ")
        st.write(" ")
        col11, col12 = st.columns((1, 1))
        with col11:
            #Option to select level of grouping: clusters or baskets
            option_tab1 = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'),
                                       key="option-tab1")
        with col12:
            st.write(" ")
            #Option to group samples by response within groups
            RD = st.checkbox("Group by response", key="responsive")
            #Option to show data in a table format
            RawD = st.checkbox("Show raw data", key="raw-data")
        if option_tab1 == "Clusters":
            if RawD:
                data.showRawData("cluster_number","Cluster number", RD)
            else:
                data.countPlot(RD,"Number of samples","cluster_number","Cluster number")
        elif option_tab1 == 'Baskets/Tissues':
            if RawD:
                data.showRawData("tissues", "Tissue/Basket", RD)
            else:
                data.countPlot(RD, "Number of samples per tissue", "tissues", "Tissues")
        st.write("---")
        st.subheader("AAC response")
        st.write(
            "The Area Above the Curve (AAC) is a measure used to analyse drug response and quantify the effect of"
            " a drug over a period of time. In the context of the GDSC dataset, the AAC is the measure of the cell"
            " or cell line overall survival in response to the drug: the larger the AAC, the more resistance to the"
            " drug is shown."
            " "
            "The AAC values per cluster o basket/tissue is shown, as well as within these for Responsive and Non-"
            "responsive samples.")
        st.write(" ")
        col21, col22 = st.columns((1, 1))
        with col21:
            # Option to select level of colouring: none, clusters or baskets
            option_tab2 = st.selectbox("Select how to group samples", ('None', 'Clusters', 'Baskets/Tissues'),
                                       key="option-tab2")
        with col22:
            st.write(" ")
            st.write(" ")
            #Option to show data in a table format
            RawD_AAC = st.checkbox("Show raw data", key="raw-data-AAC")
        if option_tab2 == 'None':
            data.AAC_scatterplot(RawD_AAC)
        else:
            with col22:
                # Option to group samples by response within groups
                RD_AAC = st.checkbox("Group by response", key="responsive-AAC")
            if option_tab2 == "Clusters":
                data.AAC_response("cluster_number", RD_AAC, "Cluster number", RawD_AAC)
            elif option_tab2 == 'Baskets/Tissues':
                data.AAC_response("tissues", RD_AAC, "Tissue/Basket", RawD_AAC)
    elif menu == "pyBasket results":
        # Option to select level of grouping: clusters or baskets
        option_page2 = st.selectbox("Select group", ('Clusters', 'Baskets/Tissues'),
                                    key="option-page2")
        st.subheader("Inferred response (mean) probability")
        st.write("In the second stage of the pyBasket pipeline, a Hierarchical Bayesian model is used to estimate the"
                 " overall mean probability of each basket or cluster to be responsive to the treatment.")
        # Option to show data in a table format
        RawD_prob = st.checkbox("Show raw data", key="raw-data-prob")
        if option_page2 == 'Baskets/Tissues':
            data.barInferredProb("baskets",RawD_prob)
        elif option_page2 == "Clusters":
            data.barInferredProb("clusters",RawD_prob)
        st.subheader("Empirical Cumulative Distribution")
        st.write("The Hierarchical Bayesian model returns a single (mean) value rather than the real distribution. The Empirical Cumulative "
                 "Distribution Function (ECDF) provides an empirical approximation of the underlying distribution of the data and assigns "
                 "probabilities taking into account the number of samples. ")
        st.write("The ECDF is constructed by arranging the observed data points in ascending order and assigning a cumulative probability to each point. "
                 "The cumulative probability for a sample is calculated as the fraction of data points that are less than or equal to that value."
                 " The credible interval can be chosen below, which is the range containing a particular percentage of probable values.")
        col1, col2 = st.columns((2, 2))
        with col2:
            #User input for credible interval
            cred_inter = st.number_input('Credible interval', value = 90)
            st.caption("90% credible interval shown by default")
        if option_page2 == "Clusters":
            with col1:
                #Option to select cluster to show ECDF
                clusterA = st.selectbox("Select cluster", data.clusters_names)
            #Option to show results in table format
            RawD_ecdf = st.checkbox("Show raw data", key="raw-data-ecdf")
            cluster_choice = data.clusters_names.index(clusterA)
            data.ecdf_indiv("clusters",clusterA,cluster_choice,RawD_ecdf,cred_inter)
        elif option_page2 == 'Baskets/Tissues':
            with col1:
                # Option to select basket/tissue to show ECDF
                basketA = st.selectbox("Select basket", data.baskets_names)
            # Option to show results in table format
            RawD_ecdf = st.checkbox("Show raw data", key="raw-data-ecdf")
            basket_choice = data.baskets_names.index(basketA)
            data.ecdf_indiv("baskets",basketA, basket_choice,RawD_ecdf,cred_inter)
    elif menu == "Statistics":
        tab21, tab22, tab23 = st.tabs(["Dimensionality reduction", "Prototypes", "Differential expression"])
        #Dimensionality reduction subpage (PCA)
        with tab21:
            st.subheader("Dimensionality reduction")
            st.write("The goal of dimensionality reduction is to project high-dimensionality data to a lower "
                     "dimensional subspace while preserving the essence of the data and the maximum amount of information.")
            st.write("Principal Component Analysis (PCA) is a dimensionality reduction method that enables the visualisation"
                     " of high-dimensional data capturing as much as possible of its variation.  "
                     "The results for PCA on the data can be explored for the data grouped by "
                     "clusters, baskets/tissues or response.")
            st.write(" ")
            # Option to select level of grouping: clusters, baskets or response
            option_PCA = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues', 'Responsive'), key="PCA")
            #Option to show data in table format
            RawD = st.checkbox("Show raw data", key="raw-data-PCA")
            if option_PCA == "Clusters":
                analysis.PCA_analysis("cluster_number", RawD)
            elif option_PCA == "Responsive":
                analysis.PCA_analysis("responsive", RawD)
            elif option_PCA == 'Baskets/Tissues':
                analysis.PCA_analysis("tissues", RawD)
        #Prototypes samples subpage
        with tab22:
            st.subheader("Prototypes")
            st.write("The prototypical sample of each cluster, basket/tissue or pattern of response has been calculated using"
                     " KMedoids. KMedoids is a partitioning technique that finds the sample (medoid or prototype)"
                    " that is the closest to the rest of samples in the same group. The dimension of the expression level of transcripts"
                     " has been reduced and plotted using Principal Component Analysis (PCA). ")
            st.write(" ")
            #Option to select level of grouping: clusters, baskets or response
            option_prototype = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'), key="Prototypes")
            prototype = Prototypes(data)
            prototype.findPrototypes(option_prototype)
        #Differential expression analysis subpage
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
                # Option to select level of grouping: clusters, baskets or response
                option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues', 'Responsive'), key="DEA")
                if option == 'Clusters':
                    subgroups = data.clusters_names
                    feature = "cluster_number"
                elif option == 'Baskets/Tissues':
                    subgroups = data.baskets_names
                    feature = "tissues"
                else:
                    subgroups = ["Responsive","Non-responsive"]
                    feature = "responsive"
                #Option to select groups to perform DEA
                groups = st.multiselect(
                    'Please select up to 2 Groups to compare', subgroups, max_selections=2)
            if len(groups)<2:
                st.write("")
            else:
                with col51:
                    #User input for corrected p-value threshold
                    pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value = 0.05)
                    st.caption("Applied on corrected p-values.")
                    #User input for log2 fold change threshold
                    logthresh = st.number_input('log2 FC threshold for significance (1 by default)', value=1.0)
                dea = DEA(data)
                dea.diffAnalysis_simple(groups[0],groups[1],feature,pthresh,logthresh)
                results = dea.showResults(feature)
                with col52:
                    st.write(" ")
                    st.write(" ")
                    dea.infoTest(groups[0],groups[1],option,pthresh,logthresh)
                st.subheader("Single Transcript DEA")
                st.write("The difference in the expression level of a selected transcript between groups being compared can be explored below. "
                         "Further information about the transcript can be found by using the GeneCards button. ")
                st.write("")
                #Option to select transcript results to be displayed
                transcript = searchTranscripts(results["Feature"])
                col53, col54 = st.columns((2,4))
                with col53:
                    st.write(" ")
                    dea.infoTranscript(transcript)
                with col54:
                    dea.boxplot(groups[0], groups[1], feature, transcript)
















