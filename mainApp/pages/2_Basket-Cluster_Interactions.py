import streamlit as st
from analysis import Analysis, heatMap
from interpretation import Prototypes, DEA
from common import add_logo,hideRows,savePlot,sideBar, saveTable, openGeneCard,savePlot_plt,searchTranscripts
from streamlit_option_menu import option_menu

add_logo()
sideBar()
hide_rows = hideRows()
st.header("Basket*Cluster interactions")
st.write("---")
menu = option_menu(None, ["All interactions", "Selected interaction"],
    icons=["bi bi-asterisk", "bi bi-braces-asterisk"],
    menu_icon="cast", default_index=0, orientation="horizontal")

if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = st.session_state["Analysis"]
    heatmap = heatMap(st.session_state["saved data"],st.session_state["File Name"])
    if "basket" in st.session_state:
        basket = st.session_state["basket"]
    if "cluster" in st.session_state:
        cluster = st.session_state["cluster"]
    subgroup, size = analysis_data.findInteraction(cluster, basket)
    if menu == "All interactions":
        st.write("""This page shows results and information about the interactions between clusters and baskets/tissues. Below what information to 
                 display can be selected, such as the number of samples, the number of responsive samples or the inferred response in each basket*cluster interaction.""")
        variable = st.radio("Select information to display",
                            ['Number of samples', 'Number of responsive samples', "Inferred response"],
                            key="HM_info", horizontal=True)
        st.write("")

        min_num = st.slider('Mark interactions with a minimum number of samples', 0, 70)
        st.info("\:star: : basket+cluster interactions with at least {} samples.\n"

                "\:large_red_square: : selected basket+cluster interaction.\n".format(min_num))
        st.write("")
        if variable == 'Number of samples':
            st.write("#### Number of samples per basket*cluster interaction")
            st.write("Explore the number of samples in each basket and cluster combination.")
            RawD = st.checkbox("Show raw data", key="raw-data-S")
            num_samples = heatmap.heatmapNum()
            if RawD:
                saveTable(num_samples, "NumSamplesHM")
                st.dataframe(num_samples, use_container_width=True)
            else:
                fig = heatmap.heatmap_interaction(num_samples,"Number of samples per interaction"
                                                         , min_num, int(cluster), basket)
                savePlot_plt(fig, str(cluster) + "_" + basket)
                st.pyplot(fig)
                st.caption("The x-axis shows the levels of clusters. The y-axis shows the levels of baskets/tissues. The colour scale represents "
                           "the number of samples.")
        elif variable == 'Number of responsive samples':
            st.write("#### Number of samples per basket*cluster interaction responsive to the drug")
            st.write(
                "Explore the number of samples in each basket and cluster combination that are responsive to the drug.")
            RawD = st.checkbox("Show raw data", key="raw-data-R")
            response_df = heatmap.heatmapResponse()
            if RawD:
                saveTable(response_df, "responseHM")
                st.dataframe(response_df, use_container_width=True)
            else:
                HM_response = heatmap.heatmap_interaction(response_df,"Responsive samples per interaction", min_num,
                                                          int(cluster), basket)
                savePlot_plt(HM_response, str(cluster) + "_" + basket)
                st.pyplot(HM_response)
                st.caption(
                    "The x-axis shows the levels of clusters. The y-axis shows the levels of baskets/tissues. The colour scale represents "
                    "the number of samples that are responsive to the drug.")
        else:
            st.write("#### Inferred response probability per basket*cluster interaction.")
            st.write(
                "Explore the inferred response probability of each tissue/basket calculated by the HBM based on observed responses.")
            RawD = st.checkbox("Show raw data", key="raw-data-R")
            inferred_df = heatmap.HM_inferredProb()
            if RawD:
                saveTable(inferred_df,"inferredHM")
                st.dataframe(inferred_df, use_container_width=True)
            else:
                HM_inferred = heatmap.heatmap_interaction(inferred_df,"Inferred basket*cluster interaction", min_num,
                                                          int(cluster), basket)
                savePlot_plt(HM_inferred, "inferred_heatmap")
                st.pyplot(HM_inferred)
                st.caption(
                    "The x-axis shows the levels of clusters. The y-axis shows the levels of baskets/tissues. The colour scale represents "
                    "the inferred probability of response to the treatment by the basket*cluster interaction.")

    elif menu == "Selected interaction":
        st.text("")
        st.write("In this page, results specific to the selected basket*cluster interaction can be explored.")
        st.text("")
        st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket, size))
        st.text("")
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "PCA", "Prototypes", "Differential Expression"])
        with tab1:
            try:
                st.write("#### Response to drug")
                st.markdown("Number of samples in **cluster {}** & **basket {}**: {} ".format(str(cluster), basket, size))
                col21, col22 = st.columns((2, 2))
                with col21:
                    analysis_data.samplesCount(subgroup)
                with col22:
                    analysis_data.responseSamples(subgroup)
                    st.caption("Samples ordered from most to least responsive (lower AAC response)")
                st.write("#### ECDF")
                RawD_ecdf = st.checkbox("Show raw data", key="raw-data-ecdf")
                analysis_data.ecdf_interaction(basket, cluster, RawD_ecdf)
                st.write("#### Transcriptional expression")
                RawD = st.checkbox("Show raw data", key="raw-data-HM1")
                if RawD:
                    saveTable(subgroup, str(cluster) + "_" + basket)
                    st.dataframe(subgroup, use_container_width=True)
                else:
                    heatmap2 = heatmap.heatmapTranscripts(subgroup)
                    savePlot_plt(heatmap2, "_transcripts")
                    st.pyplot(heatmap2)
                    st.caption(
                        "The x-axis represents all transcripts. The y-axis represents some of the samples in the selected basket*cluster interaction."
                        " The colour bar represents the expression level of the transcript for each sample.")
            except:
                st.warning("Not enough samples. Please try a different combination.")

        with tab2:
            st.subheader("Advanced PCA")
            st.write("")
            st.write("Principal Component Analysis (PCA) is a dimensionality reduction method that enables the visualisation"
                " of high-dimensional data. The results for PCA on the data can be explored for the samples in the selected"
                     " basket*cluster interaction, that are grouped by responsiveness: Responsive vs Non-responsive.")
            st.write(" ")
            #st.write("##### PCA of samples in **cluster {}** & **basket {}**".format(cluster, basket))
            RawD = st.checkbox("Show raw data", key="raw-data")
            analysis_data.adv_PCA(subgroup,RawD)
        with tab3:
            st.subheader("Prototypes of subgroup")
            st.write("")
            st.write("The prototype sample of the selected basket*cluster interaction for the Responsive and the"
                    " Non-responsive groups has been calculated using KMedoids. KMedoids finds the sample that is"
                     "the closest to the rest of samples in the group. ")
            try:
                sub_prototypes = Prototypes(data)
                sub_prototypes.findPrototypes_sub(subgroup)
            except:
                st.warning("Not enough samples. Please try a different combination.")
        with tab4:
            st.subheader("Differential expression analysis (DEA)")
            st.write("")
            st.write("The goal of Differential Expression Analysis (DEA) is to discover whether the expression "
                     "level of a feature (gene or transcript) is quantitatively different between experimental "
                     "groups or conditions.")
            st.write("T-test for the means of each feature in two independent groups or conditions is calculated."
                     "The null hypothesis is that the feature has identical average values across conditions.")
            st.write(" ")
            try:
                dea = DEA(data)
            except:
                st.warning("Not enough samples in the basket*cluster interaction. Please select another combination.")
            col41, col42 = st.columns((2, 2))
            with col41:
                option = st.selectbox("Select analysis", (
                "Samples in interaction vs rest of samples", "Within interaction: responsive vs non-responsive"), key="DEA")
                pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value=0.05)
                st.caption("Applied on corrected p-values.")
                logthresh = st.number_input('log2 FC threshold for significance (1 by default)', value=1.0)
            if option == "Samples in interaction vs rest of samples":
                st.write(" ")
                st.subheader("Samples in interaction vs all other interactions")
                st.write("DEA performed with the samples in the selected basket*cluster interaction vs all other samples.")
                if subgroup.size > 0:
                    dea.diffAnalysis_inter(subgroup,pthresh,logthresh)
                    results = dea.showResults("interaction")
                else:
                    st.warning("Not enough samples. Please try a different combination.")
                with col42:
                    st.write(" ")
                    st.write(" ")
                    try:
                        dea.infoTest((cluster,basket), 'All', 'Interaction', pthresh,logthresh)
                    except:
                        st.write(" ")
                try:
                    st.subheader("Individual transcripts DEA")
                    st.write(
                        "The difference in the expression level of a transcript/feature and the individual results from "
                        "DEA can be explored below. Further information about the transcript can be found in the button that links to its GeneCard profile.")
                    transcript = searchTranscripts(results["Feature"])
                    col53, col54 = st.columns((2, 3))
                    with col53:
                        st.write(" ")
                        dea.infoTranscript(transcript)
                    with col54:
                        dea.boxplot_inter(subgroup, transcript)
                except:
                    st.warning("Not enough samples. Please try a different combination.")
            else:
                st.write(" ")
                st.subheader("Responsive vs non-responsive samples within basket*cluster interaction")
                st.write("DEA has been performed within samples in the selected interaction and comparing Responsive vs Non-responsive samples.")
                dea = DEA(data)
                dea.diffAnalysis_response(data,subgroup, pthresh, logthresh)
                if subgroup.size > 0:
                    try:
                        results = dea.showResults("interaction")
                        st.subheader("Individual transcripts DEA")
                        st.write(
                            "The difference in the expression level of a transcript/feature and the individual results from "
                            "DEA can be explored below. Further information about the transcript can be found in the button that links to its GeneCard profile.")
                        feature = searchTranscripts(results["Feature"])
                        col53, col54 = st.columns((2, 4))
                        with col53:
                            st.write(" ")
                            dea.infoTranscript(feature)
                        with col54:
                            dea.boxplot_resp(subgroup, feature)
                    except:
                        st.warning("Not enough samples. Please try a different combination.")
                else:
                    st.warning("Not enough samples. Please try a different combination.")
                with col42:
                    st.write(" ")
                    st.write(" ")
                    try:
                        dea.infoTest('Responsive', 'Non-responsive', (cluster,basket), pthresh,logthresh)
                    except:
                        st.write(" ")







