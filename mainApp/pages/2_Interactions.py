import streamlit as st
from processing import readPickle, Results, defaultPlot_leg, Analysis, dim_PCA, heatMap
from interpretation import Prototypes, DEA
from common import add_logo,hideRows,savePlot,sideBar, saveTable

add_logo()
sideBar()
hide_rows = hideRows()
st.header("Basket*Cluster interaction")
#fig_html = mpld3.fig_to_html(fig)

if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = st.session_state["analysis"]
    heatmap = heatMap(data)
    st.subheader("Explore interactions")
    if "basket" in st.session_state:
        basket = st.session_state["basket"]
    if "cluster" in st.session_state:
        cluster = st.session_state["cluster"]
    subgroup, size = analysis_data.findInteraction(cluster, basket)
    st.text("")
    st.write("Explore results from the pyBasket pipeline for basket*cluster interactions.")
    st.text("")
    st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket,size))
    st.text("")
    tab1, tab2,tab3,tab4= st.tabs(["Interactions","PCA","Prototypes", "Differential Expression"])
    with tab1:
        st.write("")

        st.write("")
        variable = st.radio("Select information to display", ['Number of samples', 'Number of responsive samples', "Inferred response"],
                            key="HM_info", horizontal=True)
        st.write("")

        min_num = st.slider('Mark interactions with minimum number of samples', 0,70)
        st.info("\:star: : basket+cluster interactions with at least {} samples.\n"
                
                "\:large_red_square: : selected basket+cluster interaction.\n".format(min_num))
        st.write("")
        if variable == 'Number of samples':
            num_samples = heatmap.heatmapNum(data)
            HM_samples = heatmap.heatmap_interaction(data, num_samples, "Number of samples per interaction"
                                                            ,min_num,int(cluster), basket)
            savePlot(HM_samples, str(cluster)+"_"+basket)
            st.pyplot(HM_samples)
        elif variable == 'Number of responsive samples':
            response_df = heatmap.heatmapResponse(data)
            HM_response = heatmap.heatmap_interaction(data, response_df, "Responsive samples per interaction",min_num,
                                                            int(cluster), basket)
            savePlot(HM_response, str(cluster)+"_"+basket)
            st.pyplot(HM_response)
        else:
            inferred_df = heatmap.HM_inferredProb(data)
            HM_inferred = heatmap.heatmap_interaction(data,inferred_df,"Inferred basket*cluster interaction",min_num,int(cluster), basket)
            savePlot(HM_inferred, str(cluster)+"_"+basket)
            st.pyplot(HM_inferred)
        try:
            st.write("#### Response to drug")
            st.markdown("Number of samples in **cluster {}** & **basket {}**: {} ".format(str(cluster), basket, size))
            col21, col22 = st.columns((2,2))
            with col21:
                    count =analysis_data.samplesCount(subgroup)
                    st.pyplot(count,use_container_width=False)
            with col22:
                df = analysis_data.responseSamples(subgroup)
                st.dataframe(df)
                st.caption("Samples ordered from most to least responsive (lower AAC response)")
            st.write("#### Transcriptional expression")
            RawD = st.checkbox("Show raw data", key="raw-data-HM1")
            if RawD:
                saveTable(subgroup,str(cluster)+"_"+basket)
                st.dataframe(subgroup, use_container_width=True)
            else:
                heatmap2= heatmap.heatmapTranscripts(subgroup)
                savePlot(heatmap2,"_transcripts")
                st.pyplot(heatmap2)
        except:
            st.warning("Not enough samples. Please try a different combination.")

    with tab2:
        st.subheader("Advanced PCA")
        st.write("")
        st.write("Principal Component Analysis (PCA) results for samples in the selected basket*cluster interaction."
                 "Will show whether there is a clear separation between samples depending on their behaviour against"
                "the treatment: Responsive vs Non-responsive.")
        #st.write("##### PCA of samples in **cluster {}** & **basket {}**".format(cluster, basket))
        RawD = st.checkbox("Show raw data", key="raw-data")
        pca = dim_PCA(data)
        pca.adv_PCA(subgroup,RawD)
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
        st.write("Differential Expression Analysis (DEA) shows whether the expression profile of transcripts is "
                "found to be significantly different between conditions being compared.")
        try:
            dea = DEA(data)
        except:
            st.warning("Not enough samples in the basket*cluster interaction. Please select another combination.")
        col41, col42 = st.columns((2, 2))
        with col41:
            option = st.selectbox("Select analysis", (
            "Samples in interaction vs rest of samples", "Within interaction: responsive vs non-responsive"), key="DEA")
            pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value=0.05)
            logthresh = st.number_input('log2 FC threshold for significance (1 by default)', value=1.0)

        if option == "Samples in interaction vs rest of samples":
            st.write("##### Samples in interaction vs all other interactions")
            st.write("DEA performed with the samples in the selected basket*cluster interaction against any other interaction")
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
                transcript = st.selectbox("Select transcript", results["Feature"], key="transcript")
                fig = dea.boxplot_inter(subgroup, transcript)
                savePlot(fig, "DEA" + transcript)
                st.pyplot(fig)
            except:
                st.warning("Not enough samples. Please try a different combination.")
        else:
            st.write("##### Responsive vs non-responsive samples within basket*cluster interaction")
            st.write("DEA has been performed within samples in the selected interaction and comparing Responsive vs Non-responsive samples.")
            if subgroup.size > 0:
                dea = DEA(data)
                dea.diffAnalysis_response(subgroup, pthresh, logthresh)
                results = dea.showResults("interaction")
                print(type(results))
                st.subheader("Individual transcripts DEA")
                transcript = st.selectbox("Select transcript", results["Feature"], key="transcript")
                fig = dea.boxplot_resp(subgroup, transcript)
                savePlot(fig, "DEA" + transcript)
                st.pyplot(fig)
            else:
                st.warning("Not enough samples. Please try a different combination.")
            with col42:
                st.write(" ")
                st.write(" ")
                try:
                    dea.infoTest((cluster,basket), 'All', 'Interaction', pthresh,logthresh)
                except:
                    st.write(" ")







