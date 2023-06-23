import streamlit as st
from processing import readPickle, Results, defaultPlot_leg, Analysis, dim_PCA, heatMap
from mpld3 import plugins
from mpld3.utils import get_id
import mpld3
import collections
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
from interpretation import Prototypes, DEA
from common import add_logo,hideRows

add_logo()
hide_rows = hideRows()
st.header("Basket*Cluster interaction")
#fig_html = mpld3.fig_to_html(fig)


def saveRawDInt(tab, c, b):
    if st.button('Save Data', key="df-Inter"):  # Update the key to a unique value
        tab.to_csv('heatmap_'+str(c)+'_'+b+'.csv')
        st.info('Data saved as .csv in working directory', icon="ℹ️")
    else:
        st.write("")

def saveheatmap(df,c,b):
    if st.button('Save plot', key="heatmapFull_png"):  # Update the key to a unique value
        df.savefig('heatmap_'+str(c)+'_'+b+'.png')
        st.info('Data saved as .png file in working directory', icon="ℹ️")
    else:
        st.write("")

def saveheatmap_transpt(df,c,b):
    if st.button('Save plot', key="heatmaptranspt_png"):  # Update the key to a unique value
        df.savefig('heatmap_'+str(c)+'_'+b+'.png')
        st.info('Data saved as .png file in working directory', icon="ℹ️")
    else:
        st.write("")
    #st.pyplot(fig)

if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = Analysis(data)
    heatmap = heatMap(data)
    st.subheader("Explore interactions")
    col11, col12 = st.columns((2, 2))
    with col11:
        cluster = st.selectbox("Select a cluster", data.setClusters(),key="cluster")
    with col12:
        basket = st.selectbox("Select a basket/tissue", data.setBaskets(),key="basket")
    subgroup, size = analysis_data.findInteraction(cluster, basket)
    st.text("")
    st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket,size))
    st.text("")
    tab1, tab2,tab3,tab4= st.tabs(["Interactions","PCA","Prototypes", "Differential Expression"])
    with tab1:
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
            saveheatmap(HM_samples, cluster, basket)
            st.pyplot(HM_samples)
        elif variable == 'Number of responsive samples':
            response_df = heatmap.heatmapResponse(data)
            HM_response = heatmap.heatmap_interaction(data, response_df, "Responsive samples per interaction",min_num,
                                                            int(cluster), basket)
            saveheatmap(HM_response, cluster, basket)
            st.pyplot(HM_response)
        else:
            inferred_df = heatmap.HM_inferredProb(data)
            HM_inferred = heatmap.heatmap_interaction(data,inferred_df,"Inferred basket*cluster interaction",min_num,int(cluster), basket)
            saveheatmap(HM_inferred, cluster, basket)
            st.pyplot(HM_inferred)
        try:
            st.write("#### Response to drug")
            st.markdown("Number of samples in **cluster {}** & **basket {}**: {} ".format(cluster, basket, size))
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
                saveRawDInt(subgroup,cluster,basket)
                st.dataframe(subgroup, use_container_width=True)
            else:
                heatmap2= heatmap.heatmapTranscripts(subgroup)
                saveheatmap_transpt(heatmap2,cluster,basket)
                st.pyplot(heatmap2)
        except:
            st.warning("Not enough samples. Please try a different combination.")

    with tab2:
        st.subheader("Advanced PCA")
        #st.write("##### PCA of samples in **cluster {}** & **basket {}**".format(cluster, basket))
        RawD = st.checkbox("Show raw data", key="raw-data")
        pca = dim_PCA(data)
        pca.adv_PCA(subgroup,RawD)
    with tab3:
        st.subheader("Prototypes of subgroup")
        try:
            sub_prototypes = Prototypes(data)
            sub_prototypes.findPrototypes_sub(subgroup)
        except:
            st.warning("Not enough samples. Please try a different combination.")
    with tab4:
        st.subheader("Differential expression analysis (DEA)")
        option = st.selectbox("Select analysis", ("Samples in interaction vs rest of samples", "Within interaction: responsive vs non-responsive"), key="DEA")
        if option == "Samples in interaction vs rest of samples":
            st.write("##### Samples in interaction vs all other interactions")
            col41, col42 = st.columns((2,2))
            with col41:
                pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value=0.05)
                logthresh = st.number_input('log2 FC threshold for significance (1 by default)', value=1.0)
            #st.write("##### Samples in **cluster {}** & **basket {}**".format(cluster, basket))
            if subgroup.size > 0:
                dea = DEA(data)
                dea.diffAnalysis_inter(subgroup,pthresh,logthresh)
                dea.showResults("interaction")
            else:
                st.warning("Not enough samples. Please try a different combination.")
            with col42:
                st.write(" ")
                st.write(" ")
                try:
                    dea.infoTest((cluster,basket), 'All', 'Interaction', pthresh,logthresh)
                except:
                    st.write(" ")
        else:
            st.write("##### Responsive vs non-responsive samples within interaction")
            col41, col42 = st.columns((2, 3))
            with col41:
                pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value=0.05)
                logthresh = st.number_input('log2 FC threshold for significance (1 by default)', value=1.0)
            if subgroup.size > 0:
                dea = DEA(data)
                dea.diffAnalysis_response(subgroup, pthresh, logthresh)
                dea.showResults("interaction")
            else:
                st.warning("Not enough samples. Please try a different combination.")
            with col42:
                st.write(" ")
                st.write(" ")
                try:
                    dea.infoTest("responsive", "non-responsive", 'Interaction', pthresh, logthresh)
                except:
                    st.write(" ")







