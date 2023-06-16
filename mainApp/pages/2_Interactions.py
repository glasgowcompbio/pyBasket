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
    st.subheader("Explore interactions")
    col11, col12 = st.columns((2, 2))
    with col11:
        #st.write("##### Select a cluster")

        cluster = st.selectbox("Select a cluster", data.setClusters(),
                               key="cluster")
    with col12:
        #st.write("##### Select a basket/tissue")

        basket = st.selectbox("Select a basket/tissue", data.setBaskets(),
                              key="basket")
    subgroup, size = analysis_data.findInteraction(cluster, basket)
    st.text("")
    st.write("##### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket,size))
    st.text("")
    tab1, tab2,tab3,tab4= st.tabs(["Interactions","PCA","Prototypes", "Differential Expression"])
    with tab1:
        variable = st.selectbox("Select information to display", ['Number of samples', 'Number of responsive samples'],
                                key="info")
        if variable == 'Number of samples':
            heatmap = analysis_data.heatmapNum(data, int(cluster), basket)
            saveheatmap(heatmap, cluster, basket)
            st.pyplot(heatmap)
        elif variable == 'Number of responsive samples':
            heat = analysis_data.heatmapResponse(data, int(cluster), basket)
            st.pyplot(heat)
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
                heatmap2= analysis_data.heatmapTranscripts(subgroup)
                saveheatmap_transpt(heatmap2,cluster,basket)
                st.pyplot(heatmap2)
        except:
            st.warning("Not enough samples. Please try a different combination.")

    with tab2:
        st.subheader("Advanced PCA")
        #st.write("##### PCA of samples in **cluster {}** & **basket {}**".format(cluster, basket))
        RawD = st.checkbox("Show raw data", key="raw-data")
        analysis_data.adv_PCA(subgroup,RawD)
    with tab3:
        st.subheader("Prototypes of subgroup")
        try:
            sub_prototypes = Prototypes(data)
            sub_prototypes.findPrototypes_sub(subgroup)
        except:
            st.warning("Not enough samples. Please try a different combination.")
    with tab4:
        st.subheader("Differential expression analysis")
        st.write("Differential Expression Analysis of transcripts for samples in interaction vs rest of samples")
        col41, col42 = st.columns((2,2))
        with col41:
            pthresh = st.number_input('P-value threshold for significance (0.05 by default)', value=0.05)
        #st.write("##### Samples in **cluster {}** & **basket {}**".format(cluster, basket))
        if subgroup.size > 0:
            dea = DEA(data)
            dea.diffAnalysis_inter(subgroup,pthresh)
        else:
            st.warning("Not enough samples. Please try a different combination.")
        with col42:
            st.write(" ")
            st.write(" ")
            dea.infoTest((cluster,basket), 'All', 'Interaction', pthresh)



