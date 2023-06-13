import streamlit as st
from processing import readPickle, Results, add_logo, defaultPlot_leg, Analysis, hideRows
from mpld3 import plugins
from mpld3.utils import get_id
import mpld3
import collections
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
from interpretation import Prototypes

add_logo()
hide_rows = hideRows()
st.header("Basket*Cluster interaction")
#fig_html = mpld3.fig_to_html(fig)


def savePlot_advPCA(fig):
    if st.button('Save Plot', key="plot_advPCA"):  # Update the key to a unique value
        fig.savefig('plot_PCA_.png')
        st.info('Plot saved as .png in working directory', icon="ℹ️")
    else:
        st.write("")

def saveRawDInt(tab):
    if st.button('Save Data', key="df-Inter"):  # Update the key to a unique value
        tab.to_csv('data_PCA.csv')
        st.info('Data saved as .csv in working directory', icon="ℹ️")
    else:
        st.write("")

def saveRawDInt_PCA(tab):
    if st.button('Save Data', key="df-PCA"):  # Update the key to a unique value
        tab.to_csv('data_PCA.csv')
        st.info('Data saved as .csv in working directory', icon="ℹ️")
    else:
        st.write("")

def saveRawDInt_varPCA(tab):
    if st.button('Save Data', key="df-varPCA"):  # Update the key to a unique value
        tab.to_csv('data_PCA.csv')
        st.info('Data saved as .csv in working directory', icon="ℹ️")
    else:
        st.write("")
def saveheatmap(df,c,b):
    if st.button('Save plot', key="heatmapFull_png"):  # Update the key to a unique value
        df.to_csv('raw_data_'+c+'_'+b+'.png', index=False)
        st.info('Data saved as .png file in working directory', icon="ℹ️")
    else:
        st.write("")

def saveheatmap_transpt(df,c,b):
    if st.button('Save plot', key="heatmaptranspt_png"):  # Update the key to a unique value
        df.to_csv('raw_data_'+c+'_'+b+'.png', index=False)
        st.info('Data saved as .png file in working directory', icon="ℹ️")
    else:
        st.write("")

def adv_PCA(sub_df, RawD):
    try:
        df, variance = analysis_data.advanced_PCA(sub_df)
        if RawD is True:
            pcaDF, var_df = analysis_data.showRawData_PCA(df, variance)
            col11, col12 = st.columns((2, 3))
            with col11:
                st.write('##### Variance explained by component')
                saveRawDInt_varPCA(var_df)
                st.dataframe(var_df, use_container_width=True)
            with col12:
                st.write('##### PCA results')
                saveRawDInt_PCA(pcaDF)
                st.dataframe(pcaDF, use_container_width=True)
        else:
            fig = analysis_data.plot_PCA(df, "responsive")
            savePlot_advPCA(fig)
            fig_html = mpld3.fig_to_html(fig)
            components.html(fig_html, height=600, width=3000)
    except:
        st.warning("Not enough samples. Please try a different combination.")
    #st.pyplot(fig)

if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = Analysis(data)
    tab1, tab2,tab3= st.tabs(["Interactions","PCA","Prototypes"])
    with tab1:
        st.subheader("Explore interactions")
        col11, col12 = st.columns((2, 2))
        with col11:
            st.write("##### Select a cluster")

            cluster = st.selectbox("Select how to group samples", data.setClusters(),
                                  key="cluster")

        with col12:
            st.write("##### Select a basket/tissue")

            basket = st.selectbox("Select how to group samples", data.setBaskets(),
                                  key="basket")

        subgroup,size = analysis_data.findInteraction(cluster,basket)

        variable = st.selectbox("Select information to display", ['Number of samples', 'Number of responsive samples'],
                                key="info")
        if variable == 'Number of samples':
            heatmap = analysis_data.heatmapNum(data, int(cluster), basket)
            saveheatmap(heatmap, cluster, basket)
            st.pyplot(heatmap)

        elif variable == 'Number of responsive samples':
            heat = analysis_data.heatmapResponse(data, int(cluster), basket)
            st.pyplot(heat)
        st.write("#### Samples")
        st.markdown("Number of samples in **cluster {}** & **basket {}**: {} ".format(cluster, basket, size))
        count =analysis_data.samplesCount(subgroup)
        st.pyplot(count,use_container_width=False)
        RawD = st.checkbox("Show raw data", key="raw-data-HM1")
        if RawD:
            saveRawDInt(subgroup)
            st.dataframe(subgroup, use_container_width=True)
        else:
            try:
                heatmap2= analysis_data.heatmapTranscripts(subgroup)
                saveheatmap_transpt(heatmap2,cluster,basket)
                st.pyplot(heatmap2)
            except:
                st.warning("Not enough samples. Please try a different combination.")
    with tab2:
        st.subheader("Advanced PCA")
        st.write("##### PCA of samples in **cluster {}** & **basket {}**".format(cluster, basket))
        RawD = st.checkbox("Show raw data", key="raw-data")
        adv_PCA(subgroup,RawD)
    with tab3:
        st.subheader("Prototypes of subgroup")
        st.write("##### Samples in **cluster {}** & **basket {}**".format(cluster, basket))
        sub_prototypes = Prototypes(data)
        sub_prototypes.findPrototypes_sub(subgroup)


