import matplotlib.pyplot as plt
import streamlit as st
from processing import readPickle, Results, add_logo, hideRows, Analysis
import numpy as np
import seaborn as sns
import mpld3
import streamlit.components.v1 as components
from interpretation import Prototypes, Kmedoids

hide_rows = hideRows()
add_logo()
st.header("Data overview")

if "data" in st.session_state:
    data = st.session_state["data"]

    def savePlot_AAC(fig):
        if st.button('Save Plot', key="plot_AAC_png"):  # Update the key to a unique value
            fig.savefig('plot_AAC.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")

    def saveTable_AAC(df):
        if st.button('Save table', key="table_AAC_csv"):  # Update the key to a unique value
            df.to_csv('raw_data_AAC.csv', index=False)
            st.info('Data saved as .csv file in working directory', icon="ℹ️")
        else:
            st.write("")


    def displayAAC_none(feature):
        fig = data.non_group_plot(feature)
        savePlot_AAC(fig)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600, width=1000)

    tab1, tab2,tab3, tab4 = st.tabs(["Number of samples", "AAC response", "PCA analysis", "Prototypes"])
    with tab1:
        st.subheader("Number of samples")
        col11, col12 = st.columns((1, 1))
        with col11:
            option_tab1 = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'), key="option-tab1")
        with col12:
            RD = st.checkbox("Group by responsiveness", key="responsive")
            RawD = st.checkbox("Show raw data", key="raw-data")
        if option_tab1 == "Clusters":
            data.displayNums("cluster_number","Cluster number",RD, RawD,"Samples per cluster")
        elif option_tab1 == 'Baskets/Tissues':
            data.displayNums("tissues", "Tissue",RD, RawD, "Samples per tissue")

    with tab2:
        st.subheader("AAC response")
        col21, col22 = st.columns((1, 1))
        with col21:
            option_tab2 = st.selectbox("Select how to group samples", ('None','Clusters', 'Baskets/Tissues'), key="option-tab2")
        with col22:
            RD_AAC = st.checkbox("Group by responsiveness", key="responsive-AAC")
            RawD_AAC = st.checkbox("Show raw data", key="raw-data-AAC")
        if option_tab2 == "Clusters":
            data.displayAAC("cluster_number", "Cluster number", RD_AAC, RawD_AAC,"AAC response per cluster")
        elif option_tab2 == 'Baskets/Tissues':
            data.displayAAC("tissues", "Tissue", RD_AAC,RawD_AAC, "AAC response per tissue")
        elif option_tab2 == 'None':
            with col21:
                option_tab3 = st.selectbox("Show subgroups", ('None','Clusters', 'Baskets/Tissues'),
                                       key="option-tab3")
            if option_tab3 == "None":
                displayAAC_none(None)
            elif option_tab3 == "Clusters":
                displayAAC_none("cluster_number")
            elif option_tab3 == 'Baskets/Tissues':
                displayAAC_none("tissues")
    with tab3:
        analysis_data = Analysis(data)
        st.subheader("Samples PCA")
        option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues', 'Responsive'), key="PCA")
        RawD = st.checkbox("Show raw data", key="raw-data-PCA")
        if option == "Clusters":
            choices = analysis_data.patient_df["cluster_number"].unique()
            analysis_data.PCA_analysis("cluster_number", RawD)
        elif option == "Responsive":
            analysis_data.PCA_analysis("responsive", RawD)
            choices = analysis_data.patient_df["responsive"].unique()
        elif option == 'Baskets/Tissues':
            analysis_data.PCA_analysis("tissues", RawD)
            choices = analysis_data.patient_df["tissues"].unique()
    with tab4:
        st.subheader("Prototypes")
        option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'), key="Prototypes")
        prototype = Prototypes(data)
        prototype.findPrototypes(option)
        #Kmedoids()









