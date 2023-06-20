import matplotlib.pyplot as plt
import streamlit as st
from processing import readPickle, Results, add_logo, hideRows, Analysis
import numpy as np
import seaborn as sns
import mpld3
import streamlit.components.v1 as components
from interpretation import Prototypes, Kmedoids,DEA

hide_rows = hideRows()
add_logo()
st.header("Data exploration")

if "data" in st.session_state:
    data = st.session_state["data"]

    tab1, tab2,tab3, tab4,tab5 = st.tabs(["Number of samples", "AAC response", "PCA analysis", "Prototypes", "Differential expression"])
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
            RawD_AAC = st.checkbox("Show raw data", key="raw-data-AAC")
        if option_tab2 == 'None':
            with col21:
                option_tab3 = st.selectbox("Show subgroups", ('None','Clusters', 'Baskets/Tissues'),
                                       key="option-tab3")
            if option_tab3 == "None":
                data.non_group_plot(None,RawD_AAC)
            elif option_tab3 == "Clusters":
                data.non_group_plot("cluster_number",RawD_AAC)
            elif option_tab3 == 'Baskets/Tissues':
                data.non_group_plot("tissues",RawD_AAC)
        else:
            with col22:
                RD_AAC = st.checkbox("Group by responsiveness", key="responsive-AAC")
            if option_tab2 == "Clusters":
                data.displayAAC("cluster_number", "Cluster number", RD_AAC, RawD_AAC,"AAC response per cluster")
            elif option_tab2 == 'Baskets/Tissues':
                data.displayAAC("tissues", "Tissue", RD_AAC,RawD_AAC, "AAC response per tissue")

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

    with tab5:
        st.subheader("Differential expression")
        col51, col52 = st.columns((2, 2))
        with col51:
            option = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues', 'Responsive'), key="DEA")
            if option ==  'Clusters':
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
                #st.write('The current p-value threshold is: ', pthresh)
            dea = DEA(data)
            dea.diffAnalysis_simple(groups[0],groups[1],feature,pthresh,logthresh)
            #st.pyplot(volcano)
            with col52:
                st.write(" ")
                st.write(" ")
                dea.infoTest(groups[0],groups[1],option,pthresh,logthresh)












