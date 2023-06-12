import matplotlib.pyplot as plt
import streamlit as st
from processing import readPickle, Results, add_logo, hideRows, Analysis
import numpy as np
import seaborn as sns
import mpld3
import streamlit.components.v1 as components

hide_rows = hideRows()
add_logo()
st.header("Data overview")

def savePlot_PCA(fig, feature):
    if st.button('Save Plot', key="plot_PCA"):  # Update the key to a unique value
        fig.savefig('plot_PCA_'+feature+'.png')
        st.info('Plot saved as .png in working directory', icon="ℹ️")
    else:
        st.write("")

def saveRawD_PCA_var(tab, feature):
    if st.button('Save Data', key="data_varPCA"):  # Update the key to a unique value
        tab.to_csv('dataPCA_var_'+feature+'.csv')
        st.info('Data saved as .csv in working directory', icon="ℹ️")
    else:
        st.write("")

def saveRawD_PCA_df(tab, feature):
    if st.button('Save Data', key="data_dfPCA"):  # Update the key to a unique value
        tab.to_csv('dataPCA_df_' + feature + '.csv')
        st.info('Data saved as .csv in working directory', icon="ℹ️")
    else:
        st.write("")

def PCA_analysis(feature, RawD):
    df, variance = analysis_data.main_PCA(feature)
    if RawD is True:
        pcaDF, var_df = analysis_data.showRawData_PCA(df,variance)
        col11, col12 = st.columns((2, 3))
        with col11:
            st.write('##### Variance explained by component')
            saveRawD_PCA_var(var_df, feature)
            st.dataframe(var_df,use_container_width = True)

        with col12:
            st.write('##### PCA results')
            saveRawD_PCA_df(pcaDF, feature)
            st.dataframe(pcaDF,use_container_width = True)
    else:
        fig = analysis_data.plot_PCA(df, feature)
        savePlot_PCA(fig, option)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=650, width=3000)
        #st.divider()
    #interactive_legend = mpld3.plugins.InteractiveLegendPlugin(scatter, labels = labels,start_visible=True)
    #mpld3.plugins.connect(fig, interactive_legend)
    #mpld3.display()
    #st.pyplot(fig)

if "data" in st.session_state:
    data = st.session_state["data"]
    def savePlot_Nums(fig):
        if st.button('Save Plot', key="plot_nums_png"):  # Update the key to a unique value
            fig.savefig('plot_NoS.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")

    def savePlot_AAC(fig):
        if st.button('Save Plot', key="plot_AAC_png"):  # Update the key to a unique value
            fig.savefig('plot_AAC.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")

    def saveTable_Nums(df):
        if st.button('Save table', key="table_Nums_csv"):  # Update the key to a unique value
            df.to_csv('raw_data_NoS.csv', index=False)
            st.info('Data saved as .csv file in working directory', icon="ℹ️")
        else:
            st.write("")

    def saveTable_AAC(df):
        if st.button('Save table', key="table_AAC_csv"):  # Update the key to a unique value
            df.to_csv('raw_data_AAC.csv', index=False)
            st.info('Data saved as .csv file in working directory', icon="ℹ️")
        else:
            st.write("")

    def displayNums(feature, feature_title,RD, RawD, title_plot):
        if RawD is True:
            raw_num = data.raw_data_count(feature, feature_title, RD)
            saveTable_Nums(raw_num)
            st.dataframe(raw_num,use_container_width = True)
        else:
            num_plot = data.count_plot(feature, title_plot,
                                       feature_title, RD)
            savePlot_Nums(num_plot)
            st.pyplot(num_plot)


    def displayAAC(feature, feature_title,RD,RawD, title_plot):
        if RawD is True:
            raw_AAC = data.raw_data_AAC(feature, feature_title)
            saveTable_AAC(raw_AAC)
            st.dataframe(raw_AAC,use_container_width = True)
        else:
            AAC = data.AAC_plot(feature, title_plot,
                                feature_title, RD)
            savePlot_AAC(AAC)
            st.pyplot(AAC)

    def displayAAC_none(feature):
        fig = data.non_group_plot(feature)
        savePlot_AAC(fig)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600, width=1000)


    tab1, tab2,tab3 = st.tabs(["Number of samples", "AAC response", "PCA analysis"])
    with tab1:
        st.subheader("Number of samples")
        col11, col12 = st.columns((1, 1))
        with col11:
            option_tab1 = st.selectbox("Select how to group samples", ('Clusters', 'Baskets/Tissues'), key="option-tab1")
        with col12:
            RD = st.checkbox("Group by responsiveness", key="responsive")
            RawD = st.checkbox("Show raw data", key="raw-data")
        if option_tab1 == "Clusters":
            displayNums("cluster_number","Cluster number",RD, RawD,"Samples per cluster")
        elif option_tab1 == 'Baskets/Tissues':
            displayNums("tissues", "Tissue",RD, RawD, "Samples per tissue")

    with tab2:
        st.subheader("AAC response")
        col21, col22 = st.columns((1, 1))
        with col21:
            option_tab2 = st.selectbox("Select how to group samples", ('None','Clusters', 'Baskets/Tissues'), key="option-tab2")
        with col22:
            RD_AAC = st.checkbox("Group by responsiveness", key="responsive-AAC")
            RawD_AAC = st.checkbox("Show raw data", key="raw-data-AAC")
        if option_tab2 == "Clusters":
            displayAAC("cluster_number", "Cluster number", RD_AAC, RawD_AAC,"AAC response per cluster")
        elif option_tab2 == 'Baskets/Tissues':
            displayAAC("tissues", "Tissue", RD_AAC,RawD_AAC, "AAC response per tissue")
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
            PCA_analysis("cluster_number", RawD)
        elif option == "Responsive":
            PCA_analysis("responsive", RawD)
            choices = analysis_data.patient_df["responsive"].unique()
        elif option == 'Baskets/Tissues':
            PCA_analysis("tissues", RawD)
            choices = analysis_data.patient_df["tissues"].unique()








