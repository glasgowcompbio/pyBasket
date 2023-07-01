import gzip
import pickle
import mpld3
import numpy as np
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from processing import *

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "pyBasket";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hideRows():
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    return hide_table_row_index

def savePlot(fig, feature):
    if st.button('Save Plot', key="plot"+feature):  # Update the key to a unique value
        fig.savefig('plot_' + feature + '.png')
        st.info('Plot saved as .png in working directory', icon="ℹ️")
    else:
        st.write("")

def saveTable(df, feature):
    if st.button('Save table', key="table_"+feature):  # Update the key to a unique value
        df.to_csv('raw_data_'+feature+'.csv', index=False)
        st.info('Data saved as .csv file in working directory', icon="ℹ️")
    else:
        st.write("")

def sideBar():
    if "data" in st.session_state:
        data = st.session_state["data"]
        analysis_data = st.session_state["analysis"]
        st.sidebar.title("Select basket*cluster interaction")
        with st.sidebar:
            cluster = st.selectbox("Select a cluster", data.setClusters(), key="cluster")
            if "cluster" not in st.session_state:
                st.session_state["cluster"] = cluster
            basket = st.selectbox("Select a basket", data.setBaskets(), key="basket")
            if "basket" not in st.session_state:
                st.session_state["basket"] = basket
            subgroup, size = analysis_data.findInteraction(cluster, basket)
            st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket, size))
    else:
        with st.sidebar:
            st.warning("Results not found. Please upload in a results file in Home.")
