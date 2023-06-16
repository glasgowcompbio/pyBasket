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
st.header("Features importance")
if "data" in st.session_state:
    data = st.session_state["data"]
    analysis_data = Analysis(data)
    importance = data.importance_df
    importance_sort = np.sort(importance['importance_score'].values)
    fig = plt.figure(figsize=(12, 6))
    plt.bar([x for x in range(len(importance))],importance_sort  )
    st.pyplot(fig)