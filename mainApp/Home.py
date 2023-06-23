import streamlit as st
import tempfile
from processing import *
from common import add_logo
import base64

st.set_page_config(
    page_title="pyBasket",
    page_icon="👋",
    layout="wide"
)
add_logo()


st.header("pyBasket")


st.markdown("#### Please upload your data")
input_file = st.file_uploader('Upload your data file (.py format)', type='p')

file_name = ""
if input_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_file.flush()
        file_name = tmp_file.name
        print(tmp_file.name)
        st.success('The file was successfully uploaded!', icon="✅")
        save_data = readPickle(tmp_file.name)
        dict = Results(save_data,input_file.name)
        dict.setFeatures()
        st.session_state["data"] = dict
        print(input_file.name)
        st.session_state["File Name"] = input_file.name

else:
    st.write(file_name)


st.subheader("Current file")
if "File Name" in st.session_state:
    st.success('Analysis ready', icon="✅")
    with st.expander("File information"):
        st.session_state["data"].fileInfo()
        #st.table(st.session_state["data"].fileInfo())
        #st.markdown("Name of file:  " + st.session_state["File Name"])
        #print(st.session_state["data"].fileInfo())
        #st.markdown("Number of samples:  " + str(dict.numSamples(st.session_state["data"].expr_df_selected)))










