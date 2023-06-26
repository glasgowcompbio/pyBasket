import streamlit as st
import tempfile
from processing import *
from common import add_logo
import base64
import webbrowser

import streamlit.components.v1 as components

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

def openDrugBank(num):
    webpage_link = "https://go.drugbank.com/drugs/" + num
    webbrowser.open(webpage_link)
def openGoogleScholar(drug):
    webpage_link = "https://scholar.google.com/scholar?hl=es&as_sdt=0%2C5&q={}&btnG=".format(drug)
    webbrowser.open(webpage_link)

def openWikipedia(drug):
    webpage_link = "https://en.wikipedia.org/wiki/"+drug
    webbrowser.open(webpage_link)

st.subheader("Current file")
if "File Name" in st.session_state:
    st.success('Analysis ready', icon="✅")
    col11, col12 = st.columns((2,2))
    with col11:
        with st.expander("File information"):
            st.session_state["data"].fileInfo()
            #st.table(st.session_state["data"].fileInfo())
            #st.markdown("Name of file:  " + st.session_state["File Name"])
            #print(st.session_state["data"].fileInfo())
            #st.markdown("Number of samples:  " + str(dict.numSamples(st.session_state["data"].expr_df_selected)))
    with col12:
        with st.expander("Drug information"):
            name = st.session_state["File Name"].split('_')
            drug = name[2]
            st.subheader("**{}**".format(drug))
            st.write("Further information about the treatment/therapy used to generate the results.")
            accession_num = "0"
            if drug == "Erlotinib":
                accession_num = "DB00530"
            elif drug == "Docetaxel":
                accession_num = "DB01248"
            st.button('Open DrugBank', on_click=openDrugBank, args=(accession_num,))
            st.button('Open Google Scholar',on_click=openGoogleScholar, args=(drug,))
            st.button('Open Wikipedia', on_click=openWikipedia, args=(drug,))

