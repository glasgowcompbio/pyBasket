import streamlit as st
import tempfile
from processing import *
from common import add_logo,sideBar
import base64
import webbrowser
from interpretation import *
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components


st.set_page_config(
    page_title="pyBasket",
    page_icon="bi bi-basket",
    layout="wide"
)

add_logo()

sideBar()
st.header("pyBasket")
st.write("---")
menu = option_menu(None, ["Overview", "Data Upload", "File information", 'Drug information'],
    icons=["bi bi-basket", 'cloud-upload', "list-task", 'capsule'],
    menu_icon="cast", default_index=0, orientation="horizontal")

if menu == "Overview":
    st.write("### Welcome 👋")
    st.write(" ")
    st.write("The pyBasket App is a user-friendly interactive platform that allows the visualisation and exploration"
             " of results obtained from the pyBasket pipeline. The main goal of this App is to aid in the investigation of"
             " the omics profiles of the samples or patients involved in the clinical trials and gain potential biological insights"
             " into response patterns to a drug or treatment showed by the different subgroups.")
    st.write(" ")
    st.write("##### Basket trials")
    st.write("Basket trials are a type of early phase clinical trial that tests how well a new drug or treatment works in"
             " groups of patients that have different types of cancer but share the same molecular or genomic mutation or biomarker."
             " This innovative approach is more effective and cost-efficient, it requires fewer patients for the study and allows "
             "intractable or rare cancer patients to be also included in general clinical trials."
             " As they allow for multiple diseases to be studied, it facilitates faster drug development for specific "
             "subpopulations of patients and allows for a more personalised treatment approach.")
    st.write(" ")
    st.write("##### pyBasket")
    st.write("pyBasket is a two-stage approach that incorporates omics data into basket trial response prediction from cancer patients."
             " In the first stage, patients are clustered based on their omics profile using K-means clustering. "
             " These clusters assignments are used within a Hierarchical Bayesian model along with basket response rates to "
             " estimate overall response rates and predict interaction terms between baskets and clusters.")

    st.write("---")
    with st.expander("##### Basic steps to use this APP"):
        #st.write("##### General steps ")
        st.write(" ")
        col11, col12 = st.columns((2,2))
        with col11:
            st.write("##### :one: Upload the data")
            st.write("Navigate to the _Data Upload_ tab in this same page. Select a pickle file with results obtained from the pyBasket pipeline.")
            st.write(" ")
            st.write("##### :two: Check file information")
            st.write(
                "Navigate to the _File Information_ tab in this same page where you can find information about the file uploaded.")
            st.write(" ")
            st.write("##### :three: Check drug information")
            st.write(
                "Navigate to the _Drug Information_ tab in this same page where you can find further information about the drug tested in the basket trial being analysed.")
            st.write(" ")
        with col12:
            st.write("##### :four: Explore the data")
            st.write(
                "Navigate to the _Data Exploration_ subpage located in the left sidebar. There you can find general information about the samples and features analysed.")
            st.write(" ")
            st.write("##### :five: Select and analyse a basket*cluster interaction")
            st.write("In the left sidebar, select a cluster number and a basket/tissue. Samples that fall in this interaction will be selected. "
                     "Navigate to the _Interactions_ subpage located in the left sidebar to further explore results from the samples included in the selected basket*cluster interaction")
            st.write(" ")
            st.write("##### :six: Use interpretable ML methods ")
            st.write(
                "Navigate to the _Feature importance_ subpage located in the left sidebar to use interpretable Machine Learning methods and explore important features.")
            st.write(" ")

if menu == "Data Upload":
    st.markdown("#### Please upload your data")
    input_file = st.file_uploader('Upload your data file (.py format)', type='p')
    file_name = ""
    if input_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
            tmp_file.write(input_file.getvalue())
            tmp_file.flush()
            file_name = tmp_file.name
            st.success('The file was successfully uploaded!', icon="✅")
            save_data = readPickle(tmp_file.name)
            dict = Results(save_data,input_file.name)
            analysis_data = Analysis(dict)
            dict.setFeatures()
            st.session_state["data"] = dict
            st.session_state["analysis"] = analysis_data
            st.session_state["File Name"] = input_file.name
    else:
        st.write(file_name)
    if "File Name" in st.session_state:
        st.success('Analysis ready', icon="✅")

def openDrugBank(num):
    webpage_link = "https://go.drugbank.com/drugs/" + num
    webbrowser.open(webpage_link)

def openGoogleScholar(drug):
    webpage_link = "https://scholar.google.com/scholar?hl=es&as_sdt=0%2C5&q={}&btnG=".format(drug)
    webbrowser.open(webpage_link)

def openWikipedia(drug):
    webpage_link = "https://en.wikipedia.org/wiki/"+drug
    webbrowser.open(webpage_link)

def openPubMed(drug):
    webpage_link = "https://pubmed.ncbi.nlm.nih.gov/?term={}" + drug
    webbrowser.open(webpage_link)

if menu == "File information":
    st.subheader("File information")

    if "File Name" in st.session_state:
        #st.write("**Current file is:** {}".format(st.session_state["File Name"]))
        st.session_state["data"].fileInfo()


if menu == 'Drug information':
    name = st.session_state["File Name"].split('_')
    drug = name[2]
    st.subheader("Drug information")
    st.write("#### The drug used in this basket trials was: **{}**".format(drug))
    st.write("Further information about the drug used in the pyBasket analysis.")
    accession_num = "0"
    if drug == "Erlotinib":
        accession_num = "DB00530"
    elif drug == "Docetaxel":
        accession_num = "DB01248"
    st.write("#### _DrugBank_")
    st.write("_DrugBank_ is a freely accesible online database containing information on drugs and drugs targets. It combined chemical,"
             " pharmacological and pharmaceutical data with information about drug target (sequence, structure, pathway, etc."
             " Further pharmacological and chemical information about {} can be found in the link to _DrugBank_ below.".format(drug))
    st.button('Open DrugBank', on_click=openDrugBank, args=(accession_num,))

    st.write("#### _PubMed_")
    st.write("_PubMed_ is a free full-text archive of biomedical and life sciences print and electronic journal literature,"
             " and supports current biomedical and healthcare research and practice. Other research and journal literature related to "
             "{} can be found in the link to _PubMed_ below.".format(drug))
    st.button('Open PubMed',on_click=openPubMed, args=(drug,))

    #st.write("#### _Google Scholar_")
    #st.button('Open Google Scholar',on_click=openGoogleScholar, args=(drug,))

    st.write("#### _Wikipedia_")
    st.write("_Wikipedia_ is a free online accessible encyclopedia where more general non-technical information about {} can be found".format(drug))
    st.button('Open Wikipedia', on_click=openWikipedia, args=(drug,))

