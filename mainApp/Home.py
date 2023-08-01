import tempfile
from common import add_logo,sideBar,readPickle
import webbrowser
from interpretation import *
from streamlit_option_menu import option_menu
from explorer import Data
from analysis import Analysis

def openWikipedia(drug):
    webpage_link = "https://en.wikipedia.org/wiki/"+drug
    webbrowser.open(webpage_link)
def openPubMed(drug):
    webpage_link = "https://pubmed.ncbi.nlm.nih.gov/?term={}" + drug
    webbrowser.open(webpage_link)
def openDrugBank(num):
    webpage_link = "https://go.drugbank.com/drugs/" + num
    webbrowser.open(webpage_link)

st.set_page_config(
    page_title="pyBasket",
    page_icon="bi bi-basket",
    layout="wide"
)

add_logo()

sideBar()
st.header("Home")
st.write("---")
menu = option_menu(None, ["Overview", "Data Upload", "Clinical trial information"],
    icons=["bi bi-basket", 'cloud-upload', 'capsule'],
    menu_icon="cast", default_index=0, orientation="horizontal")

#Overview subpage: general information about the app and basic steps to use it
if menu == "Overview":
    st.subheader("Welcome 👋")
    st.write(" ")
    st.write("The pyBasket App is a user-friendly interactive platform that allows the visualisation and exploration"
             " of results obtained from the pyBasket pipeline. The main goal of this App is to aid in the investigation of"
             " the omics profiles of the samples or patients involved in the clinical trials and gain biological insights"
             " into the response patterns showed by the different subgroups to a drug.")
    st.write(" ")
    st.write("##### Basket trials")
    st.write("Basket trials are a type of early phase clinical trial design that tests how well a new drug or treatment works in"
             " groups of patients that have different types of cancer but share the same molecular or genomic mutation or biomarker."
             " This innovative approach is more effective and cost-efficient, it requires fewer patients for the study and allows "
             "intractable or rare cancer patients to be also included in general clinical trials."
             " As they allow for multiple diseases to be studied, it facilitates faster drug development for specific "
             "subpopulations of patients and allows for a more personalised treatment approach.")
    st.write(" ")
    st.write("##### pyBasket")
    url = "https://glasgowcompbio.github.io/pyBasket/"
    st.write("pyBasket is a two-stage approach that incorporates omics data into basket trial response prediction from cancer patients."
             " In the first stage, patients are clustered based on their omics profile using K-means clustering. "
             " These clusters assignments are used within a hierarchical Bayesian model along with basket response rates to "
             " estimate overall response rates and predict interaction terms between baskets and clusters. Further information about the pyBasket model can be found here: [link](%s)" % url)
    st.write("---")
    with st.expander("##### Basic steps to use this APP"):
        st.write(" ")
        col11, col12 = st.columns((2,2))
        with col11:
            st.write("##### :one: Upload the data")
            st.write("Navigate to the _Data Upload_ tab in this same page. Select a pickle file with results obtained from the pyBasket pipeline to be uploaded.")
            st.write(" ")
            st.write("##### :two: Check file information")
            st.write(
                "Navigate to the _File Information_ tab in this same page to find information about the file uploaded.")
            st.write(" ")
            st.write("##### :three: Check drug information")
            st.write(
                "Navigate to the _Drug Information_ tab in this same page to find further information about the drug tested in the basket trial being analysed.")
            st.write(" ")
        with col12:
            st.write("##### :four: Explore the data")
            st.write(
                "Navigate to the _Data Exploration_ subpage located in the left sidebar. Find more general information about the samples, the predicted treatment response rates and "
                "perform Dimensionality reduction and Differential Expression Analysis.")
            st.write(" ")
            st.write("##### :five: Select and analyse a basket*cluster interaction")
            st.write("In the left sidebar, select a cluster number and a basket/tissue. Samples that fall in this interaction will be selected. "
                     "Navigate to the _Basket-Cluster_Interactions_ subpage located in the left sidebar to further explore results from the samples included in the selected basket-cluster interaction")
            st.write(" ")
            st.write("##### :six: Use interpretable ML methods ")
            st.write(
                "Navigate to the _Feature Importance_Analysis subpage located in the left sidebar to use interpretable Machine Learning methods and explore important features.")
            st.write(" ")

#Upload data subpage
if menu == "Data Upload":
    st.subheader("Please upload your data")
    files = "https://gla-my.sharepoint.com/personal/ronan_daly_glasgow_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fronan%5Fdaly%5Fglasgow%5Fac%5Fuk%2FDocuments%2FJoe%20Files%2FpyBasket&ga=1"
    st.write("Click the box to select a pickle file with results from the pyBasket pipeline to be uploaded. Other results files can be sourced from the pyBasket project's [OneDrive folder](%files).")
    input_file = st.file_uploader('Upload your data file (.py format)', type='p')
    file_name = ""
    if input_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
            tmp_file.write(input_file.getvalue())
            tmp_file.flush()
            file_name = tmp_file.name
            st.success('The file was successfully uploaded!', icon="✅")
            save_data = readPickle(tmp_file.name)
            data = Data(save_data,input_file.name)
            analysis = Analysis(save_data,input_file.name)
            data.setData()
            st.session_state["data"] = data
            st.session_state["saved data"] = save_data
            st.session_state["File Name"] = input_file.name
            st.session_state["Analysis"] = analysis
    else:
        st.write(file_name)
    if "File Name" in st.session_state:
        st.success('Analysis ready', icon="✅")
        st.info("Current file: {}".format(st.session_state['File Name']))

#File and drug information subpage
if menu == "Clinical trial information":
    st.subheader("File information")
    st.write("Once a file of results is uploaded, general information about the file is displayed here.")
    if "data" in st.session_state:
        st.session_state["data"].fileInfo()
        st.write(" ")
        st.subheader("Drug information")
        name = st.session_state["File Name"].split('_')
        drug = name[2]
        st.write("#### The drug used in this basket trial was: **{}**".format(drug))
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

        st.write("#### _Wikipedia_")
        st.write("_Wikipedia_ is a free online accessible encyclopedia where more general non-technical information about {} can be found".format(drug))
        st.button('Open Wikipedia', on_click=openWikipedia, args=(drug,))
    else:
        st.warning("No data found. Please upload results in the 'Data Upload' tab")