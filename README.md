# pyBasket

This project provides codes to accompany the draft manuscript 'Omics Design For Basket Trial in Precision Medicine'.

**Abstract:**
> Unlike traditional clinical trials, which focus on a single disease, basket trials identify patients sharing a common biomarker and evaluate their treatments across multiple diseases (baskets) simultaneously. This increases efficiency, reduces cost, and offers better compatibility with personalised medicine. Despite their efficiency, these trials can struggle with reduced statistical power in smaller patient groups. i.e., in the case of rare diseases. Omics measurements, which encompass large-scale molecular data, offer a way to enhance basket trials through better stratification of patients' molecular profiles, but their use in basket trials remain limited. Here, we introduce pyBasket, a two-stage approach that utilises omics data to improve basket trial design. First, patients are grouped based their omics profiles using techniques like k-means clustering, then a hierarchical Bayesian model is used to estimate overall basket response rates and the interactions between baskets and omics clusters. Simulated results indicate that pyBasket outperforms conventional baseline methods, reducing prediction error and increasing statistical power. Crucially, our method provides insights into different patient subgroups, stratified according to omics profiles, within the same basket that exhibit varying responses to treatment. This is a result not readily available from existing alternatives that exclude omics data. We also applied pyBasket to a real-world Genomics of Drug Sensitivity in Cancer (GDSC) dataset, which mirrors a single-step large basket trial, and discovered basket-cluster combinations that are significantly more responsive to treatment. Our novel integration of omics into basket trial design enhances patient stratification in personalised medicine and uncovers molecular reasons for diverse treatment responses.
=======
Bayesian basket trial simulation in Python.
 
### App
>>>>>>> marina

The App for the pyBasket is being developed in Streamlit. To run the App, navigate to mainApp directory and from command line run: streamlit run Home.py.

<<<<<<< HEAD
We provide several ways to manage your dependencies to run pyBasket.

***A. Managing Dependencies using Pipenv***

1. Install pipenv (https://pipenv.readthedocs.io).
2. In the cloned Github repo, run `$ pipenv install`.
3. Enter virtual environment using `$ pipenv shell`.

***B. Managing Dependencies using Poetry***

1. Install poetry (https://python-poetry.org/).
2. In the cloned Github repo, run `$ poetry install`.
3. Enter virtual environment using `$ poetry shell`.

***B. Managing Dependencies using Anaconda Python***

1. Install Anaconda Python (https://www.anaconda.com/products/individual).
2. In the cloned Github repo, run `$ conda env create --file environment.yml`.
3. Enter virtual environment using `$ conda activate pyBasket`.

Once you have activated the virtual environment, you could develop run the environment, train models etc by running
   notebooks (`$ jupyter lab`). Be sure to also install graphviz if necessary https://graphviz.org/download/ to generate plots of plate diagram in notebooks.

## Examples

Example notebooks can be found [here](https://github.com/glasgowcompbio/pyBasket/tree/main/notebooks).
This includes a demonstration of the environment, as well as other notebooks to train models and
evaluate the results.

## Interactive Viewer

Required to run the App:
- Results from the pyBasket pipeline in pickle format e.g. patient_analysis_Erlotinib_cluster_5.p
