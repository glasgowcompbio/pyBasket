import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append('..')
sys.path.append('.')
"""
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


import seaborn as sns
from sklearn.cluster import KMeans
from pyBasket.common import load_obj, save_obj
from pyBasket.model import get_patient_model_simple, get_patient_model_hierarchical
from pyBasket.model import get_patient_model_hierarchical_log_odds, get_patient_model_hierarchical_log_odds_nc
from pyBasket.clustering import get_cluster_df_by_basket, plot_PCA, get_patient_df
from pyBasket.preprocessing import select_rf, check_rf
"""

data_dir = os.path.abspath(os.path.join('../pyBasket/', 'Data'))

"""
argParser = parser = argparse.ArgumentParser(
                    prog='pyBasket pipeline',
                    epilog='Text at the bottom of help')
argParser.add_argument("-d", "--data", help="Data to be analysed")

args = argParser.parse_args()
print(args.data)
"""

"""Processing of raw data"""
expr_file = os.path.join(data_dir, 'GDSCv2.exprsALL.tsv')
df = pd.read_csv(expr_file, sep='\t')
expr_df = df.drop(df.index[-25:]).transpose()
#each sample is assigned to one tissue
tissue_df = df.tail(25).astype(int).transpose()

flat_list = tissue_df.reset_index().melt(id_vars='index', var_name='tissue', value_name='belongs_to_tissue')
flat_list = flat_list[flat_list['belongs_to_tissue'] != 0]
# set the index to the sample name
flat_list = flat_list.set_index('index')
# create the dictionary
sample_dict = flat_list['tissue'].to_dict()
#Tissue assigned to each sample
tissues = np.array([sample_dict[s] for s in expr_df.index.values])

"""Load drug response
There is data for 11 drugs"""
response_file = os.path.join(data_dir, 'GDSCv2.aacALL.tsv')
response_df = pd.read_csv(response_file, sep='\t').transpose()
#Collect the data for the chosen drug
drug_name = 'Docetaxel' """turn this into a command-line parameter"""





