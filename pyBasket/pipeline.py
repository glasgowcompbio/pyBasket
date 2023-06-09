import os
import sys
import argparse
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

sys.path.append('..')
sys.path.append('.')

from pyBasket.common import load_obj, save_obj
from pyBasket.model import get_patient_model_simple, get_patient_model_hierarchical
from pyBasket.model import get_patient_model_hierarchical_log_odds, get_patient_model_hierarchical_log_odds_nc
from pyBasket.clustering import get_cluster_df_by_basket, plot_PCA, get_patient_df
from pyBasket.preprocessing import select_rf, check_rf

#data_dir = os.path.abspath(os.path.join('..', 'Data'))
#data_dir

argParser = parser = argparse.ArgumentParser(
                    prog='pyBasket pipeline',
                    epilog='Text at the bottom of help')
argParser.add_argument("-n", "--name", help="your name")

args = argParser.parse_args()
print(args[0])


