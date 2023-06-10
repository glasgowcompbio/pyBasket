import os
import sys
import argparse
import pandas as pd
import numpy as np
sys.path.append('/Users/marinaflores/Desktop/bioinformatics/MBioinfProject/mainApp/pyBasket/pyBasket')

from pyBasket.common import load_obj, save_obj
from pyBasket.preprocessing import select_rf, check_rf
from sklearn.cluster import KMeans
from pyBasket.clustering import get_cluster_df_by_basket, get_patient_df
from pyBasket.model import get_patient_model_hierarchical_log_odds, get_patient_model_hierarchical_log_odds_nc
import pymc as pm
import arviz as az

#sys.path.append('..')
#sys.path.append('.')

os.chdir('/Users/marinaflores/Desktop/bioinformatics/MBioinfProject/mainApp/pyBasket/pyBasket')
data_dir = os.path.abspath(os.path.join('..','pyBasket/Data'))
current_dir =os.getcwd()

#Function to change Ensemble IDs for genes names
def IdGenes(df):
    features_EN = df.columns.tolist()
    matching_genes = genes_df[genes_df['ensg_v99'].isin(features_EN)].reset_index(drop=True)
    matching_genesEN = matching_genes['ensg_v99'].tolist()
    genes = []
    for col in features_EN:
        found_match = False
        for i in range(len(matching_genesEN)):
            if col == matching_genesEN[i]:
                genes.append(matching_genes['gene_name_v99'][i])
                found_match = True
                break
        if not found_match:
            genes.append(col)
    return genes

with open('log.txt', 'w') as file:
    file.write("These are the results for the pyBasket pipeline\n")
    """Processing of raw data"""
    expr_file = os.path.join(data_dir, 'GDSCv2.exprsALL.tsv')
    genes_file = os.path.join(data_dir, 'Entrez_to_Ensg99.mapping_table.tsv')
    genes_df = pd.read_csv(genes_file, sep='\t')
    df = pd.read_csv(expr_file, sep='\t')
    expr_df = df.drop(df.index[-25:]).transpose()
    tissue_df = df.tail(25).astype(int).transpose()
    flat_list = tissue_df.reset_index().melt(id_vars='index', var_name='tissue', value_name='belongs_to_tissue')
    flat_list = flat_list[flat_list['belongs_to_tissue'] != 0]
    flat_list = flat_list.set_index('index')
    sample_dict = flat_list['tissue'].to_dict()
    tissues = np.array([sample_dict[s] for s in expr_df.index.values])
    """Load drug response.There is data for 11 drugs"""
    response_file = os.path.join(data_dir, 'GDSCv2.aacALL.tsv')
    response_df = pd.read_csv(response_file, sep='\t').transpose()
    drug_name = 'Docetaxel'
    file.write('The drug being tested is: {}\n'.format(drug_name))
    samples = tissue_df.index.values
    response_dict = response_df[drug_name].to_dict()
    responses = np.array([response_dict[s] for s in samples])
    file.write("Number of samples: {}\n".format(len(samples)))
    file.write("Number of responses: {}\n".format(len(responses)))
    file.write("Number of tissues/baskets: {}\n".format(len(tissues)))
    df = pd.DataFrame({
        'tissues': tissues,
        'samples': samples,
        'responses': responses
    })
    # Drop NaN responses
    df = df.dropna(subset=['responses'])
    # List of tissues
    basket_names = df['tissues'].unique()
    #file.write(basket_names.tolist())
    """Select baskets for trials"""
    df_filtered = df[df['tissues'].isin(basket_names)].reset_index(drop=True)
    sample_list = df_filtered['samples'].tolist()
    # Filter samples that are in the list of samples with responses to the selected drug
    expr_df_filtered = expr_df[expr_df.index.isin(sample_list)]
    # Change the index to be the samples names
    drug_response = df_filtered.set_index('samples').drop(columns=['tissues'])
    #Select all baskets for analysis
    """Feature selection using random forest"""
    try:  # try to load previously selected features
        fname = os.path.join(current_dir + '/results', '%s_expr_df_selected.p' % drug_name)
        expr_df_selected = load_obj(fname)

    except FileNotFoundError:  # if not found, then re-run feature selection
        expr_df_selected = select_rf(expr_df_filtered, drug_response, n_splits=5, percentile_threshold=90,
                                     top_genes=500)
        save_obj(expr_df_selected, fname)

    importance_df = check_rf(expr_df_selected, drug_response, test_size=0.2)
    selected_genes = IdGenes(expr_df_selected)
    filtered_genes = IdGenes(expr_df_filtered)
    expr_df_selected.columns = selected_genes
    importance_df.index = selected_genes
    expr_df_filtered.columns = filtered_genes

    classes = df_filtered.set_index('samples')
    C = 5  # parameter to choose
    file.write("Number of clusters: {}\n".format(C))
    kmeans = KMeans(n_clusters=C, random_state=42)
    kmeans.fit(expr_df_selected)

    cluster_labels = kmeans.labels_
    class_labels = classes.tissues.values
    # Create clustering dataframe
    cluster_df = get_cluster_df_by_basket(class_labels, cluster_labels, normalise=False)

    # Prepare patient data: dataframe with sample information for tissues, responses, cluster number and responsive
    patient_df = get_patient_df(df_filtered, cluster_labels)

    """Hierarchical Bayesian model """
    n_burn_in = int(5E3)
    n_sample = int(5E3)
    target_accept = 0.99
    model_h2 = get_patient_model_hierarchical_log_odds(patient_df)
    model_h2_nc = get_patient_model_hierarchical_log_odds_nc(patient_df)
    with model_h2_nc:
        trace_h2 = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True})

    # summary of model
    az.summary(trace_h2).round(2)
    stacked_h2 = az.extract(trace_h2)
    inferred_basket_h2 = np.mean(stacked_h2.basket_p.values, axis=1)
    inferred_cluster_h2 = np.mean(stacked_h2.cluster_p.values, axis=2)
    inferred_basket_h2_tiled = np.tile(inferred_basket_h2, (C, 1)).T
    inferred_mat_h2 = inferred_basket_h2_tiled * inferred_cluster_h2

    save_data = {
        'expr_df_filtered': expr_df_filtered,
        'expr_df_selected': expr_df_selected,
        'drug_response': drug_response,
        'class_labels': class_labels,
        'cluster_labels': cluster_labels,
        'patient_df': patient_df,
        'stacked_posterior': stacked_h2,
        'trace': trace_h2,
        'importance_df': importance_df
    }
    results = os.path.join(current_dir + '/results', 'patient_analysis_%s_cluster_%d.p' % (drug_name, C))
    save_obj(save_data, results)
    patient_df.to_csv(results)
    file.write("All results are stored in: {}".format(results))

"""
argParser = parser = argparse.ArgumentParser(
                    prog='pyBasket pipeline',
                    epilog='Text at the bottom of help')
argParser.add_argument("-d", "--data", help="Data to be analysed")

args = argParser.parse_args()
print(args.data)
"""