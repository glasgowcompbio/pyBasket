import os
import gzip
import pickle
import mpld3
import numpy as np
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import savePlot,saveTable,alt_ver_barplot,alt_scatterplot,alt_boxplot
import altair as alt

genes_path = os.path.join('..', 'pyBasket/Data', 'Entrez_to_Ensg99.mapping_table.tsv')

def readPickle(pick_file):
    try:
        with gzip.GzipFile(pick_file, 'rb') as f:
            return pickle.load(f)
    except OSError:
        logger.warning('Old, invalid or missing pickle in %s. '
                       'Please regenerate this file.' % pick_file)
        raise

class Data():

    def __init__(self,pickle_data, name):
        self.file_name = name
        self.expr_df_filtered = pickle_data[list(pickle_data)[0]]
        self.expr_df_selected = pickle_data[list(pickle_data)[1]]
        self.drug_response = pickle_data[list(pickle_data)[2]]
        self.class_labels = pickle_data[list(pickle_data)[3]]
        self.cluster_labels = pickle_data[list(pickle_data)[4]]
        self.patient_df = pickle_data[list(pickle_data)[5]]
        self.importance_df = pickle_data[list(pickle_data)[8]]
        self.stacked_posterior = pickle_data[list(pickle_data)[6]]
        self.clusters_names = sorted(list(self.patient_df['cluster_number'].unique()))
        self.baskets_names = sorted(list(self.patient_df['tissues'].unique()))

    @staticmethod
    def IdGenes(df):
        features_EN = df.columns.tolist()
        genes_df = pd.read_csv(genes_path, sep='\t')
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

    #Change responsive to be categorical
    #Change features to be by genes not by ENSEMBL ID
    def setData(self):
        self.patient_df["responsive"] = self.patient_df["responsive"].replace([0, 1], ['Non-responsive', 'Responsive'])
        selected_genes = Data.IdGenes(self.expr_df_selected)
        filtered_genes = Data.IdGenes(self.expr_df_filtered)
        self.expr_df_selected.columns = selected_genes
        self.importance_df.index = selected_genes
        self.expr_df_filtered.columns = filtered_genes

    #Display information about the input file
    def fileInfo(self):
        num_samples = len(self.expr_df_filtered.axes[0])
        num_feat_filt = len(self.expr_df_filtered.axes[1])
        num_feat_select = len(self.expr_df_selected.axes[1])
        clusters = len(self.clusters_names)
        tissues = len(self.baskets_names)
        drug = self.file_name.split('_')[2]
        info = {'File name: ': {'information': self.file_name}, 'Drug: ': {'information': drug},'Num. cell lines' : {'information':num_samples},
                'Num. transcripts (after filtering): ': {'information': num_feat_filt},
                'Num. transcripts (after feature selection): ': {'information': num_feat_select},
                'Num.clusters: ': {'information': clusters}, 'Num. tissues/baskets: ': {'information': tissues}}
        df = pd.DataFrame(data=info).T
        style = df.style.hide_index()
        style.hide_columns()
        st.dataframe(df, use_container_width=True)

    def countPlot(self, RD, title,feature,x_lab):
        self.patient_df[feature] = pd.Categorical(self.patient_df[feature])
        size = len(np.unique(self.patient_df[feature]))
        if RD:
            df_grouped = self.patient_df.groupby(["responsive", feature]).size().reset_index(name='Count')
            alt_ver_barplot(df_grouped, feature, 'Count', 2, x_lab, "Number of samples", "responsive", title, "NumSamples",
                            ["responsive", feature])
        else:
            df_grouped = self.patient_df.groupby([feature]).size().reset_index(name='Count')
            alt_ver_barplot(df_grouped, feature, 'Count', size, x_lab, "Number of samples", feature, title, "NumSamples",
                            [feature,'Count'])

    def showRawData(self, feature, x_variable, RD):
        if RD:
            features = ["responsive", feature]
            columns = ["responsive",x_variable, "Number of samples"]
        else:
            features = feature
            columns = [x_variable, "Number of samples"]
        raw_count = self.patient_df.groupby(features).size().to_frame(name = 'count').reset_index()
        raw_count.columns = columns
        if RD:
            raw_count['responsive'] = raw_count['responsive']== "Responsive"
        saveTable(raw_count, "NumOfS")
        st.dataframe(raw_count, use_container_width=True)

    def AAC_scatterplot(self, RawD):
        if RawD:
            saveTable(self.patient_df, "rawAAC")
            self.patient_df["responsive"] = self.patient_df["responsive"] == "Responsive"
            st.dataframe(self.patient_df, use_container_width=True)
        else:
            reseted_df = self.patient_df.reset_index()
            reseted_df["index"] = range(reseted_df.shape[0])
            alt_scatterplot(reseted_df, 'index', 'responses', "Sample index", "AAC response", "AAC response", "AAC", ["samples", "responses"])

    def raw_data_AAC(self,feature,x_variable):
        raw_count = self.patient_df[[feature, "responses"]].groupby(feature).mean()
        default_value = 0
        raw_count['SD'] = self.patient_df[[feature, "responses"]].groupby(feature).std().fillna(default_value)
        raw_count['Median'] = self.patient_df[[feature, "responses"]].groupby(feature).median()
        raw_count['Min'] = self.patient_df[[feature, "responses"]].groupby(feature).min()
        raw_count['Max'] = self.patient_df[[feature, "responses"]].groupby(feature).max()
        raw_count = raw_count.reset_index()
        raw_count.columns = [x_variable, "Mean", "SD", "Median", "Min", "Max"]
        return raw_count

    def AAC_response(self,feature, RD, x_lab, RawD):
        size = len(np.unique(self.patient_df[feature]))
        if RawD:
            df = Data.raw_data_AAC(self,feature, x_lab)
            saveTable(df, "rawAAC")
            st.dataframe(df, use_container_width=True)
        else:
            if RD:
                alt_boxplot(self.patient_df, feature, "responses", 2, x_lab, "AAC response", "responsive", "AAC response",
                            "AAC")
            else:
                alt_boxplot(self.patient_df, feature, "responses", size, x_lab, "AAC response", feature, "AAC response", "AAC")

    def barInferredProb(self, feature):
        stacked = self.stacked_posterior
        if feature == 'baskets':
            len_colors = len(self.baskets_names)
            inferred_basket = np.mean(stacked.basket_p.values, axis=1)
            df = pd.DataFrame({'prob': inferred_basket, 'tissue': self.baskets_names})
            title = 'Tissue/Basket'
            feature = 'tissue'
            subheader = "Inferred basket response"
        elif feature == "clusters":
            len_colors = len(self.clusters_names)
            inferred_cluster = np.mean(stacked.cluster_p.values, axis=1)
            df = pd.DataFrame({'prob': inferred_cluster, 'cluster': self.clusters_names})
            title = 'Clusters'
            feature = 'cluster'
            subheader = "Inferred cluster response"
        alt_ver_barplot(df, feature, 'prob', len_colors, title, "Inferred response probability", feature, subheader, "Inferred", [feature, 'prob'])
