import gzip
import pickle

import numpy as np
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "pyBasket";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hideRows():
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    return hide_table_row_index
def readPickle(pick_file):
    try:
        with gzip.GzipFile(pick_file, 'rb') as f:
            return pickle.load(f)
    except OSError:
        logger.warning('Old, invalid or missing pickle in %s. '
                       'Please regenerate this file.' % pick_file)
        raise

def defaultPlot():
    plt.xlabel("PC1", fontsize=12)  # X-axis label font size
    plt.ylabel("PC2", fontsize=12)
    plt.xticks(fontsize=12)  # X-axis tick label font size
    plt.yticks(fontsize=12)

def defaultPlot_leg(title, ax):
    plt.xlabel("PC1", fontsize=12)  # X-axis label font size
    plt.ylabel("PC2", fontsize=12)
    plt.xticks(fontsize=12)  # X-axis tick label font size
    plt.yticks(fontsize=12)
    ax.legend(title=title, title_fontsize=12, fontsize=12, bbox_to_anchor=(1.2, 1), markerscale=0.5)

class Results():

    def __init__(self,pickle_data, name):
        self.file_name = name
        self.expr_df_filtered = pickle_data[list(pickle_data)[0]]
        self.expr_df_selected = pickle_data[list(pickle_data)[1]]
        self.drug_response = pickle_data[list(pickle_data)[2]]
        self.class_labels = pickle_data[list(pickle_data)[3]]
        self.cluster_labels = pickle_data[list(pickle_data)[4]]
        self.patient_df = pickle_data[list(pickle_data)[5]]

    def setClusters(self):
        clusters = list(self.patient_df['cluster_number'].unique())
        clusters = sorted(clusters)
        return clusters

    def setBaskets(self):
        tissues = list(self.patient_df['tissues'].unique())
        return tissues

    def fileInfo(self):

        num_samples = len(self.expr_df_filtered.axes[0])
        num_feat_filt = len(self.expr_df_filtered.axes[1])
        num_feat_select = len(self.expr_df_selected.axes[1])
        clusters = len(np.unique(self.cluster_labels))
        tissues = len(np.unique(self.class_labels))
        name = self.file_name
        info = {'File name: ': {'information': name},'Number of samples: ': {'information': num_samples}, 'Num. transcripts (after filtering): ': {'information': num_feat_filt},
                'Num. transcripts (after feature selection): ': {'information': num_feat_select}, 'Num.clusters: ': {'information': clusters}, 'Num. tissues/baskets: ' : {'information': tissues} }
        df = pd.DataFrame(data=info).T
        style = df.style.hide_index()
        style.hide_columns()
        return st.dataframe(df,use_container_width = True)

    def count_plot(self,feature, title, x_lab, response):
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)

        if response == True:
            hue = self.patient_df["responsive"]
            palette = ["#F72585", "#4CC9F0"]
        else:
            hue = None
            palette = sns.color_palette("pastel",25)
        ax = sns.countplot(x=self.patient_df[feature],hue = hue, palette = palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.xlabel(x_lab)
        plt.ylabel("Number of samples")

        return fig

    def AAC_plot(self,feature, title, x_lab, response):
        fig = plt.figure(figsize=(12, 6))
        x = feature
        y = self.patient_df["responses"]
        plt.title(title)
        if response == True:
            hue = self.patient_df["responsive"]
            palette = ["#F72585", "#4CC9F0"]
        else:
            hue = None
            palette = sns.color_palette("pastel",25)
        ax = sns.boxplot(data = self.patient_df,x= x, y = y, hue = hue, palette = palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.xlabel(x_lab)
        plt.ylabel("AAC response")
        return fig

    def raw_data_count(self,feature,x_variable, response):
        self.patient_df['Number of samples'] = 1
        if response == True:
            features = ["responsive", feature]
            columns = ["responsive",x_variable, "Number of samples"]
        else:
            features = feature
            columns = [x_variable, "Number of samples"]
        raw_count = self.patient_df.groupby(features).size().to_frame(name = 'count').reset_index()
        raw_count.columns = columns
        return raw_count

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

    def non_group_plot(self, feature):
        if feature == None:
            hue = None
        else:
            hue = feature
        fig = plt.figure(figsize=(10, 6))
        x = np.arange(298)
        ax = sns.scatterplot(data=self.patient_df, x=x, y="responses", hue=hue,
                             palette=sns.color_palette("pastel", 25))
        plt.title("AAC response per sample")
        ax.legend(title=feature, title_fontsize=12, fontsize=12, bbox_to_anchor=(1.2, 1), markerscale=0.5)
        plt.xlabel("Sample index")
        plt.ylabel("AAC response")
        return fig


class Analysis():

    def __init__(self, Results):
        self.expr_df_filtered = Results.expr_df_filtered
        self.expr_df_selected = Results.expr_df_selected
        self.drug_response = Results.drug_response
        self.class_labels = Results.class_labels
        self.cluster_labels = Results.cluster_labels
        self.patient_df = Results.patient_df

    def main_PCA(self, feature):
        rawX = self.expr_df_selected
        y = self.patient_df[feature]
        # data scaling
        x_scaled = StandardScaler().fit_transform(rawX)
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(
            data=pca_features,
            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
        pca_df.index = rawX.index
        pca_df[feature] = y
        variance = pca.explained_variance_
        return pca_df, variance

    def advanced_PCA(self, df):
        # data scaling
        y = self.patient_df["responsive"]
        x_scaled = StandardScaler().fit_transform(df)
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(
            data=pca_features,
            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
        pca_df.index = df.index
        pca_df["responsive"] = y
        variance = pca.explained_variance_
        return pca_df, variance

    def plot_PCA(self, pcaDF, feature):
        if feature == "responsive":
            palette = ["#F72585", "#4CC9F0"]
        else:
            palette = sns.color_palette("pastel",25)
        pc1_values = pcaDF['PC1']
        pc2_values = pcaDF['PC2']
        fig = plt.figure(figsize=(10, 6))
        fig.subplots_adjust(right=0.63, top=1)
        ax = sns.scatterplot(x=pc1_values, y=pc2_values,hue=pcaDF[feature], s=30, palette=palette)
        ax.set(title='PCA decomposition')
        defaultPlot_leg(feature, ax)
        return fig

    def showRawData_PCA(self, pcaDF, variance):
        var = {'Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 'Var explained': variance}
        var_df = pd.DataFrame(var)
        pca_df = pcaDF[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
        return pca_df,var_df

    def findSubgroup(self,feature,subgroup):
        transcriptomics = self.expr_df_selected
        sub_patients = self.patient_df[self.patient_df[feature]==subgroup]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcriptomics.loc[indexes]
        return sub_transcript

    def findInteraction(self,cluster,basket):
        transcriptomics = self.expr_df_selected
        sub_patients = self.patient_df[(self.patient_df['cluster_number'] == cluster) & (self.patient_df['tissues'] == basket)]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcriptomics.loc[indexes]
        num = len(sub_transcript)
        return sub_transcript, num

    def heatmapNum(self,results,x_highlight=None, y_highlight=None):
        clusters = results.setClusters()
        baskets = results.setBaskets()
        x_highlight = clusters.index(x_highlight)
        y_highlight = baskets.index(y_highlight)
        data = []
        for basket in baskets:
            clus = []
            for cluster in clusters:
                subgroup, num = self.findInteraction(cluster,basket)
                clus.append(num)
            data.append(clus)
        df = pd.DataFrame(data, baskets,clusters)
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data=df, cmap='Blues', yticklabels='auto')
        plt.title('Number of samples in basket')
        plt.xlabel('Clusters')
        plt.ylabel('Baskets')
        plt.yticks(fontsize=8)
        if x_highlight is not None and y_highlight is not None:
            plt.gca().add_patch(
                plt.Rectangle((x_highlight, y_highlight), 1, 1, fill=False, edgecolor='red', lw=3))
        return fig

    def heatmapTranscripts(self,df):
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data=df, cmap = "RdBu_r", yticklabels='auto')
        return fig

    def heatmapResponse(self,results,x_highlight=None, y_highlight=None):
        clusters = results.setClusters()
        baskets = results.setBaskets()
        x_highlight = clusters.index(x_highlight)
        y_highlight = baskets.index(y_highlight)
        data = []
        for basket in baskets:
            response = []
            for cluster in clusters:
                sub = len(self.patient_df[(self.patient_df['cluster_number'] == cluster) & (self.patient_df['tissues'] == basket)& (self.patient_df['responsive'] == 1)])
                response.append(sub)
            data.append(response)
        df = pd.DataFrame(data, baskets,clusters)
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data=df, cmap='Blues', yticklabels='auto')
        plt.title('Number of responsive samples per basket')
        plt.xlabel('Clusters')
        plt.ylabel('Baskets')
        plt.yticks(fontsize=8)
        if x_highlight is not None and y_highlight is not None:
            plt.gca().add_patch(
                plt.Rectangle((x_highlight, y_highlight), 1, 1, fill=False, edgecolor='red', lw=3))
        return fig
