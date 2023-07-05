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
from common import savePlot,saveTable,alt_ver_barplot
import altair as alt
from explorer import Data
import arviz as az
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

genes_path = os.path.join('..', 'pyBasket/Data', 'Entrez_to_Ensg99.mapping_table.tsv')

class DE():

    def __init__(self, Results):
        self.expr_df_filtered = Results.expr_df_filtered
        self.expr_df_selected = Results.expr_df_selected
        self.drug_response = Results.drug_response
        self.class_labels = Results.class_labels
        self.cluster_labels = Results.cluster_labels
        self.patient_df = Results.patient_df
        self.stacked_posterior = Results.stacked_posterior

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

    def samplesCount(self,subgroup):
        fulldf = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        feature = fulldf['responsive'].values
        fig = plt.figure(figsize=(2, 2))  # Set the size of the figure
        ax = sns.countplot(data=subgroup, x=feature, palette= ["#F72585", "#4CC9F0"])  # Use a specific color palette
        ax.bar_label(ax.containers[0], fontsize=4)
        plt.xlabel('Responsive', fontsize=5)  # Add x-axis label
        plt.ylabel('Count',fontsize=5)
        plt.xticks(fontsize=5)  # Set the font size of x-axis tick labels
        plt.yticks(fontsize=5)
        plt.title('Count of Responsive vs Non-responsive samples',fontsize=5)
        return fig

    def responseSamples(self,subgroup):
        fulldf = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        fulldf = fulldf[['tissues', 'responses', 'cluster_number', 'responsive']]
        fulldf = fulldf.sort_values(by='responses')
        fulldf.index.name = 'Sample'
        fulldf['responsive'] = fulldf['responsive'] == 1
        return fulldf


class heatMap(DE):
    def __init__(self, Results):
        super().__init__(Results)
        self.num_samples = None

    def heatmapNum(self,results):
        clusters = results.clusters_names
        baskets = results.baskets_names
        data = []
        for basket in baskets:
            clus = []
            for cluster in clusters:
                subgroup, num = self.findInteraction(cluster,basket)
                clus.append(num)
            data.append(clus)
        df = pd.DataFrame(data, baskets,clusters)
        self.num_samples = df
        return df

    def heatmapTranscripts(self,df):
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data=df, cmap="RdBu_r", yticklabels=True)
        plt.title('Transcriptional expression per sample')
        plt.xlabel('Transcripts')
        plt.ylabel('Samples')
        plt.yticks(fontsize=9)
        plt.xticks(fontsize=9)
        return fig

    def heatmapResponse(self, results):
        clusters = results.clusters_names
        baskets = results.baskets_names
        data = []
        for basket in baskets:
            response = []
            for cluster in clusters:
                sub = len(self.patient_df[(self.patient_df['cluster_number'] == cluster) & (
                            self.patient_df['tissues'] == basket) & (self.patient_df['responsive'] == 1)])
                response.append(sub)
            data.append(response)
        df = pd.DataFrame(data, baskets, clusters)
        return df

    def HM_inferredProb(self,results):
        basket_coords, cluster_coords = results.baskets_names,results.clusters_names
        stacked = self.stacked_posterior
        inferred_mat = np.mean(stacked.joint_p.values, axis=2)
        inferred_df = pd.DataFrame(inferred_mat, index=basket_coords, columns=cluster_coords)
        return inferred_df

    def heatmap_interaction(self, results, df, title, num_Sum, x_highlight=None, y_highlight=None):
        heatMap.heatmapNum(self, results)
        x_highlight = results.clusters_names.index(x_highlight)
        y_highlight = results.baskets_names.index(y_highlight)
        fig = plt.figure(figsize=(10, 10))
        ax = sns.heatmap(data=df, cmap='Blues', yticklabels='auto')
        plt.title(title)
        plt.xlabel('Clusters')
        plt.ylabel('Baskets')
        plt.yticks(fontsize=8)

        for i, c in enumerate(self.num_samples.columns):
            for j, v in enumerate(self.num_samples[c]):
                if v >= num_Sum:
                    ax.text(i + 0.5, j + 0.5, '★', color='gold', size=20, ha='center', va='center')
        if x_highlight is not None and y_highlight is not None:
            plt.gca().add_patch(
                plt.Rectangle((x_highlight, y_highlight), 1, 1, fill=False, edgecolor='red', lw=3))

        plt.show()
        return fig


class Analysis(Data):
    def __init__(self, file,name):
        super().__init__(file,name)
        self.pca_df = None
        self.pca_variance = None
        self.pca_adv = None
        self.pca_adv_var = None
        self.num_samples = None

    def findSubgroup(self,feature,subgroup):
        transcripts = self.expr_df_selected
        sub_patients = self.patient_df[self.patient_df[feature] == subgroup]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcripts.loc[indexes]
        return sub_transcript

    def findInteraction(self,cluster,basket):
        transcripts = self.expr_df_selected
        sub_patients = self.patient_df[(self.patient_df['cluster_number'] == cluster) & (self.patient_df['tissues'] == basket)]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcripts.loc[indexes]
        num = len(sub_transcript)
        return sub_transcript, num

    def samplesCount(self,subgroup):
        fulldf = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        feature = fulldf['responsive'].values
        df_grouped = fulldf.groupby(['responsive']).size().reset_index(name='Count')
        print(df_grouped)
        alt_ver_barplot(df_grouped, "responsive", 'Count', 2, "Response", "Number of samples", "responsive", "Samples responsive vs non-responsive",
                        "NS_Inter", ["responsive", 'Count'])

    def responseSamples(self,subgroup):
        fulldf = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        fulldf = fulldf[['tissues', 'responses', 'cluster_number', 'responsive']]
        fulldf = fulldf.sort_values(by='responses')
        fulldf.index.name = 'Sample'
        fulldf['responsive'] = fulldf['responsive'] == "Responsive"
        st.dataframe(fulldf, use_container_width=True)

    @staticmethod
    def showRawData_PCA(df, var):
        var = {'Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 'Var explained': var}
        var_df = pd.DataFrame(var)
        pca_df = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
        return pca_df, var_df

    def main_PCA(self, feature):
        rawX = self.expr_df_selected
        y = self.patient_df[feature]
        x_scaled = StandardScaler().fit_transform(rawX)
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(
            data=pca_features,
            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
        pca_df.index = rawX.index
        pca_df[feature] = y
        variance = pca.explained_variance_
        self.pca_df = pca_df
        self.pca_variance = variance

    def infoPCA(self, feature):
        info = {'Technique: ': {'information': 'Principal Component Analysis'}, 'Feature: ': {'information': feature},
                'Number of components: ': {'information': 5}}
        df = pd.DataFrame(data=info).T
        style = df.style.hide_index()
        style.hide_columns()
        return st.dataframe(df, use_container_width=True)

    def plot_PCA(self,feature,adv):
        df = self.pca_adv if adv == True else self.pca_df
        if feature == "responsive":
            palette = ["#F72585", "#4CC9F0"]
        else:
            palette = sns.color_palette("Paired",25).as_hex()
        base = alt.Chart(df, title = "Principal Component Analysis").mark_circle(size=60).encode(
            x='PC1',
            y='PC2',
            color=feature+':N'
        ).interactive().properties(height=650).configure_range(
        category=alt.RangeScheme(palette))
        savePlot(base, "PCA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)

    def PCA_analysis(self, feature, RawD):
        Analysis.main_PCA(self, feature)
        if RawD is True:
            pcaDF, var_df = Analysis.showRawData_PCA(self.pca_df, self.pca_variance)
            col11, col12 = st.columns((2, 3))
            with col11:
                st.write('##### Variance explained by component')
                saveTable(self.pca_variance, "var")
                st.dataframe(var_df, use_container_width=True)
            with col12:
                st.write('##### PCA results')
                saveTable(self.pca_df, "PCA")
                st.dataframe(pcaDF, use_container_width=True)
        else:
            Analysis.plot_PCA(self, feature, adv=False)

    def advanced_PCA(self, df):
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
        self.pca_adv = pca_df
        self.pca_adv_var = variance

    def adv_PCA(self,sub_df, RawD):
        try:
            Analysis.advanced_PCA(self, sub_df)
            if RawD is True:
                pcaDF, var_df = Analysis.showRawData_PCA(self.pca_adv, self.pca_adv_var)
                col11, col12 = st.columns((2, 3))
                with col11:
                    st.write('##### Variance explained by component')
                    saveTable(self.pca_adv_var, "var")
                    st.dataframe(var_df, use_container_width=True)
                with col12:
                    st.write('##### PCA results')
                    saveTable(self.pca_adv, "PCA")
                    st.dataframe(pcaDF, use_container_width=True)
            else:
                Analysis.plot_PCA(self, "responsive",adv= True)
        except:
            st.warning("Not enough samples. Please try a different combination.")
