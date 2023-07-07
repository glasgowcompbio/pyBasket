import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import savePlot,saveTable,alt_ver_barplot
import altair as alt
from explorer import Data
from scipy.cluster import hierarchy


genes_path = os.path.join('..', 'pyBasket/Data', 'Entrez_to_Ensg99.mapping_table.tsv')

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
        df = df.reset_index()
        if feature == "responsive":
            palette = ["#F72585", "#4CC9F0"]
        else:
            palette = sns.color_palette("Paired",25).as_hex()
        base = alt.Chart(df, title = "Principal Component Analysis").mark_circle(size=60).encode(
            x='PC1',
            y='PC2',
            color=feature+':N', tooltip = ['index', feature]
        ).interactive().properties(height=650).configure_range(
        category=alt.RangeScheme(palette))
        savePlot(base, "PCA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.caption("Axis show the first and second principal components (PC1, PC2) that capture the most variation in the expression level of transcripts."
                   "Position of each data point is the scores on PC1 and PC2.")

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

class heatMap(Analysis):
    def __init__(self, file,name):
        super().__init__(file,name)
        self.num_samples = None

    def heatmapNum(self):
        clusters = self.clusters_names
        baskets = self.baskets_names
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
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        distance_matrix = hierarchy.distance.pdist(df_scaled.values, metric='euclidean')
        Z = hierarchy.linkage(distance_matrix, method='ward')
        labels = df.index.values
        fig = plt.figure(figsize=(10, 10))
        dendrogram = hierarchy.dendrogram(Z, labels=labels, orientation='right', color_threshold=0)
        reordered_df = df_scaled.iloc[dendrogram['leaves']]
        reordered_df.index = df.index
        sns.heatmap(reordered_df, cmap="RdBu_r", cbar_kws={'label': 'Expression Level'},yticklabels=True)
        plt.title('Transcriptional expression per sample')
        plt.xlabel('Transcripts')
        plt.ylabel('Samples')
        plt.xticks(fontsize=9)
        return fig

    def heatmapResponse(self):
        clusters = self.clusters_names
        baskets = self.baskets_names
        data = []
        for basket in baskets:
            response = []
            for cluster in clusters:
                sub = len(self.patient_df[(self.patient_df['cluster_number'] == cluster) & (
                            self.patient_df['tissues'] == basket) & (self.patient_df['responsive'] == "Responsive")])
                response.append(sub)
            data.append(response)
        df = pd.DataFrame(data, baskets, clusters)
        return df

    def HM_inferredProb(self):
        basket_coords, cluster_coords = self.baskets_names,self.clusters_names
        stacked = self.stacked_posterior
        inferred_mat = np.mean(stacked.joint_p.values, axis=2)
        inferred_df = pd.DataFrame(inferred_mat, index=basket_coords, columns=cluster_coords)
        return inferred_df

    def heatmap_interaction(self, df,title, num_Sum, x_highlight=None, y_highlight=None):
        x_highlight = self.clusters_names.index(x_highlight)
        y_highlight = self.baskets_names.index(y_highlight)
        fig = plt.figure(figsize=(10, 10))
        ax = sns.heatmap(data=df, cmap='Blues', yticklabels='auto')
        plt.title(title)
        plt.xlabel('Clusters')
        plt.ylabel('Baskets')
        plt.yticks(fontsize=8)

        for i, c in enumerate(df):
            for j, v in enumerate(df[c]):
                if v >= num_Sum:
                    ax.text(i + 0.5, j + 0.5, '★', color='gold', size=20, ha='center', va='center')
        if x_highlight is not None and y_highlight is not None:
            plt.gca().add_patch(
                plt.Rectangle((x_highlight, y_highlight), 1, 1, fill=False, edgecolor='red', lw=3))

        plt.show()
        return fig