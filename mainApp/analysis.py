import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import savePlot,saveTable,alt_ver_barplot, colours
import altair as alt
from explorer import Data
from scipy.cluster import hierarchy

"""
subclass: Analysis: creates an analysis object for PCA dimensionality reduction and other small analysis functions
Input: an initialised Data object.
"""
class Analysis(Data):
    def __init__(self, file,name):
        super().__init__(file,name)
        self.pca_df = None
        self.pca_variance = None
        self.pca_adv = None
        self.pca_adv_var = None
        self.num_samples = None

    #Function to find the set of samples in a basket-cluster interaction
    def findInteraction(self,cluster,basket):
        transcripts = self.expr_df_selected
        sub_patients = self.patient_df[(self.patient_df['cluster_number'] == cluster) & (self.patient_df['tissues'] == basket)]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcripts.loc[indexes]
        num = len(sub_transcript)
        return sub_transcript, num

    #Function to show the number of responsive/non-responsive samples in a basket-cluster interaction
    def samplesCount(self,subgroup):
        fulldf = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        df_grouped = fulldf.groupby(['responsive']).size().reset_index(name='Count')
        alt_ver_barplot(df_grouped, "responsive", 'Count', 2, "Response", "Number of samples", "responsive", "Samples responsive vs non-responsive",
                        "NS_Inter", ["responsive", 'Count'])
        st.caption("Number of responsive and non-responsive samples in the interaction.")

    #Function to show information related to the samples in a basket-cluster interaction
    def responseSamples(self,subgroup):
        fulldf = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        fulldf = fulldf[['tissues', 'responses', 'cluster_number', 'responsive']]
        fulldf = fulldf.sort_values(by='responses')
        fulldf.index.name = 'Sample'
        fulldf['responsive'] = fulldf['responsive'] == "Responsive"
        saveTable(fulldf, "Interaction")
        st.dataframe(fulldf, use_container_width=True)

    #Static method to show PCA results in a table
    @staticmethod
    def showRawData_PCA(df, var):
        var = {'Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 'Var explained': var}
        var_df = pd.DataFrame(var)
        pca_df = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
        return pca_df, var_df

    #Function to perform PCA in all samples
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

    #Function to plot PCA results
    def plot_PCA(self,feature,adv):
        df = self.pca_adv if adv == True else self.pca_df
        df = df.reset_index()
        if feature == "responsive":
            palette = colours(2)
        else:
            palette = colours(25)
        base = alt.Chart(df, title = "Principal Component Analysis").mark_circle(size=60).encode(
            x='PC1',
            y='PC2',
            color=feature+':N', tooltip = ['index', feature]
        ).interactive().properties(height=500, width = 600).configure_range(
        category=alt.RangeScheme(palette))
        savePlot(base, "PCA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.caption("Axis show the first and second principal components (PC1, PC2) that capture the most variation in the expression level of transcripts."
                   "Position of each data point is the scores on PC1 and PC2.")

    #Function to show PCA results in a plot or in a table (if RawD is True)
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

    #Function to perform PCA on the samples in the selected basket-cluster interaction
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

    #Function to show PCA results in a plot or in a table (if RawD is True)
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

"""
subclass of Analysis: heatMap: creates Analysis sub-object to show and filter information shown in the heatmap with interactions between clusters and baskets
Input: an initialised Analaysis object.
"""
class heatMap(Analysis):
    def __init__(self, file,name):
        super().__init__(file,name)
        self.num_samples = None

    #Function to return dataframe with information about the number of samples
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

    #Function to show heatmap with the transcriptional expression of samples in the selected basket-cluster interaction
    def heatmapTranscripts(self,df):
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        distance_matrix = hierarchy.distance.pdist(df_scaled.values, metric='euclidean')
        Z = hierarchy.linkage(distance_matrix, method='ward')
        labels = df.index.values
        fig = plt.figure(figsize=(10, 10))
        dendrogram = hierarchy.dendrogram(Z, labels=labels, orientation='right', color_threshold=np.inf,no_plot=True)
        reordered_df = df_scaled.iloc[dendrogram['leaves']]
        reordered_df.index = df.index
        sns.heatmap(reordered_df, cmap="coolwarm", cbar_kws={'label': 'Expression Level'}, xticklabels=False)
        plt.title('Transcriptional expression per sample')
        plt.xlabel('Transcripts')
        plt.ylabel('Samples')
        plt.xticks(fontsize=9)
        return fig

    #Function to return dataframe with information about the number of responsive samples
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

    #Function to get the mean inferred joint probability from the pyBasket pipeline
    def HM_inferredProb(self):
        basket_coords, cluster_coords = self.baskets_names,self.clusters_names
        stacked = self.stacked_posterior
        inferred_mat = np.mean(stacked.joint_p.values, axis=2)
        inferred_df = pd.DataFrame(inferred_mat, index=basket_coords, columns=cluster_coords)
        return inferred_df

    #Function to show interactions heatmap with the chosen information, mark the selected basket-cluster interaction and filter interactions with
    #num_Sum number of samples
    def heatmap_interaction(self, df,title, num_Sum, x_highlight=None, y_highlight=None):
        x_highlight = self.clusters_names.index(x_highlight)
        y_highlight = self.baskets_names.index(y_highlight)
        fig = plt.figure(figsize=(10, 10))
        ax = sns.heatmap(data=df, cmap="coolwarm", yticklabels='auto')
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
        return fig

