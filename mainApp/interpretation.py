import gzip
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
from processing import readPickle, Results, add_logo, defaultPlot_leg, Analysis, hideRows, defaultPlot_leg
from scipy.spatial.distance import cdist

if "data" in st.session_state:
    data = st.session_state["data"]

def Kmedoids():
        rawX = data.expr_df_selected
        #x_scaled = StandardScaler().fit_transform(rawX)
        y = data.patient_df["responsive"]
        pca_2 = PCA(n_components=5)
        X = pca_2.fit_transform(rawX)
        cobj = KMedoids(n_clusters=5).fit(X)
        labels = cobj.labels_
        unique_labels = set(labels)
        colors = [
            plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
        ]
        fig = plt.figure(figsize=(10, 6))
        for k, col in zip(unique_labels, colors):
            class_member_mask = labels == k

            xy = X[class_member_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,label=k
            )
        plt.plot(
            cobj.cluster_centers_[:, 0],
            cobj.cluster_centers_[:, 1],
            "o",
            markerfacecolor="cyan",
            markeredgecolor="k",
            markersize=6,
        )
        for i, label in enumerate(unique_labels):
            plt.annotate(label, (cobj.cluster_centers_[i, 0],  cobj.cluster_centers_[i, 1]), textcoords="offset points",
                        xytext=(0, 10), ha='center', zorder=10)
        plt.legend()
        plt.title("Real KMedoids clustering. Medoids are represented in cyan.")
        st.pyplot(fig)


class Prototypes():
    def __init__(self, Results):
        self.expr_df_selected = Results.expr_df_selected
        self.class_labels = Results.class_labels
        self.cluster_labels = Results.cluster_labels
        self.subgroup = None
        self.patient_df = Results.patient_df
        self.pseudo_medoids = []

    @staticmethod
    def pseudoMedoids(self,df,feature):
        rawX = df
        rawX = rawX.reset_index()
        rawX = rawX.drop(['index'], axis=1)
        rawX["labels"] = feature
        rawX["labels"] = rawX["labels"].astype('category').cat.codes
        unique_labels = np.unique(rawX["labels"])
        labels = rawX["labels"]
        pseudo_medoids = []
        for label in unique_labels:
            # Filter data points with the current label
            label_points = rawX[labels == label]
            # Calculate distance between every point in the cluster
            distances = cdist(label_points, label_points, metric='euclidean')
            # Sum distances for each point
            total_distances = np.sum(distances, axis=1)
            # Find the index of the data point with the minimum sum of distances: the one that is supposed to be the closest to all points
            medoid_index = np.argmin(total_distances)
            # get the sample corresponding to the medoid
            medoid = label_points.iloc[medoid_index]
            pseudo_medoids.append(medoid)
        indexes = []
        for x in pseudo_medoids:
            indexes.append(x.name)
        x_scaled = StandardScaler().fit_transform(rawX)
        pca_2 = PCA(n_components=5)
        X = pca_2.fit_transform(x_scaled)
        X_df = pd.DataFrame(
            data=X,
            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
        X_df.index = rawX.index
        X_df['labels'] = labels
        sample_medoids = X_df.loc[indexes]
        self.pseudo_medoids = pseudo_medoids
        return X, sample_medoids


    @staticmethod
    def plotMedoids(X,sample_medoids, feature):
        num_palette = len(np.unique(feature))
        labels = np.unique(feature)
        #palette = sns.color_palette("pastel", num_palette)
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        palette = ["#F72585", "#4CC9F0"] if num_palette<3 else sns.color_palette("pastel", num_palette)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=feature, s=30, palette=palette, ax=ax)
        sns.scatterplot(x=np.array(sample_medoids)[:, 0], y=np.array(sample_medoids)[:, 1], marker='o', s=100, c='black', ax=ax)
        plt.xlabel('X1')
        plt.ylabel('X2')
        for i, label in enumerate(labels):
            ax.annotate(label, (np.array(sample_medoids)[i, 0], np.array(sample_medoids)[i, 1]), textcoords="offset points",
                        xytext=(0, 10), ha='center', zorder=10)
        plt.legend(loc='lower left', bbox_to_anchor=(1.1,0))
        #ax.legend(labels=np.unique(feature), fontsize=12, markerscale=0.5, loc='lower center',
          #        bbox_to_anchor=(0.5, -0.3), ncol=6)
        plt.title('K-means Pseudo-Medoids')
        return fig
    @staticmethod
    def saveplot(fig, feature):
        if st.button('Save Plot', key="plot_Prototype"):  # Update the key to a unique value
            fig.savefig('plot_prototype_' + feature + '.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")

    @staticmethod
    def savedf(tab, feature):
        if st.button('Save Data', key="table_Prototype"):  # Update the key to a unique value
            tab.to_csv('table_prototypes_' + feature + '.csv')
            st.info('Data saved as .csv in working directory', icon="ℹ️")
        else:
            st.write("")
    @staticmethod
    def showMedoids(self,feature):
        pseudo_medoids = self.pseudo_medoids
        rawX = self.expr_df_selected.reset_index()
        pseudo_medoids_df = pd.concat(pseudo_medoids, axis=1)
        pseudo_medoids_df = pseudo_medoids_df.T
        pseudo_medoids_df.drop(labels=['labels'], axis=1, inplace=True)
        pseudo_medoids_df.insert(0, 'Group', np.unique(feature))
        indexes = rawX.loc[pseudo_medoids_df.index]['index']
        pseudo_medoids_df.index = indexes
        pseudo_medoids_df.index.name = 'Sample'
        return pseudo_medoids_df

    def findPrototypes(self, option):
        feature = self.cluster_labels if option == 'Clusters' else self.class_labels
        data,sampleMedoids = Prototypes.pseudoMedoids(self,self.expr_df_selected,feature)
        plot = Prototypes.plotMedoids(data,sampleMedoids,feature)
        Prototypes.saveplot(plot,option)
        st.pyplot(plot)
        table = Prototypes.showMedoids(self,feature)
        st.subheader("Prototype samples")
        Prototypes.savedf(table, option)
        st.dataframe(table)

    def findPrototypes_sub(self,subgroup):
        self.subgroup = subgroup
        fulldf = pd.merge(self.patient_df, self.subgroup, left_index=True, right_index=True)
        feature = fulldf['responsive'].values
        fulldf = fulldf.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'], axis=1)
        data, sampleMedoids = Prototypes.pseudoMedoids(self,fulldf, feature)
        plot = Prototypes.plotMedoids(data, sampleMedoids, feature)
        st.pyplot(plot)
        st.subheader("Prototype samples")
        table = Prototypes.showMedoids(self, feature)
        st.dataframe(table)






