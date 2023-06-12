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

def Kmedoids():
    if "data" in st.session_state:
        data = st.session_state["data"]
        analysis_data = Analysis(data)
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
                markersize=6,
            )
        plt.plot(
            cobj.cluster_centers_[:, 0],
            cobj.cluster_centers_[:, 1],
            "o",
            markerfacecolor="cyan",
            markeredgecolor="k",
            markersize=6,
        )

        plt.title("Real KMedoids clustering. Medoids are represented in cyan.")
        st.pyplot(fig)

def pseudoMedoids():
    if "data" in st.session_state:
        data = st.session_state["data"]
        rawX = data.expr_df_selected
        rawX = rawX.reset_index()
        rawX= rawX.drop(['index'], axis=1)
        rawX["labels"] = data.cluster_labels
        #x_scaled = StandardScaler().fit_transform(rawX)
        kmeans = KMeans(n_clusters=5,random_state=42)
        kmeans.fit(rawX)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        distances = kmeans.transform(rawX)
        pseudo_medoids = []
        for cluster_label in range(5):
            cluster_distances = distances[labels == cluster_label, cluster_label]
            pseudo_medoid_index = np.argmin(cluster_distances)
            pseudo_medoid = rawX[labels == cluster_label].iloc[pseudo_medoid_index]
            pseudo_medoids.append(pseudo_medoid)
        indexes = []
        for x in pseudo_medoids:
            indexes.append(x.name)
        pca_2 = PCA(n_components=5)
        X = pca_2.fit_transform(rawX)
        X_df = pd.DataFrame(
            data=X,
            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
        medoids = X_df.loc[indexes]
        palette = sns.color_palette("pastel", 25)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, s=30, palette=palette, ax = ax)
        #ax.set(title='PCA decomposition')
        sns.scatterplot(x= np.array(medoids)[:, 0],y= np.array(medoids)[:, 1], marker='o', s=100, c='red',ax = ax)
        plt.xlabel('X1')
        plt.ylabel('X2')
        defaultPlot_leg("Cluster",ax)
        plt.title('K-means Pseudo-Medoids')
        st.pyplot(fig)