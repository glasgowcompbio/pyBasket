import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


class SameBasketClustering():
    def __init__(self, groups):
        self.groups = groups

        all_features = []
        all_classes = []
        all_responses = []
        for group in self.groups:
            features = group.features
            all_features.append(features)

            N = group.features.shape[0]
            group_class = [group.idx] * N
            all_classes.extend(group_class)

            all_responses.extend(group.responses)

        self.features = np.concatenate(all_features)
        self.classes = np.array(all_classes)
        self.responses = np.array(all_responses)
        self.dist = None
        self.clusters = None

    def PCA(self, n_components=5, plot_PCA=False):
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(self.features)
        pc1_values = pcs[:, 0]
        pc2_values = pcs[:, 1]

        if plot_PCA:
            sns.set_context('poster')
            plt.figure(figsize=(5, 5))
            g = sns.scatterplot(x=pc1_values, y=pc2_values, hue=self.classes, palette='bright')
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
            plt.show()
            print('PCA explained variance', pca.explained_variance_ratio_.cumsum())

    def compute_distance_matrix(self):
        # a custom function that computes:
        # the Euclidean distance if p1 and p2 are in different baskets
        # or, returns 0 distance if p1 and p2 are in the same basket
        def mydist(p1, p2, c1, c2):
            if c1 == c2:
                return 0
            diff = p1 - p2
            return np.vdot(diff, diff) ** 0.5

        N = self.features.shape[0]
        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                p1 = self.features[i]
                p2 = self.features[j]
                c1 = self.classes[i]
                c2 = self.classes[j]
                dist[i, j] = mydist(p1, p2, c1, c2)
        self.dist = dist
        return self.dist

    def plot_distance_matrix(self):
        sns.set_context('poster')
        plt.matshow(self.dist)
        plt.colorbar()
        plt.show()

    def cluster(self, plot_dendrogram=False, max_d=70):
        condensed_dist = squareform(self.dist)
        Z = linkage(condensed_dist, method='ward')
        clusters = fcluster(Z, max_d, criterion='distance')

        if plot_dendrogram:
            plt.figure(figsize=(30, 10))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('sample index')
            plt.ylabel('distance')
            dendrogram(
                Z,
                leaf_rotation=90.,  # rotates the x axis labels
                leaf_font_size=18,  # font size for the x axis labels
            )
            plt.ylim([-5, 120])
            plt.axhline(y=max_d, color='r', linestyle='--')
            for text in plt.gca().get_xticklabels():
                pos = int(text.get_text())
                if self.responses[pos] == 1:
                    text.set_color('blue')
                else:
                    text.set_color('red')
            plt.show()

        self.clusters = clusters
        return self.clusters

    def to_df(self):
        N = self.responses.shape[0]
        data = []
        for n in range(N):
            row = [self.responses[n], self.classes[n], self.clusters[n], self.features[n]]
            data.append(row)
        df = pd.DataFrame(data, columns=['response', 'class', 'cluster', 'features'])
        return df

    def __repr__(self):
        return 'ClusteringData: %s %s' % (str(self.features.shape), str(self.classes.shape))
