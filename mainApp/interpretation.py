import gzip
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind
import mpld3
import streamlit.components.v1 as components
from sklearn_extra.cluster import KMedoids
from processing import readPickle, Results, add_logo, defaultPlot_leg, Analysis, hideRows, defaultPlot_leg
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import fdrcorrection
from lime import lime_tabular
from sklearn.inspection import permutation_importance
import sklearn
import shap
import plotly.graph_objects as go
import plotly.express as px
#from bioinfokit import analys, visuz
np.set_printoptions(suppress=True, precision=3)


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
        #fig_html = mpld3.fig_to_html(fig)
        #components.html(fig_html, height=670, width=1000)
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
        Prototypes.saveplot(plot, "subgroup")
        st.pyplot(plot)
        st.subheader("Prototype samples")
        table = Prototypes.showMedoids(self, feature)
        Prototypes.savedf(table, "subgroup")
        st.dataframe(table)

class DEA():
    def __init__(self, Results):
        self.expr_df_selected = Results.expr_df_selected
        self.class_labels = Results.class_labels
        self.cluster_labels = Results.cluster_labels
        self.patient_df = Results.patient_df
        self.df_group1 = None
        self.df_group2 = None
        self.subgroup = None
        self.ttest_res = None

    def selectGroups(self,option,feature):
        transcripts= self.expr_df_selected
        sub_patients = self.patient_df[self.patient_df[feature] == option]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcripts.loc[indexes]
        return sub_transcript

    @staticmethod
    def saveplot(fig, feature):
        if st.button('Save Plot', key="plot_DEA"):  # Update the key to a unique value
            fig.savefig('plot_' + feature + '.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")

    @staticmethod
    def savedf(tab, feature):
        if st.button('Save Data', key="table_DEA"):  # Update the key to a unique value
            tab.to_csv('table_' + feature + '.csv')
            st.info('Data saved as .csv in working directory', icon="ℹ️")
        else:
            st.write("")

    def ttest_results(self,df1,df2,pthresh,logthresh):
        ttest_results = []
        for column in df1.columns:
            t, p = ttest_ind(df1[column], df2[column])
            l2fc = np.mean(df1[column].values) - np.mean(df2[column].values)
            ttest_results.append((column, t, p, l2fc))
        dea_results = pd.DataFrame(ttest_results, columns=['Feature', 'T-Statistic', 'P-Value', 'LFC'])
        dea_results['Adjusted P-value'] = fdrcorrection(dea_results['P-Value'].values)[1]
        #_, dea_results['Adjusted P-value'],_, _ = fdrcorrection(dea_results['P-Value'].values)
                                                                     #method='fdr_bh')

        dea_results['Significant'] = (dea_results['Adjusted P-value'] < pthresh) & (abs(dea_results['LFC']) > logthresh)
        return dea_results

    def diffAnalysis_simple(self,option1, option2, feature,pthresh,logthresh):
        self.df_group1 = DEA.selectGroups(self,option1,feature)
        self.df_group2 = DEA.selectGroups(self,option2,feature)
        self.ttest_res = DEA.ttest_results(self,self.df_group1, self.df_group2,pthresh,logthresh)
        self.ttest_res.sort_values(by='Adjusted P-value', ascending=True)
        st.subheader("Volcano plot")
        fig = DEA.volcanoPlot(self,pthresh,logthresh)
        #fig_html = mpld3.fig_to_html(fig)
        DEA.saveplot(fig,"DEA")
        st.pyplot(fig)
        #components.html(fig_html, height=500, width=1200)

    def diffAnalysis_response(self,subgroup,pthresh, logthresh):
        subgroup_df = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        self.df_group1 = subgroup_df[subgroup_df["responsive"]==0]
        self.df_group2 = subgroup_df[subgroup_df["responsive"] == 1]
        self.df_group1 = self.df_group1.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'], axis=1)
        self.df_group2 = self.df_group2.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'],
                                             axis=1)
        self.ttest_res = DEA.ttest_results(self, self.df_group1, self.df_group2, pthresh, logthresh)
        fig = DEA.volcanoPlot(self, pthresh, logthresh)
        st.subheader("Volcano plot")
        DEA.saveplot(fig, "DEA:resp")
        st.pyplot(fig)

    def showResults(self,feature):
        st.subheader("Results")
        self.ttest_res['Adjusted P-value'] = self.ttest_res['Adjusted P-value'].apply('{:.6e}'.format)
        self.ttest_res['P-Value'] = self.ttest_res['P-Value'].apply('{:.6e}'.format)
        only_sig = st.checkbox('Show only significant transcripts')
        num_sig = len(self.ttest_res[self.ttest_res["Significant"] == True])
        if only_sig and num_sig >0:
            numShow = st.slider('Select transcripts to show', 0,)
            show = self.ttest_res[self.ttest_res["Significant"] == True][:numShow]
        elif only_sig:
            st.warning("No significant transcripts found.")
            show = None
        else:
            numShow = st.slider('Select transcripts to show', 0,len(self.ttest_res))
            show = self.ttest_res[:numShow]
        #print(self.ttest_res[self.ttest_res["Significant"] == True])
        DEA.savedf(show, feature)
        st.dataframe(show, use_container_width=True)
        st.caption("Ordered by most significantly different (highest adj p-value).")

    def diffAnalysis_inter(self,subgroup,pthresh,logthresh):
        indexes = subgroup.index
        filtered_df= self.expr_df_selected.drop(indexes)
        self.subgroup = subgroup
        self.ttest_res = DEA.ttest_results(self,self.subgroup,filtered_df,pthresh,logthresh)
        fig = DEA.volcanoPlot(self,pthresh,logthresh)
        st.subheader("Volcano plot")
        DEA.saveplot(fig, "DEA")
        st.pyplot(fig)
        #fig_html = mpld3.fig_to_html(fig)
        #components.html(fig_html, height=500, width=1200)

    def pPlot(self):
        fig =plt.figure(figsize=(9,5))
        self.ttest_res.reset_index()
        ax = sns.scatterplot(self.ttest_res,x= self.ttest_res.index,y = -np.log10(self.ttest_res['P-Value']), s = 15,hue='Significant', palette=['darkgrey',"#F72585"])
        plt.xlabel('Features')
        plt.ylabel('-log10(p-value)')
        ax.legend(title="Significance", title_fontsize=12, fontsize=12, bbox_to_anchor=(1.1,1),  markerscale=0.5)
        return fig

    def volcanoPlot(self, thresh, logthresh):
        df = self.ttest_res
        fig = plt.figure(figsize=(10, 5))
        plt.scatter(x=df['LFC'], y=df['P-Value'].apply(lambda x: -np.log10(x)), s=5,color="black")
        down = df[(df['LFC'] <= -logthresh) & (df['P-Value'] <= thresh)]
        up = df[(df['LFC'] >= logthresh) & (df['P-Value'] <= thresh)]

        plt.scatter(x=down['LFC'], y=down['P-Value'].apply(lambda x: -np.log10(x)), s=5, label="Down-regulated",
                    color="blue")
        plt.scatter(x=up['LFC'], y=up['P-Value'].apply(lambda x: -np.log10(x)), s=5, label="Up-regulated",
                    color="red")
        plt.xlabel("log2FC")
        plt.ylabel('-log10(p-value)')
        plt.axvline(x=-logthresh,color="grey",linestyle="--")
        plt.axvline(x=logthresh, color="grey", linestyle="--")
        #plt.axhline(y=-thresh, color="grey", linestyle="--")
        plt.axhline(y=-np.log10(thresh), color="grey", linestyle="--")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        return fig

    def infoTest(self,group1,group2,feature,pthresh,logthresh):
        size = len(self.ttest_res[self.ttest_res['Significant']==True])
        info = {'Test: ': {'information': 'T-test'}, 'Multi-sample correction: ': {'information': 'Benjamini/Hochberg'},
                'Groups compared: ': {'information': '{}: {} vs {}'.format(feature,group1,group2)},
                'P-value threshold: ': {'information': pthresh},'log2 FC threshold: ': {'information': logthresh},
                'Num. Significant transcripts: ': {'information': size}}
        df = pd.DataFrame(data=info).T
        style = df.style.hide_index()
        style.hide_columns()
        return st.dataframe(df, use_container_width=True)

class FI():
    def __init__(self, Results):
        self.expr_df_selected = Results.expr_df_selected
        self.class_labels = Results.class_labels
        self.cluster_labels = Results.cluster_labels
        self.patient_df = Results.patient_df
        self.importance = Results.importance_df
        self.drug_response = Results.drug_response
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []


    def plotImportance(self):
        importance_sort = self.importance.sort_values('importance_score', ascending=False)
        importance_vals=  importance_sort['importance_score'].values[:25]
        importance_feats = importance_sort.index[:25]
        fig = plt.figure(figsize=(12, 6))

        # Create the horizontal bar plot
        plt.barh(range(len(importance_vals)), importance_vals, align='center')
        plt.yticks(range(len(importance_vals)), importance_feats)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')

        return st.pyplot(fig)

    def prepareData(self, y):
        train_size = int(len(self.expr_df_selected) * .8)
        self.X_train, self.X_test = self.expr_df_selected.iloc[:train_size], self.expr_df_selected.iloc[train_size:]
        y_train, self.y_test = y.iloc[:train_size], y.iloc[
                                                                           train_size:]
        self.y_train = y_train.values.flatten()

    def limeInterpretation(self,n_features):
        FI.prepareData(self,self.patient_df["responsive"])
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        explainer = lime_tabular.LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, class_names=self.y_train,
                                                      discretize_continuous=True)
        exp = explainer.explain_instance(self.X_test.iloc[0], rf.predict_proba, num_features=n_features)
        fig = exp.as_pyplot_figure()

        return st.pyplot(fig)

    def permutationImportance(self):
        FI.prepareData(self, self.drug_response)
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100,random_state=42)
        rf.fit(self.X_train, self.y_train)

        # the permutation based importance
        perm_importance = permutation_importance(rf, self.X_test, self.y_test)

        sorted_idx = perm_importance.importances_mean.argsort()
        top_positive_indices = sorted_idx[-15:]

        # Get the indices of the top 50 negative values
        top_negative_indices = sorted_idx[:15]

        # Combining the indices of top positive and negative values
        top_indices = np.concatenate((top_negative_indices, top_positive_indices))

        # The top 50 values (both negative and positive) can be accessed using:
        #top_values = perm_importance.importances_mean[top_indices]
        fig = plt.figure(figsize = (10,8))
        plt.barh(self.X_train.columns[top_indices], perm_importance.importances_mean[top_indices])
        plt.xlabel("Permutation Importance")
        return st.pyplot(fig)

    def SHAP(self):
        FI.prepareData(self, self.drug_response)
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(self.expr_df_selected)
        return explainer, shap_values

    def SHAP_forces(self, index):
        explainer,values = FI.SHAP(self)
        fig = shap.force_plot(explainer.expected_value, values[index, :], self.expr_df_selected.iloc[index, :], matplotlib=True, show=False)
        return fig

    def SHAP_summary(self):
        explainer, values = FI.SHAP(self)
        fig, ax = plt.subplots()
        shap.summary_plot(values, self.expr_df_selected, show=False, plot_size=(8, 6), color='b')
        return fig








