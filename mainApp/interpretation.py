import gzip
import pickle
import statistics

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
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import fdrcorrection
from common import savePlot, saveTable
import altair as alt

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
        palette = ["#F72585", "#4CC9F0"] if num_palette<3 else sns.color_palette("Paired", num_palette).as_hex()
        df = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'class':feature})
        df3 = pd.merge(df, sample_medoids, left_index=True, right_index=True)
        scale = alt.Scale(domain=labels, range = palette)
        base = alt.Chart(df,title="Prototypes samples").mark_circle(size=100).encode(
            x='X',
            y='Y',color = alt.Color('class:N', scale = scale)
        ).interactive().properties(height=700, width = 550)
        plotMedoids =alt.Chart(df3).mark_point(filled=True, size=150).encode(
            x='PC1',
            y='PC2',color=alt.value('black')
        )
        labels = plotMedoids.mark_text(
                align='left',
                baseline='middle',
                dx=15
            ).encode(
            text='class'
            )
        base = base+plotMedoids+labels
        return base

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
        base = Prototypes.plotMedoids(data,sampleMedoids,feature)
        savePlot(base,option)
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        table = Prototypes.showMedoids(self,feature)
        st.subheader("Prototype samples")
        saveTable(table, option)
        st.dataframe(table)

    def findPrototypes_sub(self,subgroup):
        self.subgroup = subgroup
        fulldf = pd.merge(self.patient_df, self.subgroup, left_index=True, right_index=True)
        feature = fulldf['responsive'].values
        fulldf = fulldf.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'], axis=1)
        data, sampleMedoids = Prototypes.pseudoMedoids(self,fulldf, feature)
        plot,base = Prototypes.plotMedoids(data, sampleMedoids, feature)
        savePlot(plot, "subgroup")
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.subheader("Prototype samples")
        table = Prototypes.showMedoids(self, feature)
        saveTable(table, "subgroup")
        st.dataframe(table, use_container_width=True)

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
        self.transcripts = None

    def selectGroups(self,option,feature):
        transcripts= self.expr_df_selected
        sub_patients = self.patient_df[self.patient_df[feature] == option]
        indexes = list(sub_patients.index.values)
        sub_transcript = transcripts.loc[indexes]
        return sub_transcript

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
        dea_results = dea_results.sort_values(by=['Adjusted P-value'])
        dea_results['Significant'] = (dea_results['Adjusted P-value'] < pthresh) & (abs(dea_results['LFC']) > logthresh)
        return dea_results

    def diffAnalysis_simple(self,option1, option2, feature,pthresh,logthresh):
        self.df_group1 = DEA.selectGroups(self,option1,feature)
        self.df_group2 = DEA.selectGroups(self,option2,feature)
        self.ttest_res = DEA.ttest_results(self,self.df_group1, self.df_group2,pthresh,logthresh)
        self.ttest_res.sort_values(by='Adjusted P-value', ascending=True)
        fig,base = DEA.volcanoPlot(self,pthresh,logthresh)
        savePlot(fig,"DEA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)

    def diffAnalysis_response(self,subgroup,pthresh, logthresh):
        subgroup_df = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        self.df_group1 = subgroup_df[subgroup_df["responsive"]==0]
        self.df_group2 = subgroup_df[subgroup_df["responsive"] == 1]
        self.df_group1 = self.df_group1.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'], axis=1)
        self.df_group2 = self.df_group2.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'],
                                             axis=1)
        self.ttest_res = DEA.ttest_results(self, self.df_group1, self.df_group2, pthresh, logthresh)
        fig,base = DEA.volcanoPlot(self, pthresh, logthresh)
        st.subheader("Volcano plot")
        savePlot(fig, "DEA:resp")
        st.altair_chart(base, theme="streamlit", use_container_width=True)

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
        saveTable(show, feature)
        st.dataframe(show, use_container_width=True)
        st.caption("Ordered by most significantly different (highest adj p-value).")
        return self.ttest_res

    def diffAnalysis_inter(self,subgroup,pthresh,logthresh):
        indexes = subgroup.index
        filtered_df= self.expr_df_selected.drop(indexes)
        self.subgroup = subgroup
        self.ttest_res = DEA.ttest_results(self,self.subgroup,filtered_df,pthresh,logthresh)
        fig,base = DEA.volcanoPlot(self,pthresh,logthresh)
        savePlot(fig, "DEA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)

    def infoTranscript(self, transcript):
        info = self.ttest_res[self.ttest_res['Feature']==transcript].values.flatten().tolist()
        df_info = {'Feature': {'information': info[0]},'T-test result': {'information': round(info[1],3)},
                                'P-value' : {'information': info[2]}, 'LFC': {'information': round(info[3],3)},
                                'Adjusted P-value': {'information': info[4]}, 'Significant': {'information': info[5]}}
        df = pd.DataFrame(data=df_info).T
        df.style.hide(axis='index')
        df.style.hide(axis='columns')
        st.dataframe(df, use_container_width=True)

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
        direction = [((df['LFC'] <= -logthresh) & (df['P-Value'] <= thresh)),((df['LFC'] >= logthresh) & (df['P-Value'] <= thresh)),
                     ((df['LFC'] > -logthresh)&(df['LFC'] < logthresh))]
        values = ['down-regulated', 'up-regulated', 'non-significant']
        df['direction'] = np.select(direction, values)
        down = df[(df['LFC'] <= -logthresh) & (df['P-Value'] <= thresh)]
        up = df[(df['LFC'] >= logthresh) & (df['P-Value'] <= thresh)]
        df['P-Value'] = df['P-Value'].apply(lambda x: -np.log10(x))
        plt.scatter(x=down['LFC'], y=down['P-Value'].apply(lambda x: -np.log10(x)), s=5, label="Down-regulated",
                   color="blue")
        plt.scatter(x=up['LFC'], y=up['P-Value'].apply(lambda x: -np.log10(x)), s=5, label="Up-regulated",
                   color="red")
        plt.xlabel("log2FC")
        plt.ylabel('-log10(p-value)')
        plt.axvline(x=-logthresh,color="grey",linestyle="--")
        plt.axvline(x=logthresh, color="grey", linestyle="--")
        plt.axhline(y=-np.log10(thresh), color="grey", linestyle="--")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        base = alt.Chart(df, title = "Volcano plot").mark_circle(size=100).encode(
            x=alt.Y('LFC', title = "log2FC"),
            y=alt.Y('P-Value', title = '-log10(p-value)'),
            color=alt.Color('direction:O',
                            scale=alt.Scale(domain=values, range=['blue', 'red', 'black']))
        ).interactive().properties(height=700, width=400)

        threshold1 = alt.Chart(pd.DataFrame({'x': [-logthresh]})).mark_rule(strokeDash=[10, 10]).encode(x='x')
        threshold2 = alt.Chart(pd.DataFrame({'x': [logthresh]})).mark_rule(strokeDash=[10, 10]).encode(x='x')
        threshold3 = alt.Chart(pd.DataFrame({'y': [-np.log10(thresh)]})).mark_rule(strokeDash=[10, 10]).encode(y='y')
        base = base +threshold1 + threshold2 + threshold3
        return fig,base

    def infoTest(self,group1,group2,feature,pthresh,logthresh):
        size = len(self.ttest_res[self.ttest_res['Significant']==True])
        info = {'Test: ': {'information': 'T-test'}, 'Multi-sample correction: ': {'information': 'Benjamini/Hochberg'},
                'Groups compared: ': {'information': '{}: {} vs {}'.format(feature,group1,group2)},
                'P-value threshold: ': {'information': pthresh},'log2 FC threshold: ': {'information': logthresh},
                'Num. Significant transcripts: ': {'information': size}}
        df = pd.DataFrame(data=info).T
        df.style.hide(axis='index')
        df.style.hide(axis='columns')
        st.dataframe(df, use_container_width=True)

    def boxplot_inter(self, subgroup, transcript):
        indexes = subgroup.index
        filtered_df = self.expr_df_selected.drop(indexes)
        self.df_group1 = subgroup
        self.df_group2 = filtered_df
        df1 = pd.DataFrame({transcript : self.df_group1[transcript], "class" : 'Samples in interaction'})
        df2 = pd.DataFrame({transcript : self.df_group2[transcript], "class" : 'All samples'})
        full_df = pd.concat([df1, df2])
        fig,axs= plt.subplots(figsize=(8,7))
        sns.boxplot(x="class", y=transcript, data=full_df, hue = "class",palette=["#F72585", "#4CC9F0"], ax = axs)
        sns.stripplot(x="class", y=transcript, data=full_df, hue = "class",palette=["#F72585", "#4CC9F0"],ax = axs)
        plt.legend([],[], frameon=False)
        plt.ylabel("Expression level")
        plt.xlabel("Group")
        base = alt.Chart(full_df, title="Expression level of transcript {}".format(transcript)).mark_boxplot(extent='min-max', ticks=True,size=100).encode(
            x=alt.X("class", title="Group"),
            y=alt.Y(transcript, title="Expression level"),
            color=alt.Color("class", scale=alt.Scale(range=["#F72585", "#4CC9F0"])
                            )).properties(height=650, width = 300)
        return fig,base

    def boxplot_resp(self, subgroup, transcript):
        subgroup_df = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        self.df_group1 = subgroup_df[subgroup_df["responsive"] == 0]
        self.df_group2 = subgroup_df[subgroup_df["responsive"] == 1]
        self.df_group1["Response"] = "Non-responsive"
        self.df_group2["Response"] = "Responsive"
        df1 = pd.DataFrame({transcript : self.df_group1[transcript], "class" : self.df_group1["Response"]})
        df2 = pd.DataFrame({transcript : self.df_group2[transcript], "class" : self.df_group2["Response"]})
        full_df = pd.concat([df1, df2])
        fig, axs = plt.subplots(figsize=(8, 7))
        sns.boxplot(x="class", y=transcript, data=full_df, hue="class", palette=["#F72585", "#4CC9F0"], ax=axs)
        sns.stripplot(x="class", y=transcript, data=full_df, hue="class", palette=["#F72585", "#4CC9F0"], ax=axs)
        plt.legend([], [], frameon=False)
        plt.ylabel("Expression level")
        plt.xlabel("Response to treatment")
        base = alt.Chart(full_df, title="Expression level of transcript {}".format(transcript)).mark_boxplot(extent='min-max', ticks=True, size=100).encode(
            x=alt.X("class:N", title="Group"),
            y=alt.Y(transcript, title="Expression level"),
            color=alt.Color("class:N", scale=alt.Scale(range=["#F72585", "#4CC9F0"])
                            )).properties(height=650, width=300)
        return fig,base

    def boxplot(self, option1, option2,feature,transcript):
        self.df_group1 = DEA.selectGroups(self, option1, feature)
        self.df_group2 = DEA.selectGroups(self, option2, feature)
        df1 = pd.DataFrame({transcript : self.df_group1[transcript], "class" : option1})
        df2 = pd.DataFrame({transcript : self.df_group2[transcript], "class" : option2})
        full_df = pd.concat([df1, df2])
        base = alt.Chart(full_df, title="Expression level of transcript {}".format(transcript)).mark_boxplot(extent='min-max', ticks=True, size=100).encode(
            x=alt.X("class:N", title="Response"),
            y=alt.Y(transcript, title="Expression level"),
            color=alt.Color("class:N", scale=alt.Scale(range=["#F72585", "#4CC9F0"])
                            )).properties(height=650, width=300)
        return base









