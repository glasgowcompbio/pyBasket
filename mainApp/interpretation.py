from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import fdrcorrection
from common import savePlot, saveTable,alt_boxplot, colours, findSubgroup
import altair as alt

np.set_printoptions(suppress=True, precision=3)

if "data" in st.session_state:
    data = st.session_state["data"]

"""
Class: Prototypes: creates a Prototype object 
Input: needs an initialised Data object
"""
class Prototypes():
    def __init__(self, Data):
        self.expr_df_selected = Data.expr_df_selected
        self.class_labels = Data.class_labels
        self.cluster_labels = Data.cluster_labels
        self.subgroup = None
        self.patient_df = Data.patient_df
        self.pseudo_medoids = []

    #Method to compute pseudo-medoids based on clustering results
    def pseudoMedoids(self,df,feature):
        rawX = df.reset_index()
        rawX = rawX.drop(['index'], axis=1)
        rawX["labels"] = feature
        rawX["labels"] = rawX["labels"].astype('category').cat.codes
        unique_labels = np.unique(rawX["labels"])
        labels = rawX["labels"]
        pseudo_medoids = []
        for label in unique_labels:
            label_points = rawX[labels == label]
            distances = cdist(label_points, label_points, metric='euclidean')
            total_distances = np.sum(distances, axis=1)
            medoid_index = np.argmin(total_distances)
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

    #Function to plot pseudo-medoids found
    def plotMedoids(self,X,sample_medoids, feature):
        labels = np.unique(feature)
        palette = colours(len(np.unique(feature)))
        df = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'class':feature})
        df2 = self.patient_df.reset_index()
        df = pd.merge(df, df2, left_index=True, right_index=True)
        df3 = pd.merge(df, sample_medoids, left_index=True, right_index=True)
        scale = alt.Scale(domain=labels, range = palette)
        base = alt.Chart(df,title="Prototypes samples").mark_circle(size=100).encode(
            x=alt.X('X', title='PC1'),
            y=alt.X('Y', title = 'PC2'),color = alt.Color('class:N', scale = scale), tooltip = ['samples', 'class']
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

    #Function to select samples that have been chosen as pseudo-medoids to show in a table
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

    #Function to find prototypes, plot them with all samples and show them in a table
    def findPrototypes(self, option):
        feature = self.cluster_labels if option == 'Clusters' else self.class_labels
        data,sampleMedoids = Prototypes.pseudoMedoids(self,self.expr_df_selected,feature)
        base = Prototypes.plotMedoids(self,data,sampleMedoids,feature)
        savePlot(base,option)
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.caption("The prototypical sample of each level is marked in black.")
        table = Prototypes.showMedoids(self,feature)
        st.subheader("Prototype samples")
        st.write("The full transcriptional expression profile of the prototypical samples is shown below. ")
        saveTable(table, option)
        st.dataframe(table)

    # Function to find prototypes in the basket-cluster interaction subgroup of samples , plot them and show them in a table
    def findPrototypes_sub(self,subgroup):
        self.subgroup = subgroup
        fulldf = pd.merge(self.patient_df, self.subgroup, left_index=True, right_index=True)
        feature = fulldf['responsive'].values
        fulldf = fulldf.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'], axis=1)
        data, sampleMedoids = Prototypes.pseudoMedoids(self,fulldf, feature)
        base = Prototypes.plotMedoids(self,data, sampleMedoids, feature)
        savePlot(base, "subgroup")
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.caption("The prototypical sample of each level is marked in black.")
        st.subheader("Prototype samples")
        table = Prototypes.showMedoids(self, feature)
        st.write("The full transcriptional expression profile of the prototypical samples is shown below. ")
        saveTable(table, "subgroup")
        st.dataframe(table, use_container_width=True)

"""
Class: DEA (Differential Expression Analysis): creates an object to perform Differential Expression Analysis of samples 
Input: needs an initialised Data object
"""
class DEA():
    def __init__(self, Data):
        self.expr_df_selected = Data.expr_df_selected
        self.class_labels = Data.class_labels
        self.cluster_labels = Data.cluster_labels
        self.patient_df = Data.patient_df
        self.df_group1 = None
        self.df_group2 = None
        self.subgroup = None
        self.ttest_res = None
        self.transcripts = None

    #Function to split a subgroup of samples by response
    def splitResponses(self,subgroup):
        subgroup_df = pd.merge(self.patient_df, subgroup, left_index=True, right_index=True)
        self.df_group1 = subgroup_df[subgroup_df["responsive"] == "Non-responsive"]
        self.df_group2 = subgroup_df[subgroup_df["responsive"] == "Responsive"]

    #Function to perform t-test between two specified groups and applying p-value and LFC thresholds
    def ttest_results(self,df1,df2,pthresh,logthresh):
        ttest_results = []
        for column in df1.columns:
            t, p = ttest_ind(df1[column], df2[column], nan_policy='omit')
            l2fc = np.mean(df1[column].values) - np.mean(df2[column].values)
            ttest_results.append((column, t, p, l2fc))
        dea_results = pd.DataFrame(ttest_results, columns=['Feature', 'T-Statistic', 'P-Value', 'LFC'])
        dea_results = dea_results.dropna()
        dea_results['Corrected P-value'] = fdrcorrection(dea_results['P-Value'].values)[1]
        #_, dea_results['Corrected P-value'],_, _ = fdrcorrection(dea_results['P-Value'].values)
                                                                     #method='fdr_bh')
        dea_results = dea_results.sort_values(by=['Corrected P-value'])
        dea_results['Significant'] = (dea_results['Corrected P-value'] < pthresh) & (abs(dea_results['LFC']) > logthresh)
        return dea_results

    #Function to perform DEA between two groups
    def diffAnalysis_simple(self,option1, option2, feature,pthresh,logthresh):
        self.df_group1 = findSubgroup(option1,feature)
        self.df_group2 = findSubgroup(option2,feature)
        self.ttest_res = DEA.ttest_results(self,self.df_group1, self.df_group2,pthresh,logthresh)
        self.ttest_res.sort_values(by='Corrected P-value', ascending=True)
        base = DEA.volcanoPlot(self,pthresh,logthresh)
        st.subheader("Volcano plot")
        st.write(
            "The volcano plot combines results from Fold Change (FC) Analysis and T-tests to select significant features based on the selected "
            " statistical significance thresholds (adjusted p-value and LFC threshold). It shows statistical significance (corrected P value) vs the magnitude"
            " of change (LFC) between the two conditions. Below, results are shown for {} vs {}.".format(option1, option2))
        savePlot(base,"DEA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.caption("The x-axis represents the magnitude of change by the log of the FC. The y-axis represents the "
                   "statistical significance by corrected p-value. Red represents up-regulated transcripts in the second condition compared to the first condition. "
                   "Blue values represent transcripts down-regulated in the second condition compared to the first condition.")

    #Function to perform DEA within a group between responsive vs non-responsive samples
    def diffAnalysis_response(self,subgroup,pthresh, logthresh):
        DEA.splitResponses(self, subgroup)
        if len(self.df_group1) >1 and len(self.df_group2) >1:
            self.df_group1 = self.df_group1.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'], axis=1)
            self.df_group2 = self.df_group2.drop(['tissues', 'responses', 'basket_number', 'cluster_number', 'responsive'],
                                                 axis=1)
            self.ttest_res = DEA.ttest_results(self, self.df_group1, self.df_group2, pthresh, logthresh)
            base = DEA.volcanoPlot(self, pthresh, logthresh)
            st.subheader("Volcano plot")
            st.write(
                "The volcano plot combines results from Fold Change (FC) Analysis and T-tests to select significant features based on the selected "
                " statistical significance thresholds (adjusted p-value and LFC threshold). It shows statistical significance (corrected P value) vs the magnitude"
                " of change (LFC) between the two conditions. Below, results are shown for responsive vs non-responsive samples within the selected basket*cluster"
                " interaction.")
            savePlot(base, "DEA:resp")
            st.altair_chart(base, theme="streamlit", use_container_width=True)
            st.caption("The x-axis represents the magnitude of change by the log of the FC. The y-axis represents the "
                       "statistical significance by corrected p-value. Red represents up-regulated transcripts in the second condition compared to the first condition. "
                       "Blue values represent transcripts down-regulated in the second condition compared to the first condition.")
        else:
            st.warning("There are not enough samples to do DEA. Please choose another combination")

    #Function to show DEA results in a table
    def showResults(self,feature):
        st.subheader("Results")
        st.write("Results from Fold Change (FC) and T-test analyses for each transcript/feature are shown below. Significant features are those features whose adjusted "
                 "p-value is beyond the selected adjusted p-value threshold, either up or down regulated.")
        self.ttest_res['Corrected P-value'] = self.ttest_res['Corrected P-value'].apply('{:.6e}'.format)
        self.ttest_res['P-Value'] = self.ttest_res['P-Value'].apply('{:.6e}'.format)
        only_sig = st.checkbox('Show only significant transcripts.')
        num_sig = len(self.ttest_res[self.ttest_res["Significant"] == True])
        if only_sig and num_sig >0:
            numShow = st.slider('Select number of transcripts to show', 0,)
            df_show = self.ttest_res[self.ttest_res["Significant"] == True][:numShow]
        elif only_sig:
            st.warning("No significant transcripts found.")
            df_show = None
        else:
            numShow = st.slider('Select number of transcripts to show', 0,len(self.ttest_res))
            df_show = self.ttest_res[:numShow]
        df_show = df_show.drop('direction', axis = 1)
        saveTable(df_show, feature)
        st.dataframe(df_show, use_container_width=True)
        st.caption("Ordered by most significantly different (highest adj p-value).")
        return self.ttest_res

    #Function to perform DEA between samples in an interaction and rest of samples
    def diffAnalysis_inter(self,subgroup,pthresh,logthresh):
        indexes = subgroup.index
        filtered_df= self.expr_df_selected.drop(indexes)
        self.subgroup = subgroup
        self.ttest_res = DEA.ttest_results(self,self.subgroup,filtered_df,pthresh,logthresh)
        st.write("#### Volcano plot")
        st.write(
            "The volcano plot combines results from Fold Change (FC) Analysis and T-tests to select significant features based on the selected "
            " statistical significance thresholds (corrected p-value and LFC threshold). It shows statistical significance (Corrected P value) vs the magnitude"
            " of change (LFC) between the two conditions. Below, results are shown for samples in the selected basket*cluster interaction"
            " vs any other sample.")
        base = DEA.volcanoPlot(self,pthresh,logthresh)
        savePlot(base, "DEA")
        st.altair_chart(base, theme="streamlit", use_container_width=True)
        st.caption("The x-axis represents the magnitude of change by the log of the FC. The y-axis represents the "
                   "statistical significance by corrected p-value. Red represents up-regulated transcripts in the second condition compared to the first condition. "
                   "Blue values represent transcripts down-regulated in the second condition compared to the first condition.")

    #Function to show information about a transcript chosen in a table
    def infoTranscript(self, transcript):
        info = self.ttest_res[self.ttest_res['Feature']==transcript].values.flatten().tolist()
        df_info = {'Feature': {'information': info[0]},'T-test result': {'information': round(info[1],3)},
                                'P-value' : {'information': info[2]}, 'LFC': {'information': round(info[3],3)},
                                'Corrected P-value': {'information': info[4]}, 'Significant': {'information': info[5]}}
        df = pd.DataFrame(data=df_info).T
        df.style.hide(axis='index')
        df.style.hide(axis='columns')
        st.write(" ")
        st.write("##### DEA results for {}".format(transcript))
        st.dataframe(df, use_container_width=True)

    #Function to plot DEA results in a volcano plot
    def volcanoPlot(self, thresh, logthresh):
        df = self.ttest_res
        direction = [(df['LFC'] <= -logthresh),(df['LFC'] >= logthresh),
                     ((df['LFC'] > -logthresh)&(df['LFC'] < logthresh))]
        values = ['down-regulated', 'up-regulated', 'non-significant']
        df['direction'] = np.select(direction, values)
        df['Corrected P-value'] = df['Corrected P-value'].apply(lambda x: -np.log10(x))
        base = alt.Chart(df, title = "Volcano plot").mark_circle(size=100).encode(
            x=alt.Y('LFC', title = "log2FC"),
            y=alt.Y('Corrected P-value', title = '-log10(corrected p-value)'),
            color=alt.Color('direction:O',
                            scale=alt.Scale(domain=values, range=['blue', 'red', 'black'])), tooltip = ['LFC','Corrected P-value','Feature']
        ).interactive().properties(height=700, width=400)
        threshold1 = alt.Chart(pd.DataFrame({'x': [-logthresh]})).mark_rule(strokeDash=[10, 10]).encode(x='x')
        threshold2 = alt.Chart(pd.DataFrame({'x': [logthresh]})).mark_rule(strokeDash=[10, 10]).encode(x='x')
        threshold3 = alt.Chart(pd.DataFrame({'y': [-np.log10(thresh)]})).mark_rule(strokeDash=[10, 10]).encode(y='y')
        base = base +threshold1 + threshold2 + threshold3
        return base

    #Function to show information about the type of DEA being performed
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

    #Function to show in a boxplot differences in expression between two groups being compared
    def boxplot_inter(self, subgroup, transcript):
        indexes = subgroup.index
        filtered_df = self.expr_df_selected.drop(indexes)
        self.df_group1 = subgroup
        self.df_group2 = filtered_df
        df1 = pd.DataFrame({transcript : self.df_group1[transcript], "Group" : 'Samples in interaction'})
        df2 = pd.DataFrame({transcript : self.df_group2[transcript], "Group" : 'All samples'})
        full_df = pd.concat([df1, df2])
        alt_boxplot(full_df, "Group", transcript, 2, "Group","Expression level", "Group", "Expression level of transcript {}".format(transcript), "DEA_"+transcript)
        st.caption(
            "The x-axis represents the two groups being compared. The y-axis is the expression level of the chosen transcript.")

    # Function to show in a boxplot differences in expression between two groups being compared
    def boxplot_resp(self, subgroup, transcript):
        DEA.splitResponses(self, subgroup)
        df1 = pd.DataFrame({transcript : self.df_group1[transcript], "class" : self.df_group1["responsive"]})
        df2 = pd.DataFrame({transcript : self.df_group2[transcript], "class" : self.df_group2["responsive"]})
        full_df = pd.concat([df1, df2])
        alt_boxplot(full_df, "class", transcript, 2, "Group", "Expression level", "class", "Expression level of transcript {}".format(transcript), "DEA"+transcript)
        st.caption(
        "The x-axis represents the two groups being compared. The y-axis is the expression level of the chosen transcript.")

    # Function to show in a boxplot differences in expression between two groups being compared
    def boxplot(self, option1, option2, feature, transcript):
        self.df_group1 = findSubgroup(option1, feature)
        self.df_group2 = findSubgroup(option2, feature)
        df1 = pd.DataFrame({transcript: self.df_group1[transcript], "Group": option1})
        df2 = pd.DataFrame({transcript: self.df_group2[transcript], "Group": option2})
        full_df = pd.concat([df1, df2])
        alt_boxplot(full_df, "Group:N", transcript+':Q', 2, "Group", "Expression level", "Group",
                    "Expression level of transcript {}".format(transcript), "DEA_" + transcript)
        st.caption("The x-axis represents the two groups being compared. The y-axis is the expression level of the chosen transcript.")




