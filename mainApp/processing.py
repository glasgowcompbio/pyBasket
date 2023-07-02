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
from common import savePlot,saveTable
import altair as alt
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

genes_path = os.path.join('..', 'pyBasket/Data', 'Entrez_to_Ensg99.mapping_table.tsv')

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
        self.importance_df = pickle_data[list(pickle_data)[8]]
        self.stacked_posterior = pickle_data[list(pickle_data)[6]]
        self.clusters_names = sorted(list(self.patient_df['cluster_number'].unique()))
        self.baskets_names = sorted(list(self.patient_df['tissues'].unique()))

    def setClusters(self):
        clusters = list(self.patient_df['cluster_number'].unique())
        clusters = sorted(clusters)
        return clusters

    def setBaskets(self):
        tissues = list(self.patient_df['tissues'].unique())
        return tissues

    def setFeatures(self):
        selected_genes = Results.IdGenes(self.expr_df_selected)
        filtered_genes = Results.IdGenes(self.expr_df_filtered)
        self.expr_df_selected.columns = selected_genes
        self.importance_df.index = selected_genes
        self.expr_df_filtered.columns = filtered_genes
    def setPatients(self):
        reseted_df = self.patient_df.reset_index()
        return reseted_df

    def fileInfo(self):
        num_samples = len(self.expr_df_filtered.axes[0])
        num_feat_filt = len(self.expr_df_filtered.axes[1])
        num_feat_select = len(self.expr_df_selected.axes[1])
        clusters = len(np.unique(self.cluster_labels))
        tissues = len(np.unique(self.class_labels))
        drug = self.file_name.split('_')[2]
        info = {'File name: ': {'information': self.file_name},'Drug: ': {'information': drug}, 'Num. transcripts (after filtering): ': {'information': num_feat_filt},
                'Num. transcripts (after feature selection): ': {'information': num_feat_select}, 'Num.clusters: ': {'information': clusters}, 'Num. tissues/baskets: ' : {'information': tissues} }
        df = pd.DataFrame(data=info).T
        style = df.style.hide_index()
        style.hide_columns()
        return st.dataframe(df,use_container_width = True)

    @staticmethod
    def IdGenes(df):
        features_EN = df.columns.tolist()
        genes_df = pd.read_csv(genes_path, sep='\t')
        matching_genes = genes_df[genes_df['ensg_v99'].isin(features_EN)].reset_index(drop=True)
        matching_genesEN = matching_genes['ensg_v99'].tolist()
        genes = []
        for col in features_EN:
            found_match = False
            for i in range(len(matching_genesEN)):
                if col == matching_genesEN[i]:
                    genes.append(matching_genes['gene_name_v99'][i])
                    found_match = True
                    break
            if not found_match:
                genes.append(col)
        return genes

    def count_plot(self,feature, title, x_lab, response):
        if response == True:
            hue = self.patient_df["responsive"]
            self.patient_df["responsive"] = self.patient_df["responsive"].replace([0, 1], ['Non-responsive', 'Responsive'])
            self.patient_df[feature] = pd.Categorical(self.patient_df[feature])
            df_grouped = self.patient_df.groupby(["responsive", feature]).size().reset_index(name='Count')
            palette = ["#F72585", "#4CC9F0"]
            base = alt.Chart(df_grouped).mark_bar().encode(
                alt.X(feature, type = "nominal"),
                alt.Y('Count', axis=alt.Axis(grid=False)),
                alt.Color("responsive"),tooltip=["responsive", feature]
            ).properties(height=650).configure_range(category=alt.RangeScheme(palette))
        else:
            hue = None
            palette = sns.color_palette("Paired", 25).as_hex()
            base = alt.Chart(self.patient_df).transform_aggregate(
                count='count()',
                groupby=[feature]
            ).mark_bar().encode(
                alt.X(feature + ':N', title=x_lab),
                alt.Y('count:Q', title="Number of samples"), alt.Color(feature + ':N'),
                tooltip=['count:Q', feature]
            ).properties(
                height=650,
                title="Number of samples"
            ).configure_range(
                category=alt.RangeScheme(palette))
        return base

    def displayNums(self,feature, feature_title, RD, RawD, title_plot):
        if RawD is True:
            raw_num = Results.raw_data_count(self,feature, feature_title, RD)
            saveTable(raw_num, "NumOfS")
            st.dataframe(raw_num, use_container_width=True)
        else:
            num_plot = Results.count_plot(self,feature, title_plot, feature_title, RD)
            savePlot(num_plot,"NumOfS")
            st.altair_chart(num_plot,theme = "streamlit",use_container_width=True)

    def AAC_plot(self,feature, title, x_lab, response):
        if response == True:
            palette = ["#F72585", "#4CC9F0"]
            df = Results.setPatients(self)
            df[feature] = pd.Categorical(df[feature])
            base = alt.Chart(df, title= "AAC response").mark_boxplot(extent='min-max', ticks=True, size = 60).encode(
                x=alt.X(feature, title=x_lab, scale=alt.Scale(padding=1)),
                y=alt.Y("responses", title="AAC response"),
                color = alt.Color("responsive:N", scale=alt.Scale(range=palette)
            )).properties(height=650).configure_facet(spacing=0).configure_view(stroke=None)
        else:
            palette = sns.color_palette("Paired",25).as_hex()
            df = Results.setPatients(self)
            df[feature] = pd.Categorical(df[feature])
            base = alt.Chart(df,title= "AAC response").mark_boxplot(extent='min-max',ticks=True,size=60).encode(
                x=alt.X(feature,title = x_lab),
                y=alt.Y("responses",title = x_lab),color = alt.Color(feature + ':N')
            ).properties(height=650).configure_range(category=alt.RangeScheme(palette))
        return base

    def displayAAC(self,feature, feature_title,RD,RawD, title_plot):
        if RawD is True:
            raw_AAC = Results.raw_data_AAC(self,feature, feature_title)
            saveTable(raw_AAC, "AAC")
            st.dataframe(raw_AAC,use_container_width = True)
        else:
            fig= Results.AAC_plot(self,feature, title_plot,
                                feature_title, RD)
            savePlot(fig,"AAC")
            st.altair_chart(fig, theme="streamlit", use_container_width=True)

    def displayAAC_none(feature):
        fig = Results.non_group_plot(feature)
        savePlot(fig, "AAC")
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600, width=1000)

    def raw_data_count(self, feature, x_variable, response):
        self.patient_df['Number of samples'] = 1
        if response == True:
            features = ["responsive", feature]
            columns = ["responsive",x_variable, "Number of samples"]
        else:
            features = feature
            columns = [x_variable, "Number of samples"]
        raw_count = self.patient_df.groupby(features).size().to_frame(name = 'count').reset_index()
        raw_count.columns = columns
        if response == True:
            raw_count['responsive'] = raw_count['responsive']== 1
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

    def non_group_plot(self, feature,RawD):
        self.patient_df["index"] = range(self.patient_df.shape[0])
        if RawD is True:
            saveTable(self.patient_df, "rawAAC")
            df = self.patient_df
            df["responsive"] = df["responsive"] <1
            st.dataframe(df)
        else:
            if feature == None:
                df = Results.setPatients(self)
                base = alt.Chart(df,title= "AAC response per sample").mark_circle(size=100).encode(
                    x=alt.Y("index", title="Sample index"),
                    y=alt.Y("responses", title="AAC response"),
                    tooltip=["samples", "responses"]
                ).interactive().properties(height=650)
            else:
                palette = sns.color_palette("Paired", 25).as_hex()
                df = Results.setPatients(self)
                df[feature] = pd.Categorical(df[feature])
                base = alt.Chart(df, title= "AAC response per sample").mark_circle(size=100).encode(
                    x=alt.Y("index", title="Sample index"),
                    y=alt.Y("responses", title="AAC response"),color = feature,
                    tooltip=["samples", "responses", feature]
                ).interactive().properties(height=650).configure_range(category=alt.RangeScheme(palette))
            savePlot(base, "AAC")
            st.altair_chart(base,theme = "streamlit",use_container_width=True)


class Analysis():

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
        feature = fulldf['responsive'].values
        fulldf = fulldf[['tissues', 'responses', 'cluster_number', 'responsive']]
        fulldf = fulldf.sort_values(by='responses')
        fulldf.index.name = 'Sample'
        fulldf['responsive'] = fulldf['responsive'] == 1
        return fulldf


class heatMap(Analysis):
    def __init__(self, Results):
        super().__init__(Results)
        self.num_samples = None

    def heatmapNum(self,results):
        clusters = results.setClusters()
        baskets = results.setBaskets()
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
        """
        base = alt.Chart(df).mark_rect().encode(
            x='x:O',
            y='y:O',
            color='z:Q'
        )
        """
        return fig

    def heatmapResponse(self, results):
        clusters = results.setClusters()
        baskets = results.setBaskets()
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
        basket_coords, cluster_coords = results.setBaskets(),results.setClusters()
        stacked = self.stacked_posterior
        inferred_mat = np.mean(stacked.joint_p.values, axis=2)
        inferred_df = pd.DataFrame(inferred_mat, index=basket_coords, columns=cluster_coords)
        return inferred_df

    def heatmap_interaction(self, results, df, title, num_Sum, x_highlight=None, y_highlight=None):
        heatMap.heatmapNum(self, results)
        x_highlight = results.setClusters().index(x_highlight)
        y_highlight = results.setBaskets().index(y_highlight)
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

class dim_PCA(Analysis):
    def __init__(self, Results):
        super().__init__(Results)
        self.pca_df = None
        self.pca_variance = None
        self.pca_adv = None
        self.pca_adv_var = None
        self.num_samples = None

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

    def plot_PCA(self,feature,adv):
        df = self.pca_adv if adv == True else self.pca_df
        if feature == "responsive":
            palette = ["#F72585", "#4CC9F0"]
        else:
            palette = sns.color_palette("Paired",25).as_hex()
        pc1_values = df['PC1']
        pc2_values = df['PC2']
        base = alt.Chart(df, title = "Principal Component Analysis").mark_circle(size=60).encode(
            x='PC1',
            y='PC2',
            color=feature+':N'
        ).interactive().properties(height=650).configure_range(
        category=alt.RangeScheme(palette))

        fig = plt.figure(figsize=(11, 6))
        fig.subplots_adjust(right=0.63, top=1)
        ax = sns.scatterplot(x=pc1_values, y=pc2_values,hue=df[feature], s=30, palette=palette)
        ax.set(title='PCA decomposition')
        defaultPlot_leg(feature, ax)
        return fig, base

    @staticmethod
    def showRawData_PCA(df,var):
        var = {'Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 'Var explained': var}
        var_df = pd.DataFrame(var)
        pca_df = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
        return pca_df,var_df

    def infoPCA(self, feature):
        info = {'Technique: ': {'information': 'Principal Component Analysis'}, 'Feature: ': {'information': feature},
                'Number of components: ': {'information': 5}}
        df = pd.DataFrame(data=info).T
        style = df.style.hide_index()
        style.hide_columns()
        return st.dataframe(df, use_container_width=True)

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
        self.pca_df = pca_df
        self.pca_variance = variance

    def PCA_analysis(self,feature, RawD):
        dim_PCA.main_PCA(self,feature)
        if RawD is True:
            pcaDF, var_df = dim_PCA.showRawData_PCA(self.pca_df, self.pca_variance)
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
            fig,base = dim_PCA.plot_PCA(self,feature,adv=False)
            savePlot(fig, "PCA_"+feature)
            st.altair_chart(base, theme="streamlit", use_container_width=True)


    def adv_PCA(self,sub_df, RawD):
        try:
            dim_PCA.advanced_PCA(self, sub_df)
            if RawD is True:
                pcaDF, var_df = dim_PCA.showRawData_PCA(self.pca_adv, self.pca_adv_var)
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
                fig,base = dim_PCA.plot_PCA(self, "responsive",adv= True)
                savePlot(fig, "PCA_subgroup")
                st.altair_chart(base, theme="streamlit", use_container_width=True)
        except:
            st.warning("Not enough samples. Please try a different combination.")
