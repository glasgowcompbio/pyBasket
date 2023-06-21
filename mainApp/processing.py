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
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

genes_path = '/Users/marinaflores/Desktop/bioinformatics/MBioinfProject/mainApp/pyBasket/pyBasket/Data/Entrez_to_Ensg99.mapping_table.tsv'

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "pyBasket";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hideRows():
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    return hide_table_row_index

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

    def fileInfo(self):
        num_samples = len(self.expr_df_filtered.axes[0])
        num_feat_filt = len(self.expr_df_filtered.axes[1])
        num_feat_select = len(self.expr_df_selected.axes[1])
        clusters = len(np.unique(self.cluster_labels))
        tissues = len(np.unique(self.class_labels))
        name = self.file_name
        info = {'File name: ': {'information': name},'Number of samples: ': {'information': num_samples}, 'Num. transcripts (after filtering): ': {'information': num_feat_filt},
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

    @staticmethod
    def savePlot(fig, feat):
        if st.button('Save Plot', key="plot_"+feat):  # Update the key to a unique value
            fig.savefig('plot_'+feat+'.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")

    @staticmethod
    def saveTable(df, feat):
        if st.button('Save table', key="table_"+feat):  # Update the key to a unique value
            df.to_csv('raw_data_'+feat+'.csv', index=False)
            st.info('Data saved as .csv file in working directory', icon="ℹ️")
        else:
            st.write("")

    def count_plot(self,feature, title, x_lab, response):
        fig = plt.figure(figsize=(12, 8))
        plt.title(title)
        if response == True:
            hue = self.patient_df["responsive"]
            palette = ["#F72585", "#4CC9F0"]
        else:
            hue = None
            palette = sns.color_palette("pastel",25)
        ax = sns.countplot(y=self.patient_df[feature],hue = hue, palette = palette)
        ax.set_xticklabels(ax.get_xticklabels())
        plt.xlabel(x_lab)
        plt.ylabel("Number of samples")
        return fig

    def displayNums(self,feature, feature_title, RD, RawD, title_plot):
        if RawD is True:
            raw_num = Results.raw_data_count(self,feature, feature_title, RD)
            Results.saveTable(raw_num, "NumOfS")
            st.dataframe(raw_num, use_container_width=True)
        else:
            num_plot = Results.count_plot(self,feature, title_plot, feature_title, RD)
            Results.savePlot(num_plot,"NumOfS")
            st.pyplot(num_plot)

    def AAC_plot(self,feature, title, x_lab, response):
        fig = plt.figure(figsize=(12, 6))
        x = feature
        y = self.patient_df["responses"]
        plt.title(title)
        if response == True:
            hue = self.patient_df["responsive"]
            palette = ["#F72585", "#4CC9F0"]
        else:
            hue = None
            palette = sns.color_palette("pastel",25)
        ax = sns.boxplot(data = self.patient_df,x= x, y = y, hue = hue, palette = palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.xlabel(x_lab)
        plt.ylabel("AAC response")
        return fig

    def displayAAC(self,feature, feature_title,RD,RawD, title_plot):
        if RawD is True:
            raw_AAC = Results.raw_data_AAC(self,feature, feature_title)
            Results.saveTable(raw_AAC, "AAC")
            st.dataframe(raw_AAC,use_container_width = True)
        else:
            AAC = Results.AAC_plot(self,feature, title_plot,
                                feature_title, RD)
            Results.savePlot(AAC,"AAC")
            st.pyplot(AAC)

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
        if RawD is True:
            Results.saveTable(self.patient_df, "rawAAC")
            df = self.patient_df
            df["responsive"] = df["responsive"] <1
            st.dataframe(df)
        else:
            if feature == None:
                hue = None
                palette = None
            else:
                hue = feature
                palette = sns.color_palette("pastel", 25)
            fig = plt.figure(figsize=(10, 5))
            x = np.arange(298)
            ax = sns.scatterplot(data=self.patient_df, x=x, hue = hue, y="responses", palette = palette)
            plt.title("AAC response per sample")
            fig.subplots_adjust(right=0.63, top=1)
            defaultPlot_leg(feature, ax)
            #ax.legend(title=feature, title_fontsize=12, fontsize=12, bbox_to_anchor=(1.2, 1), markerscale=0.5)
            plt.xlabel("Sample index")
            fig.subplots_adjust(right=0.63, top=1)
            plt.ylabel("AAC response")
            Results.savePlot(fig, "AAC")
            fig_html = mpld3.fig_to_html(fig)
            components.html(fig_html, height=670, width=1000)


class Analysis():

    def __init__(self, Results):
        self.expr_df_filtered = Results.expr_df_filtered
        self.expr_df_selected = Results.expr_df_selected
        self.drug_response = Results.drug_response
        self.class_labels = Results.class_labels
        self.cluster_labels = Results.cluster_labels
        self.patient_df = Results.patient_df
        self.pca_df = None
        self.pca_variance = None
        self.pca_adv = None
        self.pca_adv_var = None
        self.stacked_posterior = Results.stacked_posterior
        self.num_samples = None

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
            palette = sns.color_palette("pastel",25)
        pc1_values = df['PC1']
        pc2_values = df['PC2']
        fig = plt.figure(figsize=(10, 6))
        fig.subplots_adjust(right=0.63, top=1)
        ax = sns.scatterplot(x=pc1_values, y=pc2_values,hue=df[feature], s=30, palette=palette)
        ax.set(title='PCA decomposition')
        defaultPlot_leg(feature, ax)
        return fig

    @staticmethod
    def showRawData_PCA(df,var):
        var = {'Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 'Var explained': var}
        var_df = pd.DataFrame(var)
        pca_df = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
        return pca_df,var_df

    @staticmethod
    def saveRawD_PCA(feature,name):
        if st.button('Save Data', key="data_PCA"+name):  # Update the key to a unique value
            feature.to_csv('dataPCA.csv')
            st.info('Data saved as .csv in working directory', icon="ℹ️")
        else:
            st.write("")

    @staticmethod
    def savePlot_PCA(fig, feature):
        if st.button('Save Plot', key="plot_PCA"):  # Update the key to a unique value
            fig.savefig('plot_PCA_' + feature + '.png')
            st.info('Plot saved as .png in working directory', icon="ℹ️")
        else:
            st.write("")
    def PCA_analysis(self,feature, RawD):
        Analysis.main_PCA(self,feature)
        if RawD is True:
            pcaDF, var_df = Analysis.showRawData_PCA(self.pca_df, self.pca_variance)
            col11, col12 = st.columns((2, 3))
            with col11:
                st.write('##### Variance explained by component')
                Analysis.saveRawD_PCA(self.pca_variance, "var")
                st.dataframe(var_df, use_container_width=True)
            with col12:
                st.write('##### PCA results')
                Analysis.saveRawD_PCA(self.pca_df, "PCA")
                st.dataframe(pcaDF, use_container_width=True)
        else:
            fig = Analysis.plot_PCA(self,feature,adv=False)
            Analysis.savePlot_PCA(fig, feature)
            fig_html = mpld3.fig_to_html(fig)
            components.html(fig_html, height=650, width=1000)

    def adv_PCA(self,sub_df, RawD):
        try:
            Analysis.advanced_PCA(self, sub_df)
            if RawD is True:
                pcaDF, var_df = Analysis.showRawData_PCA(self.pca_adv, self.pca_adv_var)
                col11, col12 = st.columns((2, 3))
                with col11:
                    st.write('##### Variance explained by component')
                    Analysis.saveRawD_PCA(self.pca_adv_var, "var")
                    st.dataframe(var_df, use_container_width=True)
                with col12:
                    st.write('##### PCA results')
                    Analysis.saveRawD_PCA(self.pca_adv, "PCA")
                    st.dataframe(pcaDF, use_container_width=True)
            else:
                fig = Analysis.plot_PCA(self, "responsive",adv= True)
                Analysis.savePlot_PCA(fig, "subgroup")
                fig_html = mpld3.fig_to_html(fig)
                components.html(fig_html, height=600, width=3000)
        except:
            st.warning("Not enough samples. Please try a different combination.")

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
        #sns.clustermap(data=df, cmap = "RdBu_r", yticklabels='auto')
        plt.title('Transcriptional expression per sample')
        plt.xlabel('Transcripts')
        plt.ylabel('Samples')
        plt.yticks(fontsize=9)
        plt.xticks(fontsize=9)
        return fig

    def heatmapResponse(self,results):
        clusters = results.setClusters()
        baskets = results.setBaskets()
        data = []
        for basket in baskets:
            response = []
            for cluster in clusters:
                sub = len(self.patient_df[(self.patient_df['cluster_number'] == cluster) & (self.patient_df['tissues'] == basket)& (self.patient_df['responsive'] == 1)])
                response.append(sub)
            data.append(response)
        df = pd.DataFrame(data, baskets,clusters)
        return df

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

    def HM_inferredProb(self,results):
        basket_coords, cluster_coords = results.setBaskets(),results.setClusters()
        stacked = self.stacked_posterior
        predt_basket = np.mean(stacked.basket_p.values, axis=1)
        inferred_basket = pd.DataFrame({'prob': predt_basket}, index = basket_coords)
        inferred_basket =  inferred_basket.values
        inferred_cluster = np.mean(stacked.cluster_p.values, axis=2)
        inferred_mat = inferred_basket * inferred_cluster
        inferred_df = pd.DataFrame(inferred_mat, index=basket_coords, columns=cluster_coords)
        return inferred_df

    def heatmap_interaction(self,results,df, title,num_Sum,x_highlight=None, y_highlight=None):
        Analysis.heatmapNum(self,results)
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


