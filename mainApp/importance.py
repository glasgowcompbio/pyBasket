import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.inspection import permutation_importance
import streamlit as st
import sklearn
import shap
import numpy as np
import pandas as pd
from common import savePlot, saveTable

if "data" in st.session_state:
    data = st.session_state["data"]

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
        importance_vals= importance_sort['importance_score'].values[:25]
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

    def limeInterpretation(self,sample,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index
        df = df.drop("index",axis = 1)
        FI.prepareData(self,self.drug_response)
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100,random_state=42)
        rf.fit(self.X_train, self.y_train)

        explainer = lime_tabular.LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, class_names=["drug response"],
                                                      verbose=True, mode='regression')

        exp = explainer.explain_instance(df.iloc[index[0]], rf.predict, num_features=n_features)
        fig = exp.as_pyplot_figure()
        raw_data = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
        if RawD:
            saveTable(raw_data, sample + "LIME")
            st.dataframe(raw_data)
        else:
            savePlot(fig,sample+"LIME")
            st.pyplot(fig)
        st.caption(
            "Green values: positive impact, increase model score. Red values: negative impact, decreases model score. ")

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

    def SHAP_forces(self, sample, explainer,values,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"]==sample].index
        top_feature_indices = np.argsort(np.abs(values[index]))[::-1]
        top_feature_indices = top_feature_indices[0][:n_features+1]

        selected_shap_values = values[index, top_feature_indices]
        selected_features = round(self.expr_df_selected.iloc[index, top_feature_indices],3)
        raw_data = pd.DataFrame({'Feature': selected_features.columns,'SHAP value': selected_features.iloc[0].values})
        fig = shap.force_plot(explainer.expected_value, selected_shap_values, selected_features, matplotlib=True, show=False)
        st.write("  ")
        if RawD:
            saveTable(raw_data, sample + "SHAP")
            st.write("  ")
            st.dataframe(raw_data)
        else:
            savePlot(fig, sample + "SHAP")
            st.write("  ")
            st.pyplot(fig)

    def SHAP_summary(self,explainer,values):
        fig, ax = plt.subplots()
        shap.summary_plot(values, self.expr_df_selected, show=False, plot_size=(8, 6), color='b')
        return fig

    def displaySamples(self,cluster,basket):
        transcripts = self.expr_df_selected
        self.patient_df = self.patient_df.reset_index()
        if cluster == "None":
            sub_patients = self.patient_df[(self.patient_df['tissues'] == basket)]
        elif basket == "None":
            sub_patients = self.patient_df[(self.patient_df['cluster_number'] == cluster)]
        else:
            sub_patients = self.patient_df[
                (self.patient_df['cluster_number'] == cluster) & (self.patient_df['tissues'] == basket)]
        selection = sub_patients
        samples = selection["samples"]
        transcript_df = transcripts.loc[samples]
        num = len(transcript_df)
        return samples,num

    def filterSamples(self,samples,response):
        df = self.patient_df.loc[self.patient_df["samples"].isin(samples.tolist())]
        if response == 'Only responsive samples':
            resp = 1
            df = df[df["responsive"] == resp]
        elif response == "Only non-responsive samples":
            resp = 0
            df = df[df["responsive"]==resp]
        else:
            df = df
        return df
