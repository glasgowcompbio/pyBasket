import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.inspection import permutation_importance
import streamlit as st
import sklearn
import shap
import numpy as np

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

    def limeInterpretation(self,sample,n_features):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index
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

    def SHAP_forces(self, sample):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"]==sample].index
        explainer,values = FI.SHAP(self)
        fig = shap.force_plot(explainer.expected_value, values[index, :], self.expr_df_selected.iloc[index, :], matplotlib=True, show=False)
        return fig

    def SHAP_summary(self):
        explainer, values = FI.SHAP(self)
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
        elif response == "Only non-responsive samples":
            resp = 0

        df = df[df["responsive"]==resp]
        return df
