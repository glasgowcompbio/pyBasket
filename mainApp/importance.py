import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.inspection import permutation_importance
#from rfpimp import permutation_importances
import streamlit as st
import sklearn
import shap
import streamlit.components
import numpy as np
import pandas as pd
from common import savePlot, saveTable, openGeneCard,alt_hor_barplot
from alibi.explainers import ALE, plot_ale
import altair as alt


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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def plotImportance(self, RawD, num_feats):
        importance_sort = self.importance.sort_values('importance_score', ascending=False)
        importance_vals= importance_sort['importance_score'].values[:num_feats]
        importance_vals = importance_vals[::-1]
        importance_feats = importance_sort.index[:num_feats]
        importance_feats = importance_feats[::-1]
        raw_D = pd.DataFrame({'Features':importance_feats,'Importance score':importance_vals})
        raw_D = raw_D.sort_values(by='Importance score', ascending=False)
        if RawD:
            saveTable(raw_D,"RF-FI")
            st.dataframe(raw_D)
        else:
            alt_hor_barplot(raw_D, 'Importance score', 'Features', num_feats, 'Importance score', "Features",
                            'Features', "Feature Importance by Random Forest", "RF-FI")

    def prepareData(self, y):
        train_size = int(len(self.expr_df_selected) * .8)
        self.X_train, self.X_test = self.expr_df_selected.iloc[:train_size], self.expr_df_selected.iloc[train_size:]
        y_train, self.y_test = y.iloc[:train_size], y.iloc[train_size:]
        self.y_train = y_train.values.flatten()
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train.values, self.y_train)
        return rf

    def limeInterpretation(self,sample,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index
        df = df.drop("index",axis = 1)
        rf = FI.prepareData(self,self.drug_response)
        explainer = lime_tabular.LimeTabularExplainer(self.expr_df_selected.values, feature_names=self.expr_df_selected.columns, class_names=["drug response"],
                                                      verbose=True, mode='regression')
        exp = explainer.explain_instance(df.iloc[index[0]], rf.predict, num_features=n_features,)
        fig = exp.as_pyplot_figure()
        raw_data = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
        st.write("##### The predicted value for sample {} is {}".format(sample, round(exp.local_pred[0],3)))
        st.write(" ")
        if RawD:
            saveTable(raw_data, sample + "LIME")
            st.dataframe(raw_data)
        else:
            savePlot(fig,sample+"LIME")
            st.pyplot(fig)
        return raw_data

    def permutationImportance(self, num_feats, RawD):
        rf = FI.prepareData(self, self.drug_response)
        perm_importance = permutation_importance(rf, self.X_test.values, self.y_test.values,n_repeats=10, random_state=42)
        perm_sorted_idx = perm_importance.importances_mean.argsort()
        perm_sorted_idx = perm_sorted_idx[::-1]
        perm_sorted_idx = perm_sorted_idx[:num_feats]
        raw_D = pd.DataFrame({'Features': self.X_test.columns[perm_sorted_idx], 'Importance': perm_importance.importances_mean[perm_sorted_idx]})
        if RawD:
            saveTable(raw_D, "PBI")
            st.dataframe(raw_D)
        else:
            alt_hor_barplot(raw_D, 'Importance', 'Features', num_feats, 'Mean Importance score', "Features", 'Features', "Feature Importance by Permutation based", "PIB")

    def SHAP(self):
        rf = FI.prepareData(self, self.drug_response)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(self.expr_df_selected)
        self.model = rf
        return explainer, shap_values

    def SHAP_pred(self, sample):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index[0]
        X_array = np.array([self.expr_df_selected.iloc[index, :]])
        pred = self.model.predict(X_array)
        pred = round(pred[0], 4)
        true_label = self.drug_response.iloc[index]
        true_label = round(true_label[0],4)
        return pred, true_label

    def SHAP_bar_indiv(self,sample, explainer,values,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index
        shap_values = explainer(self.expr_df_selected)
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.plots.bar(shap_values[index], show=True, max_display = n_features)
        mean_values = np.abs(values[index]).mean(0)
        raw_df = pd.DataFrame({'Feature': self.expr_df_selected.columns, 'mean |SHAP value|': mean_values})
        raw_df = raw_df.sort_values(by=['mean |SHAP value|'], ascending=False)
        raw_df = raw_df.iloc[:n_features]
        transcripts = raw_df['Feature']
        if RawD:
            saveTable(raw_df, sample + "SHAP_bar")
            st.write("  ")
            st.dataframe(raw_df)
        else:
            savePlot(fig, sample + "SHAP_bar")
            st.write("  ")
            st.pyplot(fig)
        return transcripts

    def SHAP_results(self,values):
        shap_sum = np.abs(values).mean(axis=0)
        importance_df = pd.DataFrame([self.expr_df_selected.columns.tolist(), shap_sum.tolist()]).T
        importance_df.columns = ['Transcript', 'SHAP importance']
        importance_df = importance_df.sort_values('SHAP importance', ascending=False)
        return importance_df

    def SHAP_forces(self, sample, explainer,values,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"]==sample].index
        selected_shap_values = values[index, :n_features]
        selected_features = round(self.expr_df_selected.iloc[index, :n_features],2)
        raw_data = pd.DataFrame({'Feature': selected_features.columns, 'Feature value': selected_features.iloc[0].values, 'SHAP value': selected_shap_values[0]})
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, selected_shap_values,selected_features, link="logit",show=True)
        st.write("  ")
        if RawD:
            saveTable(raw_data, sample + "SHAP_force")
            st.write("  ")
            st.dataframe(raw_data)
        else:
            #savePlot(fig, sample + "SHAP_force")
            st.write("  ")
            st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)

    def SHAP_decision(self, sample, explainer,values,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index
        selected_shap_values = values[index, :n_features]
        selected_features = round(self.expr_df_selected.iloc[index, :n_features], 2)
        raw_data = pd.DataFrame({'Feature': selected_features.columns, 'SHAP value': selected_features.iloc[0].values})
        fig, ax = plt.subplots(figsize=(5, 3))
        transcripts = raw_data['Feature']
        shap.decision_plot(explainer.expected_value, selected_shap_values, selected_features)
        if RawD:
            saveTable(raw_data, sample + "SHAP_dec")
            st.dataframe(raw_data)
        else:
            savePlot(fig, sample + "_SHAP_dec")
            st.pyplot(fig)
        return transcripts



    def SHAP_summary(self,values,num_feats):
        fig, ax = plt.subplots()
        shap.summary_plot(values, self.expr_df_selected,show=False, plot_size=(8, 6), color='b', max_display = num_feats)
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

    def SHAP_dependence(self,values, feature):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots()
        shap.dependence_plot(feature, values, self.expr_df_selected, feature_names=self.expr_df_selected.columns, ax= ax)
        savePlot(fig, "PDP")
        st.pyplot(fig)
        st.caption("The x-axis is the actual feature value. "
                   "The y-axis represents the SHAP value for the feature: how much knowing that feature value changes the "
                   "output of the model for that prediction."
                   "The color is a second feature that interacts with the chosen feature.")

class Global(FI):
    def __init__(self, Results):
        super().__init__(Results)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.transcripts = self.expr_df_selected.columns.tolist()

    def global_ALE_mult(self, feature, g1, g2, option):
        rf = super(Global, self).prepareData(self.drug_response)
        if option =="clusters":
            group1, num1 = super(Global,self).displaySamples(g1, "None")
            group2, num2 = super(Global, self).displaySamples(g2, "None")
        elif option == "baskets":
            group1, num1 = super(Global, self).displaySamples("None",g1)
            group2, num2 = super(Global, self).displaySamples("None",g2)
        samples_g1 = group1.values
        samples_g2 = group2.values
        lr_ale = ALE(rf.predict, feature_names=self.transcripts, target_names=['drug response'])
        df1 = self.expr_df_selected.loc[samples_g1].to_numpy()
        df2 = self.expr_df_selected.loc[samples_g2].to_numpy()
        lr_exp1 = lr_ale.explain(df1)
        lr_exp2 = lr_ale.explain(df2)
        index = self.transcripts.index(feature)
        values1 = lr_exp1.data['ale_values'][index]
        values2 = lr_exp2.data['ale_values'][index]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_ale(lr_exp1, features=[feature],ax=ax, line_kw={'label': g1})
        plot_ale(lr_exp2, features=[feature], ax=ax,line_kw={'label': g2})
        min1 = min(values1)
        min2 = min(values2)
        min_limit = min1 if min1<min2 else min2
        max1 = max(values1)
        max2 = max(values2)
        max_limit = max1 if max1 > max2 else max2
        ax.set_ylim(min_limit+(min_limit*2), max_limit+max_limit*2)
        plt.title("ALE for transcript {} in groups {} vs {}".format(feature, g1, g2))
        change1 = max1 - min1
        change2 = max2 - min2
        st.write(
            "##### The magnitude of impact of the feature in the model's prediction is for samples in {} is {}.".format(g2,
                round(change2.tolist()[0], 6)))
        st.write(
            "##### The magnitude of impact of the feature in the model's prediction is for samples in {} is {}.".format(g1,
                round(change1.tolist()[0], 6)))
        st.pyplot(fig)

    def global_ALE(self,feature):
        np.set_printoptions(precision=10)
        rf = super(Global, self).prepareData(self.drug_response)
        lr_ale = ALE(rf.predict, feature_names=self.transcripts, target_names=['drug response'])
        df = self.expr_df_selected.to_numpy()
        lr_exp = lr_ale.explain(df)
        index = self.transcripts.index(feature)
        values = lr_exp.data['ale_values'][index]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_ale(lr_exp, features=[feature], ax=ax)
        ax.set_ylim(min(values)+min(values)*2, max(values)+max(values)*2)
        plt.title("ALE for transcript {}".format(feature))
        change = max(values) - min(values)
        st.write("##### The magnitude of impact of the feature in the model's prediction is {}.".format(round(change.tolist()[0],6)))
        savePlot(fig, "ALE-all")
        st.pyplot(fig)
        st.caption(
            "The x-axis represents the values or intervals of the feature. The y-axis represents the accumulated effects or changes in the"
            " model's predictions.")

    def global_ALE_single(self,feature,g1,g2,option):
        rf = super(Global, self).prepareData(self.drug_response)
        if option =="clusters":
            group1, num1 = super(Global,self).displaySamples(g1, "None")
        elif option == "baskets":
            group1, num1 = super(Global, self).displaySamples("None",g1)
        elif option == "interaction":
            group1, num1 = super(Global, self).displaySamples(g1,g2)
        samples_g1 = group1.values
        lr_ale = ALE(rf.predict, feature_names=self.transcripts, target_names=['drug response'])
        df1 = self.expr_df_selected.loc[samples_g1].to_numpy()
        lr_exp1 = lr_ale.explain(df1)
        index = self.transcripts.index(feature)
        values = lr_exp1.data['ale_values'][index]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_ale(lr_exp1, features=[feature], ax=ax)
        ax.set_ylim(min(values)+min(values)*2, max(values)+max(values)*2)
        plt.title("ALE for transcript {} in group {}".format(feature,option))
        change = max(values) - min(values)
        st.write("##### The magnitude of impact of the feature in the model's prediction is {}.".format(
            round(change.tolist()[0], 6)))
        savePlot(fig, "ALE-single")
        st.pyplot(fig)
        st.caption("The x-axis represents the values or intervals of the feature. The y-axis represents the accumulated effects or changes in the"
                   " model's predictions.")

    def splitResponse(self, resp):
        self.patient_df = self.patient_df.reset_index()
        selection = self.patient_df[(self.patient_df['responsive'] == resp)]
        samples = selection["samples"]
        return samples

    def global_ALE_resp(self,feature):
        rf = super(Global, self).prepareData(self.drug_response)
        samples_g1 = Global.splitResponse(self,'Responsive').values
        samples_g2 = Global.splitResponse(self,'Non-responsive').values
        lr_ale = ALE(rf.predict, feature_names=self.transcripts, target_names=['drug response'])
        df1 = self.expr_df_selected.loc[samples_g1].to_numpy()
        df2 = self.expr_df_selected.loc[samples_g2].to_numpy()
        lr_exp1 = lr_ale.explain(df1)
        lr_exp2 = lr_ale.explain(df2)
        index = self.transcripts.index(feature)
        values1 = lr_exp1.data['ale_values'][index]
        values2 = lr_exp2.data['ale_values'][index]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_ale(lr_exp1, features=[feature], ax=ax, line_kw={'label': "Non-responsive"})
        plot_ale(lr_exp2, features=[feature], ax=ax, line_kw={'label': "Responsive"})
        min1 = min(values1)
        min2 = min(values2)
        min_limit = min1 if min1 < min2 else min2
        max1 = max(values1)
        max2 = max(values2)
        max_limit = max1 if max1 > max2 else max2
        ax.set_ylim(min_limit + (min_limit * 2), max_limit + (max_limit *2))
        plt.title("ALE for transcript {} in groups {} vs {}".format(feature, "Non-responsive", "Responsive"))
        change1 = max1 - min1
        change2 = max2-min2
        st.write("##### The magnitude of impact of the feature in the model's prediction for Responsive samples is {}.".format(
            round(change2.tolist()[0], 6)))
        st.write(
            "##### The magnitude of impact of the feature in the model's prediction for Non-responsive samples is {}.".format(
                round(change1.tolist()[0], 6)))
        savePlot(fig, "ALE-resp")
        st.pyplot(fig)
        st.caption(
            "The x-axis represents the values or intervals of the feature. The y-axis represents the accumulated effects or changes in the"
            " model's predictions.")

