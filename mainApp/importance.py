import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.inspection import permutation_importance
#from rfpimp import permutation_importances
import streamlit as st
import sklearn
import shap
import numpy as np
import pandas as pd
from common import savePlot, saveTable
import seaborn as sns
from PyALE import ale
from alibi.explainers import ALE, plot_ale

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


    def plotImportance(self, RawD):
        importance_sort = self.importance.sort_values('importance_score', ascending=False)
        #importance_sort = importance_sort[::-1]
        importance_vals= importance_sort['importance_score'].values[:25]
        importance_vals = importance_vals[::-1]

        importance_feats = importance_sort.index[:25]
        importance_feats = importance_feats[::-1]
        raw_D = pd.DataFrame({'Features':importance_feats,'Importance score':importance_vals})
        fig = plt.figure(figsize=(12, 6))

        plt.barh(range(len(importance_vals)), importance_vals, align='center', color = sns.color_palette("pastel",25))
        plt.yticks(range(len(importance_vals)), importance_feats)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        if RawD:
            saveTable(raw_D,"RF-FI")
            st.dataframe(raw_D)
        else:
            savePlot(fig, "RF-FI")
            st.pyplot(fig)

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
        explainer = lime_tabular.LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, class_names=["drug response"],
                                                      verbose=True, mode='regression')
        exp = explainer.explain_instance(df.iloc[index[0]], rf.predict, num_features=n_features,)
        fig = exp.as_pyplot_figure()
        raw_data = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
        st.write("##### The predicted value for sample {} is {}".format(sample, round(exp.local_pred[0],3)))
        if RawD:
            saveTable(raw_data, sample + "LIME")
            st.dataframe(raw_data)
        else:
            savePlot(fig,sample+"LIME")
            st.pyplot(fig)
        st.caption(
            "Green values: positive impact, increase model score. Red values: negative impact, decreases model score. ")

    def permutationImportance(self):
        rf = FI.prepareData(self, self.drug_response)

        # the permutation based importance
        perm_importance = permutation_importance(rf, self.X_test, self.y_test,n_repeats=10, random_state=42)

        #sorted_idx = perm_importance.importances_mean.argsort()
        perm_sorted_idx = perm_importance.importances_mean.argsort()
        perm_sorted_idx = perm_sorted_idx[:25]
        perm_sorted_idx = perm_sorted_idx[::-1]
        #top_positive_indices = sorted_idx[-15:]

        # Get the indices of the top 50 negative values
        #top_negative_indices = sorted_idx[:15]

        # Combining the indices of top positive and negative values
        #top_indices = np.concatenate((top_negative_indices, top_positive_indices))

        # The top 50 values (both negative and positive) can be accessed using:
        #top_values = perm_importance.importances_mean[top_indices]
        fig = plt.figure(figsize = (10,8))
        plt.boxplot(
            perm_importance.importances[perm_sorted_idx].T,
            vert=False,
            labels=self.X_test.columns[perm_sorted_idx],
        )
        #plt.barh(self.X_train.columns[top_indices], perm_importance.importances_mean[top_indices])
        plt.xlabel("Permutation Importance")
        return st.pyplot(fig)


        #perm = PermutationImportance(rf, cv=None, refit=False, n_iter=50).fit(X_train, y_train)
        #perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)

    def SHAP(self):
        rf = FI.prepareData(self, self.drug_response)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(self.expr_df_selected)
        return explainer, shap_values

    def SHAP_bar_indiv(self,sample, explainer,values,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"] == sample].index
        shap_values = explainer(self.expr_df_selected)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[index], show=True, max_display=n_features)
        #shap.waterfall_plot(shap_values[index].base_values[0], values[0], self.expr_df_selected.iloc[0])
        mean_values = np.abs(values[index]).mean(0)
        raw_df = pd.DataFrame({'Feature': self.expr_df_selected.columns, 'mean |SHAP value|': mean_values})
        raw_df = raw_df.sort_values(by=['mean |SHAP value|'], ascending=False)
        raw_df = raw_df.iloc[:n_features]
        if RawD:
            saveTable(raw_df, sample + "SHAP_bar")
            st.write("  ")
            st.dataframe(raw_df)
        else:
            savePlot(fig, sample + "SHAP_bar")
            st.write("  ")
            st.pyplot(fig)


    def SHAP_results(self,values):
        shap_sum = np.abs(values).mean(axis=0)
        importance_df = pd.DataFrame([self.expr_df_selected.columns.tolist(), shap_sum.tolist()]).T
        importance_df.columns = ['Transcript', 'SHAP importance']
        importance_df = importance_df.sort_values('SHAP importance', ascending=False)
        return importance_df

    def SHAP_forces(self, sample, explainer,values,n_features,RawD):
        df = self.expr_df_selected.reset_index()
        index = df[df["index"]==sample].index
        #top_feature_indices = np.argsort(np.abs(values[index]))[::-1]
        #top_feature_indices = top_feature_indices[0][:n_features+1]

        selected_shap_values = values[index, :n_features]
        selected_features = round(self.expr_df_selected.iloc[index, :n_features],2)
        #raw_data = pd.DataFrame({'Feature': selected_features.columns,'SHAP value': selected_features.iloc[0].values})
        #fig = shap.force_plot(explainer.expected_value, selected_shap_values, selected_features, matplotlib=True, show=False)
        raw_data = pd.DataFrame({'Feature': selected_features.columns, 'SHAP value': selected_features.iloc[0].values})
        fig = shap.force_plot(explainer.expected_value, selected_shap_values,selected_features, link="logit",matplotlib=True, show=False)
        st.write("  ")
        if RawD:
            saveTable(raw_data, sample + "SHAP")
            st.write("  ")
            st.dataframe(raw_data)
        else:
            savePlot(fig, sample + "SHAP_force")
            st.write("  ")
            st.pyplot(fig)

    def SHAP_bar(self,explainer,sample):
        shap_values = explainer(self.expr_df_selected)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=15)
        #shap.summary_plot(values, self.expr_df_selected, show=False, plot_size=(8, 6), color='b')
        return fig

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
        lr_ale = ALE(rf.predict, feature_names=self.X_train.columns, target_names=['drug response'])
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
        ax.set_ylim(min_limit+(min_limit*2), max_limit+(max_limit/5))
        plt.title("ALE for transcript {} in groups {} vs {}".format(feature, g1, g2))
        return st.pyplot(fig)

    def global_ALE(self,feature):
        rf = super(Global, self).prepareData(self.drug_response)
        lr_ale = ALE(rf.predict, feature_names=self.X_train.columns, target_names=['drug response'])
        self.expr_df_selected = self.expr_df_selected.to_numpy()
        lr_exp = lr_ale.explain(self.expr_df_selected)
        index = self.transcripts.index(feature)
        values = lr_exp.data['ale_values'][index]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_ale(lr_exp, features=[feature], ax=ax)
        ax.set_ylim(min(values) - 0.01, max(values) + 0.01)
        plt.title("ALE for transcript {}".format(feature))
        return st.pyplot(fig)

    def global_ALE_single(self,feature,g1,g2,option):
        rf = super(Global, self).prepareData(self.drug_response)
        if option =="clusters":
            group1, num1 = super(Global,self).displaySamples(g1, "None")
        elif option == "baskets":
            group1, num1 = super(Global, self).displaySamples("None",g1)
        elif option == "interaction":
            group1, num1 = super(Global, self).displaySamples(g1,g2)
        samples_g1 = group1.values
        lr_ale = ALE(rf.predict, feature_names=self.X_train.columns, target_names=['drug response'])
        df1 = self.expr_df_selected.loc[samples_g1].to_numpy()
        lr_exp1 = lr_ale.explain(df1)
        index = self.transcripts.index(feature)
        values = lr_exp1.data['ale_values'][index]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_ale(lr_exp1, features=[feature], ax=ax)
        ax.set_ylim(min(values) +min(values)*2, max(values) +max(values)/5)
        plt.title("ALE for transcript {} in group {}".format(feature,option))
        return st.pyplot(fig)

    def splitResponse(self, resp):
        self.patient_df = self.patient_df.reset_index()
        selection = self.patient_df[(self.patient_df['responsive'] == resp)]
        samples = selection["samples"]
        transcript_df = self.expr_df_selected.loc[samples]
        return samples

    def global_ALE_resp(self,feature):
        rf = super(Global, self).prepareData(self.drug_response)
        samples_g1 = Global.splitResponse(self,0).values
        samples_g2 = Global.splitResponse(self,1).values
        lr_ale = ALE(rf.predict, feature_names=self.X_train.columns, target_names=['drug response'])
        df1 = self.expr_df_selected.loc[samples_g1].to_numpy()
        print(df1)
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
        ax.set_ylim(min_limit + (min_limit * 2), max_limit + (max_limit / 5))
        plt.title("ALE for transcript {} in groups {} vs {}".format(feature, "Non-responsive", "Responsive"))
        return st.pyplot(fig)

