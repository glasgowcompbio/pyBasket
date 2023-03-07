import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.cluster import SpectralCoclustering
from statsmodels.stats.multitest import multipletests


def get_coords(save_data):
    data_df = save_data['patient_df']
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()
    return basket_coords, cluster_coords


def get_predicted_basket_df(save_data):
    basket_coords, cluster_coords = get_coords(save_data)
    stacked = save_data['stacked_posterior']
    inferred_basket = np.mean(stacked.basket_p.values, axis=1)
    predicted_basket_df = pd.DataFrame({'prob': inferred_basket}).set_index(basket_coords)
    return predicted_basket_df


def plot_basket_probs(df):
    plt.figure(figsize=(12, 3))
    ax = sns.barplot(data=df, x=df.index, y='prob')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title('Inferred basket response')
    plt.ylabel('Response probability')


def get_basket_cluster_prob_df(save_data):
    basket_coords, cluster_coords = get_coords(save_data)
    stacked = save_data['stacked_posterior']

    inferred_basket = get_predicted_basket_df(save_data).values
    inferred_cluster = np.mean(stacked.cluster_p.values, axis=2)
    inferred_mat = inferred_basket * inferred_cluster

    inferred_df = pd.DataFrame(inferred_mat, index=basket_coords, columns=cluster_coords)
    return inferred_df


def plot_basket_cluster_heatmap(df, x_highlight, y_highlight):
    plt.figure(figsize=(7, 7))
    sns.heatmap(data=df, cmap='Blues')
    plt.title('Inferred basket/cluster combination')
    plt.xlabel('Clusters')
    plt.ylabel('Baskets')

    # Highlight a single cell by drawing a rectangle around it
    plt.gca().add_patch(
        plt.Rectangle((x_highlight, y_highlight), 1, 1, fill=False, edgecolor='red', lw=3))


def find_top_k_indices(save_data, k):
    basket_coords, cluster_coords = get_coords(save_data)
    inferred_df = get_basket_cluster_prob_df(save_data)
    arr = inferred_df.values

    # use np.argpartition() to get indices of top k values
    top_k_indices = np.argpartition(arr.flatten(), -k)[-k:]

    # use np.unravel_index() to convert 1D indices to 2D indices
    top_k_indices = np.unravel_index(top_k_indices, arr.shape)

    # get values of top k indices
    top_k_values = arr[top_k_indices]

    # sort indices and values in descending order
    sort_indices = np.argsort(top_k_values)[::-1]
    top_k_indices = (top_k_indices[0][sort_indices], top_k_indices[1][sort_indices])
    top_k_values = top_k_values[sort_indices]

    # convert to dataframe
    df = pd.DataFrame({
        'basket_idx': top_k_indices[0],
        'cluster_idx': top_k_indices[1],
        'basket': basket_coords[top_k_indices[0]],
        'cluster': cluster_coords[top_k_indices[1]],
        'probability': top_k_values
    })

    df_count = get_member_counts(save_data, df)
    join_df = df.join(df_count, how='inner')
    return join_df


def find_bottom_k_indices(save_data, k):
    basket_coords, cluster_coords = get_coords(save_data)
    inferred_df = get_basket_cluster_prob_df(save_data)
    arr = inferred_df.values

    # use np.argpartition() to get indices of bottom k values
    bottom_k_indices = np.argpartition(arr.flatten(), k)[:k]

    # use np.partition() to get values of bottom k indices
    bottom_k_values = arr.flatten()[bottom_k_indices]
    bottom_k_values = np.partition(bottom_k_values, k - 1)[:k]

    # sort indices and values in ascending order
    sort_indices = np.argsort(bottom_k_values)
    bottom_k_indices = bottom_k_indices[sort_indices]
    bottom_k_values = bottom_k_values[sort_indices]

    # reshape indices back to 2D array shape
    bottom_k_indices = np.unravel_index(bottom_k_indices, arr.shape)

    # convert to dataframe
    df = pd.DataFrame({
        'basket_idx': bottom_k_indices[0],
        'cluster_idx': bottom_k_indices[1],
        'basket': basket_coords[bottom_k_indices[0]],
        'cluster': cluster_coords[bottom_k_indices[1]],
        'probability': bottom_k_values
    })

    df_count = get_member_counts(save_data, df)
    join_df = df.join(df_count, how='inner')
    return join_df


def select_partition(save_data, query_basket, query_cluster):
    df = save_data['patient_df']
    filtered_df = df[(df['tissues'] == query_basket) & (df['cluster_number'] == query_cluster)]
    return filtered_df


def get_member_counts(save_data, df):
    counts = []
    for idx, row in df.iterrows():
        member_count = len(select_partition(save_data, row['basket'], row['cluster']))
        counts.append([member_count])
    count_df = pd.DataFrame(counts, columns=['count']).set_index(df.index)
    return count_df


def plot_responsive_count(selected_df):
    non_responsive_count = np.sum(selected_df['responsive'] == 0)
    responsive_count = np.sum(selected_df['responsive'] == 1)
    count_df = pd.DataFrame({
        'response': ['non-responsive', 'responsive'],
        'count': [non_responsive_count, responsive_count]
    })

    plt.figure(figsize=(5, 5))
    ax = sns.barplot(data=count_df, x='response', y='count')
    ax.set_xticklabels([0, 1])
    plt.title('Sample count')
    plt.xlabel(None)


def get_member_expression(selected_df, save_data):
    members = selected_df.index
    expr_df_selected = save_data['expr_df_selected']
    member_df = expr_df_selected.loc[members]
    return member_df


def plot_bicluster_partition(member_df):
    # perform biclustering using Spectral Co-Clustering
    model = SpectralCoclustering(n_clusters=10, random_state=0)
    model.fit(member_df.values)

    # reorder rows and columns based on bicluster assignments
    member_df_reorder = member_df.iloc[np.argsort(model.row_labels_)]
    member_df_reorder = member_df_reorder.iloc[:, np.argsort(model.column_labels_)]
    sns.heatmap(member_df_reorder)


def df_diff(df1, df2):
    indices_to_drop = df1.index
    df2_filtered = df2.drop(indices_to_drop)
    return df2_filtered


def ttest_dataframe(df1, df2, only_significant=False):
    """
    Performs t-tests between corresponding columns in two dataframes and returns a new dataframe with the p-value
    and test statistic for each comparison, corrected for multiple testing using the Bonferroni correction.
    If only_significant is True (default), only significant columns are returned, and the output is sorted by
    'P-Value (Bonferroni)' in descending order.
    """
    n_tests = len(df1.columns)

    ttest_results = []
    for column in df1.columns:
        t, p = ttest_ind(df1[column], df2[column])
        ttest_results.append((column, t, p))

    results_df = pd.DataFrame(ttest_results, columns=['Feature', 'T-Statistic', 'P-Value'])
    _, results_df['P-Value (Bonferroni)'], _, _ = multipletests(results_df['P-Value'],
                                                                method='bonferroni')
    results_df['Significant'] = results_df['P-Value (Bonferroni)'] < 0.05

    if only_significant:
        results_df = results_df.loc[results_df['Significant']].sort_values(by=['T-Statistic'],
                                                                           key=abs,
                                                                           ascending=False)

    return results_df.set_index('Feature')


def plot_expression_boxplot(selected_feature, member_df, all_expr_df):
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [member_df[selected_feature].values, all_expr_df[selected_feature].values]
    positions = [1, 2]

    ax.boxplot(data, positions=positions, labels=['Members', 'All'])
    ax.set_title(selected_feature)
    ax.set_ylabel('Normalised Count')
