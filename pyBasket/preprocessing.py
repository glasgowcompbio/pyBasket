import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def select_rf(expr_df_filtered, drug_response, n_splits=5, percentile_threshold=90, top_genes=500):

    # Initialize the KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform k-fold cross-validation
    mse_test_scores = []
    r2_test_scores = []
    selected_genes_list = []
    for i, (train_index, test_index) in enumerate(kf.split(expr_df_filtered)):
        # Split the data into training and test sets
        X_train, X_test = expr_df_filtered.iloc[train_index], expr_df_filtered.iloc[test_index]
        y_train, y_test = drug_response.iloc[train_index], drug_response.iloc[test_index]

        y_train = y_train.values.flatten()
        y_test = y_test.values.flatten()

        # Train a Random Forest model on the training data for feature selection
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Rank the importance of each gene based on the decrease in the impurity measure
        importance_scores = rf.feature_importances_

        # Select the top features based on a predefined percentile threshold
        threshold = np.percentile(importance_scores, percentile_threshold)
        selected_genes = np.where(importance_scores >= threshold)[0]
        selected_genes_list.append(selected_genes)

        # Evaluate the quality of the model on the test set
        y_test_pred = rf.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test_scores.append(mse_test)
        r2_test_scores.append(r2_test)

        print(f'Fold {i + 1} - Test set results:')
        print('MSE:', mse_test)
        print('R^2:', r2_test)
        print('-' * 50)

    # Compute the frequency of each selected gene across the k folds
    selected_genes_freq = {}
    for genes in selected_genes_list:
        for gene in genes:
            selected_genes_freq[gene] = selected_genes_freq.get(gene, 0) + 1

    # Select the top 500 genes based on the frequency
    top_genes = sorted(selected_genes_freq.items(), key=lambda x: x[1], reverse=True)[:top_genes]
    top_gene_names = [gene[0] for gene in top_genes]

    # select the top genes from expression dataframe
    expr_df_selected = expr_df_filtered.iloc[:, top_gene_names]
    return expr_df_selected


def check_rf(expr_df_selected, drug_response, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(expr_df_selected, drug_response,
                                                        test_size=test_size, random_state=42)
    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    # Train a final Random Forest model on the entire training set with the top genes
    X_train_selected = X_train
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_final.fit(X_train_selected, y_train)

    # Evaluate the quality of the model on the test set
    X_test_selected = X_test
    y_test_pred = rf_final.predict(X_test_selected)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print('Test set results using selected features:')
    print('MSE:', mse_test)
    print('R^2:', r2_test)