#!/usr/bin/env python
# coding: utf-8

# # CS 598 Practical Statistical Learning: Project 2, Fall 2023
#
# ### TO RUN THIS SCRIPT ###
#     The script looks for files named "train.csv" and "test.csv" in the same directory.
#     Example run:   > python mymain.py
#
# Team members:
# * Kurt Tuohy (ktuohy): SVD implementation
# * Neal Ryan (nealpr2): most data preprocessing, OLS implementation
# * Alelign Faris (faris2): mymain.py

# ## Import Libraries and Load Data

# Import libraries
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Set seed
SEED = 4031
np.random.seed(SEED)

# Lists for training and test datasets

train_datasets = []
test_datasets = []
train_datasets.append(pd.read_csv("train.csv"))
test_datasets.append(pd.read_csv("test.csv"))

# Preprocessing steps
# 1. Group training data by department
# 2. Pivot each department's data so that stores are rows and dates are columns, with values = weekly sales
# 3. Center store values
# 4. Perform SVD
# 5. Re-add store means
# 6. Use the SVD output as y_train for x_train = [Year, Year^2, Week, Store]

n_components = 8


def smooth_weekly_sales(train_data, fold, n_components=n_components):
    """
    Given a fold's training dataset of weekly sales by store and department,
    use SVD to smooth each department's weekly sales across stores.
    Return the dataset with smoothed sales.
    """

    # Store each department's processed data
    smoothed_sales_list = []

    t_split = dict(tuple(train_data.groupby(["Dept"])))

    depts = list(t_split)

    for dept in depts:

        # Get needed columns for department
        t_data = t_split[dept]

        # Pivot each store/dept combo's sales by date.
        # Rows and stores are depts, columns are dates, values are sales figures.
        t_pivot = t_data.pivot(index=["Store", "Dept"], columns="Date", values="Weekly_Sales").reset_index().fillna(0)

        # Save list of Date values for dept
        t_dates = t_pivot.columns[2:]

        # Extract just the dates and sales for further processing
        t_sales = t_pivot.to_numpy()[:, 2:]
        t_dept_store = t_pivot.to_numpy()[:, :2]

        # Get store means
        t_store_means = np.mean(t_sales, axis=1)[:, np.newaxis]

        # Center the sales by store
        t_centered_sales = t_sales - t_store_means

        # Perform SVD on centered sales data if there are more than n_components components
        if t_centered_sales.shape[0] > n_components:
            U, S, V = np.linalg.svd(t_centered_sales)

            # Keep only the top n_components components
            U_reduced = U[:, :n_components]
            S_reduced = np.diag(S[:n_components])
            V_reduced = V[:n_components, :]

            # Reconstruct smoothed sales
            t_sales_smooth = np.dot(U_reduced, np.dot(S_reduced, V_reduced)) + t_store_means
        else:
            # If there are not enough components to perform SVD, use original sales
            t_sales_smooth = t_sales

        # Join Store and Dept back to smoothed sales figures
        t_dept_store_sales_smooth = np.concatenate((t_dept_store, t_sales_smooth), axis=1)
        t_columns = ["Store", "Dept"] + t_dates.tolist()

        # Convert smoothed sales from array to dataframe
        t_dept_store_sales_smooth_df = pd.DataFrame(t_dept_store_sales_smooth, columns=t_columns)

        # Unpivot the smoothed sales
        t_sales_unpivot = t_dept_store_sales_smooth_df.melt(id_vars=["Store", "Dept"],
                                                            var_name="Date",
                                                            value_name="Weekly_Sales").reset_index(drop=True)

        # Save dept's smoothed sales
        smoothed_sales_list.append(t_sales_unpivot)

    # Stack the accumulated smoothed sales
    t_dept_store_sales = pd.concat(smoothed_sales_list, axis=0, ignore_index=True).reset_index(drop=True)

    return t_dept_store_sales


def dates_to_years_and_weeks(data):
    """
    Convert sales dates in data to numeric years and categorical weeks.
    """

    tmp = pd.to_datetime(data["Date"])
    data["Wk"] = tmp.dt.isocalendar().week
    data["Yr"] = tmp.dt.year
    data["Wk"] = pd.Categorical(data["Wk"], categories=[i for i in range(1, 53)])  # 52 weeks

    return data


n_datasets = len(train_datasets)

# Loop over folds
for j in range(n_datasets):

    # Get a pair of training and test sets
    train = train_datasets[j]
    test = test_datasets[j]

    test_pred = pd.DataFrame()

    # Identify the distinct store/dept pairs shared by the training and test set.
    # Will only process these.

    train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how='inner', on=['Store', 'Dept'])

    # Join the distinct store/dept pairs to the training set.
    train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')

    # Now join the distinct store/dept pairs to the test set.
    test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')

    # Smooth weekly sales in the training dataset
    train_smooth = smooth_weekly_sales(train_split, j, n_components)

    # Convert training sales dates to years + weeks
    train_split = dates_to_years_and_weeks(train_smooth)
    train_split = train_split.sort_values(by=['Store', 'Dept', 'Date'], ascending=[True, True, True])

    # Get design matrices for training y and X.
    # y is just the target variable, Weekly_Sales.
    # X has pivoted weeks, where individual weeks are separate 0/1 columns.
    y, X = patsy.dmatrices('Weekly_Sales ~ Weekly_Sales + Store + Dept + Yr  + Wk',
                           data=train_split,
                           return_type='dataframe')

    # Group by store and department to help build separate model for store/dept combo.
    # Create dict where key = (Store, Dept) and value = dataframe of Store/Date/Weekly_Sales.
    train_split = dict(tuple(X.groupby(["Store", "Dept"])))

    # Convert test sales dates to years + weeks
    test_split = dates_to_years_and_weeks(test_split)

    # Get design matrices for text y and X.
    # y is the Year, and the design matrix is "Store + Dept + Yr + Wk".
    # Note that test sets don't have the Weekly_Sales target variable.
    y, X = patsy.dmatrices('Yr ~ Store + Dept + Yr  + Wk',
                           data=test_split,
                           return_type='dataframe')

    # Re-add Date column to the design matrix X
    X['Date'] = test_split['Date']
    # Get dictionary where keys are (Store, Dept) tuples, and values are the
    # "Yr  + Wk + Date" design matrices corresponding to each key.
    test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    # Get the training departments
    keys = list(train_split)

    # Loop over (store, dept) tuples
    for key in keys:

        # Get training and test design matrices corresponding to (store, dept)
        X_train = train_split[key]
        X_test = test_split[key]

        # Target variable for (store, dept)
        Y = X_train['Weekly_Sales']
        # Drop ID and target to get just a table of predictors
        X_train = X_train.drop(['Weekly_Sales', 'Store', 'Dept'], axis=1)

        # Identify columns that are all zero in training predictors, and drop them
        # from both training and test X.
        # This should drop weeks that are not represented in the training data.
        # How does this affect test X? Are there cases where all test weeks would be dropped?
        cols_to_drop = X_train.columns[(X_train == 0).all()]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        # Identify X training columns that are highly collinear with the columns to the left.
        # Note that this doesn't check the Intercept column.
        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
            col_name = X_train.columns[i]
            # Extract the current column and all previous columns
            tmp_Y = X_train.iloc[:, i].values
            tmp_X = X_train.iloc[:, :i].values

            coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
            if np.sum(residuals) < 1e-10:
                cols_to_drop.append(col_name)

        # Drop those collinear columns from both training and test X.
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        # add quadratic Yrs
        try:
            X_train["Yr^2"] = X_train["Yr"] * X_train["Yr"]
            X_test["Yr^2"] = X_test["Yr"] * X_test["Yr"]
        except:
            pass

        # Fit a regular ordinary least squares model on training Weekly_Sales.
        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)

        tmp_pred = X_test[['Store', 'Dept', 'Date']]
        X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)

        tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

    test_pred['Weekly_Pred'].fillna(0, inplace=True)

    # Replace negative predictions with 0
    test_pred.loc[test_pred['Weekly_Pred'] < 0, 'Weekly_Pred'] = 0

    # Load test data from the given file
    test_data = pd.read_csv("test.csv")

    # Copy the 'IsHoliday' column from test_data to test_pred
    test_pred['IsHoliday'] = test_data['IsHoliday']

    # Select and reorder the columns in test_pred
    new_cols = ['Store', 'Dept', 'Date', 'IsHoliday', 'Weekly_Pred']
    test_pred = test_pred[new_cols]

    # Convert 'Store' and 'Dept' columns to int64 data type
    test_pred[['Store', 'Dept']] = test_pred[['Store', 'Dept']].astype(np.int64)

    # Round the 'Weekly_Pred' column to 2 decimal places
    test_pred['Weekly_Pred'] = test_pred['Weekly_Pred'].round(2)

    # Save the output to CSV
    file_path = f'mypred.csv'
    test_pred.to_csv(file_path, index=False)
