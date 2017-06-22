import pandas as pd
import numpy as np
import timeit as t

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# reads in the files from csv, combines them for our train and test dfs and returns them
# NOT VERIFIED
def read_raw_data_from_file(fraction, seed):
    result_df = pd.read_csv('data/train_2016.csv', index_col='parcelid', parse_dates=['transactiondate'])
    prop_df = pd.read_csv('data/properties_2016.csv', index_col='parcelid')

    test_df = prop_df[~prop_df.index.duplicated(keep='first')]

    train_df = result_df.join(test_df)

    return train_df.sample(frac=fraction, random_state=seed), test_df.sample(frac=fraction, random_state=seed)

# VERIFIED
def read_sample_data_from_file():
    return pd.read_csv('data/train_sample.csv',index_col='parcelid', parse_dates=['transactiondate']), pd.read_csv('data/test_sample.csv', index_col='parcelid')

def read_cleaned_data_from_file():
    return pd.read_csv('data/train_cleaned.csv',index_col='parcelid'), pd.read_csv('data/test_cleaned.csv', index_col='parcelid')

'''
# NOT VERIFIED
def normalize_categorical(column, train_df, test_df):
    # grouping the rows in train_df into groups of the same value in column and getting the mean of every column in each group
    group_df = train_df.groupby(column).mean()
    min_mean, max_mean = group_df['logerror'].min(), group_df['logerror'].max()
    # creating a new column in group that contains the normalized order of each group
    group_df['normalized'] = group_df.index.map(lambda row: (group_df.loc[row]['logerror'] - min_mean) / (max_mean - min_mean))
    # return appropriate normalized values in train_df and test_df, if the value is not in train_df, returning the mode in the normalized value
    return train_df[column].map(lambda row: group_df.loc[row]['normalized']), test_df[column].apply(lambda row: group_df.loc[row]['normalized'] if row in group_df.index else group_df.loc[train_df[column].mode()[0]]['normalized'])

# NOT VERIFIED
def standardize_categorical(column, train_df, test_df):
    # grouping the rows in train_df into groups of the same value in column and getting the mean of every column in each group
    group_df = train_df.groupby(column).mean()
    mean, std = group_df['logerror'].mean(), group_df['logerror'].std()
    # creating a new column in group that contains the standardized order of each group
    group_df['standardized'] = group_df.index.map(lambda row: (group_df.loc[row]['logerror'] - mean) / std)
    # return appropriate standardized values in train_df and test_df, if the value is not in train_df, returning the mode in the standardized value
    return train_df[column].map(lambda row: group_df.loc[row]['standardized']), test_df[column].apply(lambda row: group_df.loc[row]['standardized'] if row in group_df.index else group_df.loc[train_df[column].mode()[0]]['standardized'])
'''

# uses the condition variable to decide on how to deal with the column var
# VERIFIED
def conditional_delete_norm_std(condition, column, train_df, test_df):
    if condition ==0:
        return train_df.drop(column, axis = 1) , test_df.drop(column, axis = 1)
    elif condition == 1:
        return normalize_categorical(column, train_df, test_df)
    else:
        return standardize_categorical(column, train_df, test_df)

# drops columns that over fit the data as decided by a heatmap, where a column with more than a 90% correlation with another column would be dropped
# then uses threshold var to drop columns with more than threshold% of missing variables
# VERIFIED
def drop_columns_high_corr(corr_threshold, train_df, test_df):
    # dropping the columns that have strong correlation with other columns, keeping only from each group
    corr_matrix = train_df.corr()
    #corr_matrix.to_csv('corr2.csv')

    columns = train_df.columns[3:].copy()

    for index1 in columns:
        for index2 in columns:
            value = corr_matrix.loc[index1][index2]
            if (abs(value) > corr_threshold and index1 != index2 and index1 in train_df.columns and index2 in train_df.columns):
                #print("---------")
                #print(index1, " corr with logerror:", corr_matrix.loc[index1]['logerror'])
                #print(index2, " corr with logerror:", corr_matrix.loc[index2]['logerror'])
                if (abs(corr_matrix.loc[index1]['logerror'])>abs(corr_matrix.loc[index2]['logerror'])):
                    #print("Dropping ", index2)
                    train_df = train_df.drop(index2, axis=1)
                    test_df = test_df.drop(index2, axis=1)
                else:
                    #print("Dropping ", index1)
                    train_df = train_df.drop(index1, axis=1)
                    test_df = test_df.drop(index1, axis=1)

    #corr_matrix = train_df.corr()
    #corr_matrix.to_csv('corr2_after.csv')

    return train_df, test_df

# VERIFIED
def drop_columns_little_info(missing_threshold, train_df, test_df):
    # dropping columns with 0 or 1 unique value
    for column in train_df.columns:
        count = train_df[column].value_counts()
        if count.size < 2:
            train_df = train_df.drop(column, axis=1)
            test_df = test_df.drop(column, axis=1)

    # computing the percentage of missing data in each column in train_df
    null_tf = train_df.isnull()
    sum_missing = null_tf.sum()
    percent_missing = sum_missing / null_tf.count()

    # dropping the columns with missing data percent higher than missing_threshold
    train_df = train_df.drop((percent_missing[percent_missing > missing_threshold]).index, axis=1)
    test_df = test_df.drop((percent_missing[percent_missing > missing_threshold]).index, axis=1)

    return train_df, test_df

# fills in the missing values of the columns in each df
# VERIFIED
def fill_missing(train_df, test_df):
    # list used for those pieces of data which are continuous
    mean_list = ['calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'latitude', 'longitude', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']

    # list used for categorical data both in int/float or string forms
    mode_list = ['bathroomcnt', 'fullbathcnt', 'rawcensustractandblock', 'censustractandblock', 'heatingorsystemtypeid', 'buildingqualitytypeid', 'unitcnt', 'regionidcity', 'regionidzip', 'propertylandusetypeid', 'regionidcounty', 'fips', 'calculatedbathnbr','yearbuilt','bedroomcnt', 'roomcnt'] # 'propertyzoningdesc',  'propertycountylandusecode'

    for column in mode_list:
        if column in train_df.columns:
            # Some positive numerical columns (e.g. roomcnt) have 0 (as opposed to NaN) as data
            # Making data that should be positive be NaN if they contain 0
            # Then compute mode based on the non-NaN values, replace the NaN values with the mode
            train_df[column] = train_df[column].apply(lambda x: np.nan if x == 0 else x) #and type(x) is not object else x)    # for a column in train_df, change any zero values to Nan
            train_df[column] = train_df[column].fillna(test_df[column].mode()[0]) #fill Nan values with mode values
            test_df[column] = test_df[column].apply(lambda x: np.nan if x == 0 else x) # and type(x) is not object else x)
            test_df[column] = test_df[column].fillna(test_df[column].mode()[0])

    for column in mean_list:
        # not handling 0 here because no column in the mean list contains 0
        train_df[column] = train_df[column].fillna(test_df[column].mean())
        test_df[column] = test_df[column].fillna(test_df[column].mean())

    return train_df, test_df

# changes the transaction date into only the month present
def transaction_date(train_df):
    train_df['transaction_month'] = train_df['transactiondate'].dt.month
    train_df = train_df.drop('transactiondate', axis=1)
    return train_df

# converts specified values dtype's to int
def convert_to_int(train_df, test_df):
    int_list = ['buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertylandusetypeid', 'regionidcity', 'regionidcounty', 'regionidzip', 'yearbuilt']

    for column in int_list:
        if column in train_df.columns:
            train_df[column] = train_df[column].astype(int)
            test_df[column] = test_df[column].astype(int)

    return train_df, test_df

def scale(train_df, test_df, type=0):
    scaler_list = [
        ('Min Max', MinMaxScaler()),
        ('Normalizer', Normalizer()),
        ('Standardizer', StandardScaler())
    ]

    #train_df = train_df.drop(['censustractandblock', 'lotsizesquarefeet', 'unitcnt'], axis = 1)
    #test_df = test_df.drop(['censustractandblock', 'lotsizesquarefeet', 'unitcnt'], axis = 1)
    trans_list = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedfloor1squarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'landtaxdollarvaluecnt', 'lotsizesquarefeet', 'numberofstories', 'poolcnt', 'poolsizesum', 'roomcnt', 'structuretaxvaluedollarcnt', 'taxamount', 'taxdelinquencyflag', 'taxvaluedollarcnt', 'threequarterbathnbr', 'unitcnt']
    new_trans_list = []

    for item in trans_list:
        if item in train_df:
            new_trans_list.append(item)
    if type >= len(scaler_list): quit('Change scaler_num!')
    name, scaler = scaler_list[type]
    print('Scaler', name, 'used.')

    test_df[new_trans_list] = scaler.fit_transform(test_df[new_trans_list])
    train_df = train_df[['logerror', 'transaction_month']].join(test_df)

    return train_df, test_df

# attempts to correct skewness in data by transforming the data points by applying the log function to them
def transform(train_df, test_df):
    trans_list = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedfloor1squarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'landtaxdollarvaluecnt', 'lotsizesquarefeet', 'numberofstories', 'poolcnt', 'poolsizesum', 'roomcnt', 'structuretaxvaluedollarcnt', 'taxamount', 'taxdelinquencyflag', 'taxvaluedollarcont', 'threequarterbathnbr', 'unitcnt']
    print('Train', '-' * 20)
    for column in train_df.columns: print(column, train_df[column].skew())
    print('Test', '-' * 20)
    for column in test_df.columns: print(column, test_df[column].skew())

    for column in test_df.columns:
        if column in trans_list:
            column_max = test_df[column].max()
            column_skew = test_df[column].skew()

            if column_skew < -1:
                train_df[column] = train_df[column].apply(lambda x: np.log1p(column_max - x))
                test_df[column] = test_df[column].apply(lambda x: np.log1p(column_max - x))

            elif column_skew < -0.5:
                train_df[column] = train_df[column].apply(lambda x: np.sqrt(column_max - x))
                test_df[column] = test_df[column].apply(lambda x: np.sqrt(column_max - x))

            elif column_skew > 0.5 and column_skew <= 1:
                train_df[column] = train_df[column].apply(np.sqrt)
                test_df[column] = test_df[column].apply(np.sqrt)

            elif column_skew > 1:
                train_df[column] = train_df[column].apply(np.log1p)
                test_df[column] = test_df[column].apply(np.log1p)

    print('Train', '-' * 20)
    for column in train_df.columns: print(column, train_df[column].skew())
    print('Test', '-' * 20)
    for column in test_df.columns: print(column, test_df[column].skew())

    return train_df, test_df

#runs the regression algorithms taking in train and test df,
#k is the number of k folds, if not greater than 1, no k fold cross validation will be done
#DEFAULT: k = 0, sample_frac = .9, seed =1; unless specified
def regression_analysis(train_df, test_df, k=0, sample_frac=.9, seed=1):
    # list used for all models for our ensemble
    model_list = [
        #('SVR', SVR()),
        #('Random Forest', RandomForestRegressor()),
        #('K Neighbor', KNeighborsRegressor()),
        #('Decision Tree', DecisionTreeRegressor()),
        ('Linear Regression', LinearRegression()),
    ]

    # looping for running the models in model_list
    if k>1:
        for name, model in model_list:
            start = t.default_timer()
            print(name)
            model.fit(train_df.drop('logerror', axis = 1), train_df['logerror'])
            mean_score = np.mean(cross_val_score(model, train_df.drop('logerror', axis = 1), train_df['logerror'], cv = k, scoring = make_scorer(mean_absolute_error)))
            print(mean_score)
            print('Took:', t.default_timer() - start)
            # for use when trying to split and then fit the data
    else:
        #runs train_test_split function on the data frame
        x_train, x_test, y_train, y_test = train_test_split(train_df.drop('logerror', axis=1), train_df['logerror'], train_size=sample_frac, random_state=seed)
        for name, model in model_list:
            start = t.default_timer()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = mean_absolute_error(y_test, pred)
            print(name)
            print(score)
            print('Took:', t.default_timer() - start)


def regression_predict(train_df, test_df):
    list_dates = ['201610', '201611', '201612']
    predict1 = pd.DataFrame({'parcelid': test_df.index})
    predict1 = predict1.set_index('parcelid')
    predict2 = pd.DataFrame({'parcelid': test_df.index})
    predict2 = predict2.set_index('parcelid')

    alg = LinearRegression()
    alg.fit(train_df.drop('logerror', axis = 1), train_df['logerror'])

    for date in list_dates:
        test_df['transaction_month'] = date[-2:]
        predict1[date] = alg.predict(test_df)
        predict2['2017'+date[-2:]] = predict1[date].copy()

    predict = predict1.join(predict2)
    predict.to_csv('data/custom_sub.csv')

def main():
    '''train_df, test_df = read_raw_data_from_file(1, 1)
    train_df.to_csv('data/train_sample.csv')
    test_df.to_csv('data/test_sample.csv')
    print('Done loading in data...')
    train_df, test_df = read_sample_data_from_file()

    # Delete two attributes with many categorical values
    train_df, test_df = conditional_delete_norm_std(0, 'propertyzoningdesc', train_df, test_df)
    train_df, test_df = conditional_delete_norm_std(0, 'propertycountylandusecode', train_df, test_df)

    # Drops the columns which have a high number of missing values
    train_df, test_df = drop_columns_little_info(0.5, train_df, test_df)

    # Fills missing or Nan values in columns with means or modes
    train_df, test_df = fill_missing(train_df, test_df)

    # Drops the columns which have a high corelation to things other than parcelid or logerror or transactionmonth
    train_df, test_df = drop_columns_high_corr(0.9, train_df, test_df)

    train_df = transaction_date(train_df)
    train_df, test_df = convert_to_int(train_df, test_df)

    train_df.to_csv('data/train_cleaned.csv')
    test_df.to_csv('data/test_cleaned.csv')
    print('Done!')'''

    train_df, test_df = read_cleaned_data_from_file()
    train_df, test_df = scale(train_df, test_df)
    train_df, test_df = transform(train_df, test_df)

    #regression_analysis(train_df, test_df, k=5)
    regression_predict(train_df, test_df)

main()
