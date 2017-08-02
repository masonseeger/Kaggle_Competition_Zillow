import pandas as pd
import numpy as np

from knnimpute import knn_impute_optimistic

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, StandardScaler
from scipy.stats import boxcox

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from timeit import default_timer as timer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

def read_raw_data():
    result_df = pd.read_csv('data/train_2016.csv', index_col='parcelid', parse_dates=['transactiondate'])
    prop_df = pd.read_csv('data/properties_2016.csv', index_col='parcelid')
    
    result_df = result_df.drop(11016594)
    
    result_df = result_df.join(prop_df)
    
    return result_df, prop_df

def read_encoded_data():
    return pd.read_csv('data/train_new_cleaned.csv', index_col='parcelid'), pd.read_csv('data/test_new_cleaned.csv', index_col='parcelid')

def read_scaled_data_cat():
    return pd.read_csv('data/train_scaled_nocat.csv', index_col='parcelid'), pd.read_csv('data/test_scaled_nocat.csv', index_col='parcelid'), pd.read_csv('data/cat_columns.csv', index_col='parcelid')

def read_cat_processed():
    return pd.read_csv('data/train_scaled_cat.csv', index_col='parcelid'), pd.read_csv('data/test_scaled_cat.csv', index_col='parcelid')

# VERIFIED
def drop_outliers_logerror(train_df, multiplier):
    '''des = train_df['logerror'].describe()
    print('Summary of "logerror" column:')
    print(des)
    
    lower_quartile = des.loc['25%']
    upper_quartile = des.loc['75%']
    interquartile = upper_quartile - lower_quartile
    
    print('Computing fences with multiplier %f...' % (multiplier))
    outer_fences = [lower_quartile - interquartile * multiplier, upper_quartile + interquartile * multiplier]
    print('Computed outer fences:', outer_fences)
    
    train_df = train_df[(train_df['logerror'] > outer_fences[0]) & (train_df['logerror'] < outer_fences[1])]'''
    
    train_df = train_df[train_df['logerror'] > -0.402925]
    train_df = train_df[train_df['logerror'] < 0.415825]
    
    return train_df

# VERIFIED
def fill_flag(train_df, test_df):
    flag_dict = {'fireplaceflag': True, 'hashottuborspa': True, 'taxdelinquencyflag': 'Y'}
    
    for column in flag_dict:
        test_df[column] = test_df[column].apply(lambda x: 1 if x == flag_dict[column] else 0)
        train_df[column] = train_df[column].apply(lambda x: 1 if x == flag_dict[column] else 0)
        print('Processing %s...' % (column))
        print(test_df[column].head())
        print(test_df[column].value_counts())
    
    test_df['poolcnt'] = test_df['poolcnt'].fillna(0)
    train_df['poolcnt'] = train_df['poolcnt'].fillna(0)
    
    return train_df, test_df

# drops columns that over fit the data as decided by a heatmap, where a column with more than a 90% correlation with another column would be dropped
# then uses threshold var to drop columns with more than threshold% of missing variables
# VERIFIED
def drop_columns_high_corr(corr_threshold, train_df, test_df):
    corr_matrix = train_df.corr()

    columns = test_df.columns

    for index1 in columns:
        for index2 in columns:
            value = corr_matrix.loc[index1][index2]
            if (abs(value) > corr_threshold and index1 != index2 and index1 in train_df.columns and index2 in train_df.columns):
                if (abs(corr_matrix.loc[index1]['logerror'])>abs(corr_matrix.loc[index2]['logerror'])):
                    print("Dropping ", index2)
                    train_df = train_df.drop(index2, axis=1)
                    test_df = test_df.drop(index2, axis=1)
                
                else:
                    print("Dropping ", index1)
                    train_df = train_df.drop(index1, axis=1)
                    test_df = test_df.drop(index1, axis=1)

    return train_df, test_df

# VERIFIED
def drop_columns_little_info(missing_threshold, train_df, test_df):
    for column in train_df.columns:
        count = train_df[column].value_counts()
        if count.size < 2:
            train_df = train_df.drop(column, axis=1)
            test_df = test_df.drop(column, axis=1)

    temp_df = pd.DataFrame()
    for column in train_df.columns:
        if train_df[column].isnull().sum()>0:
            temp_df[column] = train_df[column]

    temp_df.to_csv('temp_df.csv')

    objects = (null_tf.columns)
    y_pos = np.arange(len(objects))
    performance = percent_missing
    plt.barh(y_pos, performance, align ='center', alpha = 0.5)
    plt.yticks(y_pos, objects)
    #plt.invert_yaxis()
    plt.xlabel('Percent Missing')
    plt.title('Missing Value Percentages Per Attribute')

    plt.show()

    train_df = train_df.drop((percent_missing[percent_missing > missing_threshold]).index, axis=1)
    test_df = test_df.drop((percent_missing[percent_missing > missing_threshold]).index, axis=1)

    return train_df, test_df

# VERIFIED
def str_to_num_encode(train_df, test_df):
    str_list = ['propertyzoningdesc', 'propertycountylandusecode']
    
    encoder = LabelEncoder()
    
    for column in str_list:
        test_df[column] = encoder.fit_transform(test_df[column].astype('str'))
    
    train_df = train_df[['logerror', 'transaction_month']].join(test_df)
    
    return train_df, test_df

# fills in the missing values of the columns in each df
# VERIFIED
def fill_missing(train_df, test_df):
    mean_list = ['calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'latitude', 'longitude', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']

    mode_list = ['bathroomcnt', 'fullbathcnt', 'rawcensustractandblock', 'censustractandblock', 'heatingorsystemtypeid', 'buildingqualitytypeid', 'unitcnt', 'regionidcity', 'regionidzip', 'propertylandusetypeid', 'regionidcounty', 'fips', 'calculatedbathnbr','yearbuilt','bedroomcnt', 'roomcnt']
    
    for column in mode_list:
        if column in train_df.columns:
            train_df[column] = train_df[column].apply(lambda x: np.nan if x == 0 else x)
            train_df[column] = train_df[column].fillna(test_df[column].mode()[0])
            test_df[column] = test_df[column].apply(lambda x: np.nan if x == 0 else x)
            test_df[column] = test_df[column].fillna(test_df[column].mode()[0])

    for column in mean_list:
        train_df[column] = train_df[column].fillna(test_df[column].mean())
        test_df[column] = test_df[column].fillna(test_df[column].mean())
    
    for column in test_df.columns:print(column, test_df[column].isnull().sum())
    
    # NOT VERIFIED
    null_tf = test_df.isnull()
    if null_tf.any().any():
        print('****************************\n***********************\n************************\n************************\nUsing KNN to fill missing values...')
        test_df = pd.DataFrame(knn_impute_optimistic(test_df.values, null_tf.values, 1), columns=test_df.columns)
    
    train_df = train_df[['logerror', 'transaction_month']].join(test_df)
    
    return train_df, test_df

# changes the transaction date into only the month present
# VERIFIED
def transaction_date(train_df):
    train_df['transaction_month'] = train_df['transactiondate'].dt.month
    train_df = train_df.drop('transactiondate', axis=1)
    return train_df

# VERIFIED
def cat_split(train_df, test_df, ascending=True):
    attr_to_corr = {
        'heatingorsystemtypeid': -0.00078720875264967497,
        'propertylandusetypeid': 0.0013902076759614891,
        'propertyzoningdesc': 6.4796611646452326e-05,
        'regionidcity': 0.00093619783205819832,
        'regionidzip': -3.9298250892690326e-05,
        'censustractandblock': 0,
        'regionidcounty': 0.0026132142789992686
    }
    
    '''attr_to_corr = {
        'heatingorsystemtypeid': 0.0065925300077240845,
        'propertylandusetypeid': 0.0048965672440217335,
        'propertyzoningdesc': 0.0032496335869956362,
        'regionidcity': 0.0059083849180240813,
        'regionidzip': 0.0051095117671334928,
        'censustractandblock': 0,
        'regionidcounty': 0.024501392675523249
    }'''
    
    corr_to_attr = {abs(attr_to_corr[column_name]): column_name for column_name in attr_to_corr}
    sorted_corr = sorted(map(abs, attr_to_corr.values()), reverse=not ascending)
    cat_list = [corr_to_attr[corr] for corr in sorted_corr]; print(cat_list)
    
    cat_df = test_df[cat_list]; print(cat_df.head())
    
    return train_df.drop(cat_list, axis=1), test_df.drop(cat_list, axis=1), cat_df

# VERIFIED
def scale(train_df, test_df, type=0):
    scaler_list = [
        ('Min Max', MinMaxScaler()),
        ('Normalizer', Normalizer()),
        ('Standardizer', StandardScaler())
    ]
    
    name, scaler = scaler_list[type]
    
    test_df.loc[:, :] = scaler.fit_transform(test_df)
    
    train_df = train_df[['logerror', 'transaction_month']].join(test_df)
    
    return train_df, test_df

# VERIFIED
def box_cox_transform(train_df, test_df):
    trans_list = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedfloor1squarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'landtaxdollarvaluecnt', 'lotsizesquarefeet', 'numberofstories', 'poolcnt', 'poolsizesum', 'roomcnt', 'structuretaxvaluedollarcnt', 'taxamount', 'taxdelinquencyflag', 'taxvaluedollarcnt', 'threequarterbathnbr', 'unitcnt']

    for column in test_df:
        if column in trans_list:
            test_df[column], my_lambda = boxcox(test_df[column] + 1)  # determines transformation on test set
            train_df[column] = boxcox(train_df[column] + 1, lmbda=my_lambda)  # applies same transformation on training set
    
    return train_df, test_df

def dummy_creator(train_df, test_df):
    dummy_list = ['heatingorsystemtypeid', 'propertylandusetypeid', 'regionidcounty']
    
    test_df = pd.get_dummies(test_df, columns=dummy_list)
    
    train_df = train_df[['logerror', 'transaction_month']].join(test_df)
    
    return train_df, test_df

# NOT VERIFIED - we're not using previously processed categorical columns to process later categorical columns
'''
def cat_process(train_df, test_df, cat_df, cat_column, attr_num=0, corr_threshold=0):
    # VERIFIED, except see comments
    log_corr = train_df.corr()['logerror'].drop('logerror').sort_values(ascending=False)#; print(log_corr)
    
    if attr_num != 0:
        target_columns = log_corr.index.values[: attr_num]
    else:
        target_columns = log_corr[log_corr >= corr_threshold].index
    
    column_to_weight = {}
    weight_sum = log_corr.loc[target_columns].sum()
    
    for i in range(len(target_columns)):
        column_to_weight[target_columns[i]] = log_corr.iloc[i] / weight_sum
    
    test_df[cat_column] = cat_df[cat_column].values     # NOT YET VERIFIED - THIS is changing test_df!
    
    group_by_column = test_df[list(target_columns) + [cat_column]].groupby(cat_column)
    
    group_df = group_by_column.mean()
    # Now, for each distinct value of cat_column, we have several means. We have a mean for each numerical column of interest.
    # So group_df is a DataFrame where each row index is a distinct cat_column value. The columns are the numerical columns. A value is a mean for the corresponding cat_column value, for that numerical column.
    
    group_df['weighted_score'] = sum(group_df[current_column] * column_to_weight[current_column] for current_column in target_columns)
    
    # NOT VERIFIED
    # Possible alternative to last line:
    # for each unique value v in cat_column:
    #     cat_df[ cat_df[cat_column] == v ] = weighted score for v
    
    # VERIFIED
    cat_df[cat_column] = cat_df[cat_column].map(lambda x: group_df.loc[x]['weighted_score'])
    
    return cat_df[cat_column]
'''

# VERIFIED
def cat_process_v2(train_df, test_df, cat_column_series, attr_num=0, corr_threshold=0):
    cat_column = cat_column_series.name
    
    # creating a Series of the correlations of the numerical attributes to 'logerror'
    log_corr = train_df.corr()['logerror'].drop('logerror').sort_values(ascending=False)
    
    # getting the columns that either have the highest correlations or have correlation higher than corr_threshold
    if attr_num != 0:
        target_columns = log_corr.index.values[: attr_num]
    else:
        target_columns = log_corr[log_corr >= corr_threshold].index
    
    # getting the weights for the columns in target_columns
    # by computing the ratio of its correlation to the sum of all correlations
    column_to_weight = {}
    weight_sum = log_corr.loc[target_columns].sum()
    
    for i in range(len(target_columns)):
        column_to_weight[target_columns[i]] = log_corr.iloc[i] / weight_sum
        
    # creating a new column in test_df, which now contains
    # one extra column of raw categorical data that is passed in
    test_df[cat_column] = cat_column_series.values
    
    # groupby object that contains all groups, each of which contains the rows
    # of the same distinct value in cat_column_series (the argument)
    group_by_column = test_df[list(target_columns) + [cat_column]].groupby(cat_column)
    
    group_df = group_by_column.mean()
    # Now, for each distinct value of cat_column, we have several means. We have a mean for each numerical column of interest.
    # So group_df is a DataFrame where each row index is a distinct cat_column value. The columns are the numerical columns. A value is a mean for the corresponding cat_column value, for that numerical column.
    
    # creating another column in group_df, which contains the weighted sum of the columns in target_columns for each group
    group_df['weighted_score'] = sum(group_df[current_column] * column_to_weight[current_column] for current_column in target_columns)
    
    # mapping each value to appropriate value from group_df['weighted_score']
    #test_df[cat_column] = cat_column_series.map(lambda x: group_df.loc[x]['weighted_score'])
    value_to_num = {value: group_df.loc[value]['weighted_score'] for value in group_df.index}#; print(value_to_num) # a dictionary that maps distinct values in cat_column_series to appropriate numerical values
    
    test_df[cat_column] = test_df[cat_column].map(value_to_num)
    train_df = train_df[['logerror', 'transaction_month']].join(test_df)
    
    return train_df, test_df

# VERIFIED
def custom_predict(train_df, test_df):
    msg = 'gradboost(2000,0.001)'
    
    list_dates = ['201610', '201611', '201612']

    predict_file = pd.read_csv('data/sample_submission.csv', index_col = 'ParcelId')
    
    alg = GradientBoostingRegressor(
        n_estimators = 2000,
        learning_rate = 0.001,
    )
    
    alg.fit(train_df.drop('logerror', axis = 1), train_df['logerror'])

    test_df.insert(0, 'transaction_month', -1)
    
    for date in list_dates:
        test_df['transaction_month'] = date[-2:]

        res = alg.predict(test_df)
        
        predict_file[date] = res
        predict_file['2017' + date[-2:]] = res
    
    print('Writing predictions to file...')
    predict_file.to_csv('data/custom_sub_%s.csv' % (msg))

def main():
    print('\nReading raw data...')
    train_df, test_df = read_raw_data()
    
    print('\nDropping outliers in logerror...')
    train_df = drop_outliers_logerror(train_df, 5.75); print(train_df.shape)
    
    print('\nConverting date to month...')
    train_df = transaction_date(train_df)
    
    print('\nFilling flag...')
    train_df, test_df = fill_flag(train_df, test_df)
    
    print('\nDropping columns with too few values...')
    train_df, test_df = drop_columns_little_info(0.5, train_df, test_df)
    
    '''print('\nLabel encoding...')
    train_df, test_df = str_to_num_encode(train_df, test_df)
    
    print('\nFilling missing values...')
    train_df, test_df = fill_missing(train_df, test_df); print(test_df.head())
    
    print('\nDropping columns with high correlation...')
    train_df, test_df = drop_columns_high_corr(0.9, train_df, test_df)
    
    print('\nWriting to files...')
    train_df.to_csv('data/train_new_cleaned.csv')
    test_df.to_csv('data/test_new_cleaned.csv')
    
    
    
    
    print('\nReading encoded data...')
    train_df, test_df = read_encoded_data(); print(train_df.shape, test_df.shape)
    
    print('\nSplitting data...')
    train_df, test_df, cat_df = cat_split(train_df, test_df); print(train_df.shape, test_df.shape)
    
    print('\nScaling numerical data...')
    train_df, test_df = scale(train_df, test_df); print(test_df.head())
    
    print('\nWriting to files...')
    train_df.to_csv('data/train_scaled_nocat.csv')
    test_df.to_csv('data/test_scaled_nocat.csv')
    cat_df.to_csv('data/cat_columns.csv')
    
    
    
    
    print('\nReading scaled data and cat columns...')
    train_df, test_df, cat_df = read_scaled_data_cat()
    
    start = timer()
    
    for column in cat_df.columns:
        print('Processing', column, '...')
        
        print(train_df.shape, test_df.shape)
        train_df, test_df = cat_process_v2(train_df, test_df, cat_df[column], attr_num=3)
        print(train_df.shape, test_df.shape)
    
    print('Took', timer() - start)
    train_df.to_csv('data/train_scaled_cat.csv')
    test_df.to_csv('data/test_scaled_cat.csv')
    
    
    
    
    print('\nReading cat processed data...')
    train_df, test_df = read_cat_processed(); print(train_df.shape, test_df.shape)
    
    print('\nCreating dummy values...')
    train_df, test_df = dummy_creator(train_df, test_df); print(train_df.shape, test_df.shape)
    
    print('\nPerforming Box-Cox trasformation...')
    train_df, test_df = box_cox_transform(train_df, test_df); print(train_df.shape, test_df.shape)
    
    print('\nRunning prediction...')
    custom_predict(train_df, test_df)
    
    '''
    
    print('\nFinished.')

main()
