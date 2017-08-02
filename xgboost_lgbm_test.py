'''
                                                ,  ,
                                               / \/ \
                                              (/ //_ \_
     .-._                                      \||  .  \
      \  '-._                            _,:__.-"/---\_ \
 ______/___  '.    .--------------------'~-'--.)__( , )\ \
`'--.___  _\  /    |             Here        ,'    \)|\ `\|
     /_.-' _\ \ _:,_          Be Dragons           " ||   (
   .'__ _.' \'-/,`-~`                                |/
       '. ___.> /=,|  Abandon hope all ye who enter  |
        / .-'/_ )  '---------------------------------'
        )'  ( /(/
             \\ "
              '=='

'''
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc


##### READ IN RAW DATA, CHANGES TRANSACTIONDATE INTO A DATE
def read_raw_data():
    print( "\nReading data from disk ...")
    prop = pd.read_csv('data/properties_2016.csv')
    train = pd.read_csv("data/train_2016.csv", parse_dates = ['transactiondate'])
    return train, prop

##### CHANGES DATES INTO MONTHS
def date_to_month(train):
    train['transaction_month'] = train['transactiondate'].dt.month
    train = train.drop('transactiondate', axis=1)
    return train

##### READS IN SUBMISSION FILE
def read_submission_file():
    return pd.read_csv('data/sample_submission.csv')

##### PROCESS DATA FOR LIGHTGBM
def lgbm_data_processing(train, prop):
    print( "\nProcessing data for LightGBM ..." )
    #changes data into a smaller dtype, making the program faster
    for c, dtype in zip(prop.columns, prop.dtypes):	
        if dtype == np.float64:		
            prop[c] = prop[c].astype(np.float32)

    #this is now filling in the missing values with the median
    df_train = train.merge(prop, how='left', on='parcelid')
    df_train.fillna(df_train.median(), inplace=True)

    x_train = df_train.drop(['parcelid', 'logerror', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = df_train['logerror'].values
    print(x_train.shape, y_train.shape)

    train_columns = x_train.columns
    print('Train_columns: ',train_columns)

    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)

    #del df_train; gc.collect()

    #creats train and test samples for lgbm to use when doing analysis
    '''split = 90000
    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
    x_train = x_train.values.astype(np.float32, copy=False)
    x_valid = x_valid.values.astype(np.float32, copy=False)'''
   
    #makes a dataset that lgbm can read, pretty sure .json
    d_train = lgb.Dataset(x_train, label=y_train)
    #d_valid = lgb.Dataset(x_valid, label=y_valid)
    

    return d_train, train_columns, df_train

##### GET LIGHTGBM PARAMETERS
def get_lgbm_params():
    params = {}
    params['max_bin'] = 10
    params['learning_rate'] = 0.0021 # shrinkage_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'l1'          # or 'mae'
    params['sub_feature'] = 0.5      # feature_fraction 
    params['bagging_fraction'] = 0.85 # sub_row
    params['bagging_freq'] = 40
    params['num_leaves'] = 512        # num_leaf
    params['min_data'] = 500         # min_data_in_leaf
    params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

    return params

##### RUN LIGHTGBM PREDICTIONG FOR MULTIPLE MONTHS
def lgbm_predict_multi_month(d_train, train_columns, prop, list_dates, params, predict_file_lgbm):
    #watchlist = [d_valid]
    print("\nFitting LightGBM model ...")
    clf = lgb.train(params, d_train, 430)

    #del d_train, d_valid; gc.collect()
    #del x_train, x_valid; gc.collect()

    prop.insert(0, 'transaction_month', -1) 
    print("\nPrepare for LightGBM prediction ...")
    print("   Read sample file ...")
    sample = pd.read_csv('data/sample_submission.csv')
    print("   ...")
    sample['parcelid'] = sample['ParcelId']
    print("   Merge with property data ...")
    df_test = prop
    print("   ...")
    del sample, prop; gc.collect()
    print("   ...")
    x_test = df_test[train_columns]
    print("   ...")
    del df_test; gc.collect()
    print("   Preparing x_test...")
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)
    print("   ...")
    #x_test = x_test.values.astype(np.float32, copy=False)

    #generates the predictions for the three different months in the submission file
    for date in list_dates:
        print(date[-2:])
        print(type(date[-2:]))
        x_test['transaction_month'] = int(date[-2:])
        print("\nStart LightGBM prediction ...")
        #changed the name of x_test becuase changing to numpy array mutates the whole list, and we need the list for later on
        x_testorino = x_test.values.astype(np.float32, copy=False)
        # num_threads > 1 will predict very slow in kernal
        clf.reset_parameter({"num_threads":1})
        p_test = clf.predict(x_testorino)

        p_test = 0.97*p_test + 0.03*0.011
        predict_file_lgbm[date] = p_test
        predict_file_lgbm['2017' + date[-2:]] = p_test
    

        print( "\nAdjusted LightGBM predictions:" )
        print( pd.DataFrame(p_test).head() )

    del x_test; gc.collect()

    return predict_file_lgbm


##### RUN LIGHTGBM FOR ANALYZING DIFFERENT MONTHS
def lgbm_analysis_multi_month(d_train, train_columns, df_train, list_dates, params, predict_file_lgbm):
    
    print("\nFitting LightGBM model ...")
    clf = lgb.train(params, d_train, 430)

    print("\nPrepare for LightGBM prediction ...")
    print("   Read sample file ...")
    sample = pd.read_csv('data/sample_submission.csv')
    print("   ...")
    sample['parcelid'] = sample['ParcelId']
    print("   Merge with property data ...")
    #df_test = prop
    df_test = df_train[90000:]
    print(df_test.head())
    print("   ...")
    del sample; gc.collect()
    print("   ...")
    x_test = df_test[train_columns]
    print("   ...")
    del df_test; gc.collect()
    print("   Preparing x_test...")
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)
    print("   ...")

    #generates the predictions for the three different months in the submission file
    for date in list_dates:
        print(date[-2:])
        print(type(date[-2:]))
        x_test['transaction_month'] = int(date[-2:])
        print("\nStart LightGBM prediction ...")
        
        #changed the name of x_test becuase changing to numpy array mutates the whole list, and we need the list for later on
        x_testorino = x_test.values.astype(np.float32, copy=False)
        # num_threads > 1 will predict very slow in kernal
        clf.reset_parameter({"num_threads":1})
        p_test = clf.predict(x_testorino)

        p_test = 0.97*p_test + 0.03*0.011
        
        
        score = mean_absolute_error(df_train['logerror'][90000:], p_test)
        print('score: ', score)

    print( "\nAdjusted LightGBM predictions:" )
    print( pd.DataFrame(p_test).head() )

##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)
def re_read_prop():
    print( "\nRe-reading properties file ...")
    return  pd.read_csv('data/properties_2016.csv')

##### PROCESS DATA FOR XGBOOST
def process_data_and_params_xgboost(train, prop):
    print( "\nProcessing data for XGBoost ...")
    #fill na values and encode objects into labels if needed
    for c in prop.columns:
        prop[c]=prop[c].fillna(-1)
        if prop[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(prop[c].values))
            prop[c] = lbl.transform(list(prop[c].values))

    #makes a dummy training set for 
    prop = pd.get_dummies(prop,columns = ['heatingorsystemtypeid', 'propertylandusetypeid'])
        
    #makes a new training set and testing set
    train_df = train.merge(prop, how='left', on='parcelid')
    #inserts transaction_month into the prop dataframe
    prop.insert(0, 'transaction_month', -1)

    #fills in the flag values
    flag_dict = {'fireplaceflag': True, 'hashottuborspa': True, 'taxdelinquencyflag': 'Y'}
    for column in flag_dict:
        prop[column] = prop[column].apply(lambda x: 1 if x == flag_dict[column] else 0)
        train_df[column] = train_df[column].apply(lambda x: 1 if x == flag_dict[column] else 0)
    prop['poolcnt'] = prop['poolcnt'].fillna(0)
    train_df['poolcnt'] = train_df['poolcnt'].fillna(0)
    
    x_train = train_df.drop(['parcelid', 'logerror'], axis=1)
    x_test = prop.drop(['parcelid'], axis=1)

   

    # prints out the shape        
    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

    # drop out ouliers
    train_df=train_df[ train_df.logerror > -.4 ]
    train_df=train_df[ train_df.logerror < .418 ]
    x_train=train_df.drop(['parcelid', 'logerror'], axis=1)
    y_train = train_df["logerror"].values.astype(np.float32)
    y_mean = np.mean(y_train)

    #this code will be for analysis
    #x_train, x_test, y_train, y_test = train_test_split(train_df.drop('logerror', axis=1), train_df['logerror'], train_size=.8, random_state=1)

    print('After removing outliers:')     
    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': .3995,
        'base_score': y_mean,
        'silent': 1
    }

    return x_train, y_train, x_test, y_mean, xgb_params #, y_test

##### GET XGBOOST PARAMS
def get_xgb_params(y_mean):
    # xgboost params
    ''' xgb_params = {
        'eta': 0.032,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': .3995,
        'base_score': y_mean,
        'silent': 1
    }'''

    #another set of params to possibly try
    '''xgb_params = {
        'eta': 0.06,
        'max_depth': 5,
        'subsample': 0.77,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'silent': 1
    }'''

##### PRETTY SURE THIS GETS NUM_BOOST_ROUNDS
#def get_num_boost_rounds():
    # cross-validation
    #print( "Running XGBoost CV ..." )
    #cv_result = xgb.cv(xgb_params, 
    #                   dtrain, 
    #                   nfold=5,
    #                   num_boost_round=200,
    #                   early_stopping_rounds=50,
    #                   verbose_eval=10, 
    #                   show_stdv=False
    #                  )
    #num_boost_rounds = len(cv_result)
    
##### RUN XGBOOST
def xgb_predict_multi_month(x_train, y_train, x_test, train_columns, xgb_params, list_dates, predict_file_xgb):
    print("\nSetting up data for XGBoost ...")

    dtrain = xgb.DMatrix(x_train, y_train)

    num_boost_rounds = 242
    print("\nXGBoost tuned with CV in:")
    print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
    print("num_boost_rounds="+str(num_boost_rounds))
    print('x_test columns', x_test.columns)

    # train model
    print( "\nTraining XGBoost ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

    #predicts for the three months in the sample submission file
    for date in list_dates:
        x_test['transaction_month'] =int(date[-2:])
        dtest = xgb.DMatrix(x_test)
        print( "\nPredicting with XGBoost ...")
        xgb_pred = model.predict(dtest)
        print(xgb_pred.shape)
        print(predict_file_xgb.shape)
        predict_file_xgb[date] = xgb_pred
        predict_file_xgb['2017' + date[-2:]] = xgb_pred

    print( "\nXGBoost predictions:" )
    print( pd.DataFrame(xgb_pred).head() )

    return predict_file_xgb

##### ANALYZE DIFFERENT PREDICTION METHODS
def xgb_analyze_multi_month(x_train, y_train, x_test, y_test, train_columns, xgb_params, list_dates, predict_file_xgb):
    print("\nSetting up data for XGBoost ...")


    dtrain = xgb.DMatrix(x_train, y_train)

    num_boost_rounds = 210
    print("\nXGBoost tuned with CV in:")
    print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
    print("num_boost_rounds="+str(num_boost_rounds))
    print('x_test columns', x_test.columns)
    # train model
    print( "\nTraining XGBoost ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)


    #predicts for the three months in the sample submission file
    for date in list_dates:
        x_test['transaction_month'] =int(date[-2:])
        dtest = xgb.DMatrix(x_test)
        print( "\nPredicting with XGBoost ...")
        xgb_pred = model.predict(dtest)
        score = mean_absolute_error(y_test, xgb_pred)
        print('Score: ', score)


#def xgb_analyze_multi_month(x_train, y_train, x_test, y_test, train_columns, xgb_params, list_dates, predict_file_xgb):
    

    
##### COMBINE PREDICTIONS
def combine_weighted_pred(xgb_pred, lgbm_pred, lgb_weight = .2):
    print( "\nCombining XGBoost and LightGBM predicitons ...")


    return (xgb_pred*(1-lgb_weight) + lgbm_pred*lgb_weight)
            

##### WRITES RESULTS TO FILE
def write_results(pred):
    print( "\nCombined predictions:" )
    print( pd.DataFrame(pred).head() )
    #makes sure the dataframe is in the correct form before writing
    pred['ParcelId'] =  pred['ParcelId'].astype('int')
    pred = pred.set_index('ParcelId')
    pred = pred.drop('Unnamed: 0', axis = 1)
    pred = pred.round(decimals = 4)
    
    pred.to_csv('data/custom_sub.csv')

def main():
    list_dates = ['201610', '201611', '201612']
    
    '''lgb_sub = read_submission_file()
    xgb_sub = read_submission_file()
    train, prop = read_raw_data()
    
    train = date_to_month(train)
    
    d_train, train_columns, df_train = lgbm_data_processing(train, prop)
    lgbm_params = get_lgbm_params()

    #lgbm_analysis_multi_month(d_train, train_columns, df_train, list_dates, lgbm_params, lgb_sub)
    #lgb_sub = lgbm_predict_multi_month(d_train, train_columns, prop, list_dates, lgbm_params, lgb_sub)

    prop = re_read_prop()
    x_train, y_train, x_test, y_mean, xgb_params =  process_data_and_params_xgboost(train, prop)
    #xgb_params = get_xgb_params(y_mean)
    

    #xgb_analyze_multi_month(x_train, y_train, x_test, y_test, train_columns, xgb_params, list_dates, xgb_sub)
    #xgb_sub = xgb_predict_multi_month(x_train, y_train, x_test, train_columns, xgb_params, list_dates, xgb_sub)

    #Used for convenience when testing out different weights
    #lgb_sub.to_csv('data/lgb_sub.csv')
    #xgb_sub.to_csv('data/xgb_sub.csv')'''
    
    lgb_sub = pd.read_csv('data/lgb_sub.csv')
    xgb_sub = pd.read_csv('data/xgb_sub.csv')
    print(lgb_sub.shape)
    print(xgb_sub.shape)
    prediction = combine_weighted_pred(xgb_sub, lgb_sub, lgb_weight = .40)
    print(prediction.shape, prediction.head())
    write_results(prediction)
    
    print( "\nFinished ..." )
main()
