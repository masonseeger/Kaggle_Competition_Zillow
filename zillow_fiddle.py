import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

print('Loading data ...')

train = pd.read_csv('data/boxcox_train.csv')
prop = pd.read_csv('data/boxcox_test.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train
print('DF_TRAIN', df_train.head())
print('===============================================')

x_train = df_train.drop(['parcelid', 'logerror', 'transaction_month'], axis=1)
y_train = df_train['logerror'].values
print(x_train, y_train)
print('===============================================')

train_columns = x_train.columns
print('TRAIN_COLUMNS', train_columns)
print('===============================================')

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

      
d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l2'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

watchlist = [d_valid]
clf = lgb.train(params, d_train, 500, watchlist)

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()

print("Prepare for the prediction ...")
sample = pd.read_csv('data/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = pd.concat([sample, prop], axis = 1)
print('DF_TEST', df_test.head())
print('===============================================')

del sample, prop; gc.collect()

x_test = df_test[train_columns]
print('X_TEST', x_test.head())
print('===============================================')
del df_test; gc.collect()

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)
p_test = 0.97*p_test + 0.03*0.011
print('P_TEST', p_test)
print('===============================================')

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_starter.csv', index=False)




