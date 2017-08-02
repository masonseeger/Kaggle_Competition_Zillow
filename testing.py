import pandas as pd
import numpy as np

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def generate_sub(step):
    lgb_sub = pd.read_csv('data/testing/lgb_testing.csv', index_col='ParcelId')#; print(lgb_sub.head())
    xgb_sub = pd.read_csv('data/testing/xgb_testing.csv', index_col='ParcelId')#; print(xgb_sub.head())
    grad_sub = pd.read_csv('data/testing/grad_testing.csv', index_col='ParcelId')#; print(grad_sub.head())
    
    weight_list = []
    
    for lgb_weight in frange(0, 1.01, step):
        if lgb_weight <= 1.01:
            for xgb_weight in frange(0, 1.01 - lgb_weight, step):
                weight_list.append((round(lgb_weight, 2), round(xgb_weight, 2), round(1 - lgb_weight - xgb_weight, 2)))
    
    #print(len(weight_list))
    #for item in weight_list: print(item)
    
    for lgb_weight, xgb_weight, grad_weight in weight_list:
        print('Processing weight (%s, %s, %s)...' % (str(lgb_weight), str(xgb_weight), str(grad_weight)))
        
        current_sub = lgb_weight * lgb_sub + xgb_weight * xgb_sub + grad_weight * grad_sub#; print(current_sub)
        current_sub.to_csv('sub/(%s, %s, %s).csv' % (str(lgb_weight), str(xgb_weight), str(grad_weight)))
    
def main():
    generate_sub(0.1)

main()