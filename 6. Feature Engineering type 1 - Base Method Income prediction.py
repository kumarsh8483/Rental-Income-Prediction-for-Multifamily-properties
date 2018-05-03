import pandas
import numpy
import os
import csv

#Base method of income prediction - Unweighted mean of k-nearest neighbors

strat_train_set = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Final Datasets/strat_train_set.csv')
strat_test_set = pandas.read_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Final Datasets/strat_test_set.csv')
strat_train_set.drop(['Unnamed: 0'],axis=1,inplace=True)
strat_test_set.drop(['Unnamed: 0'],axis=1,inplace=True)
strat_train_set.columns

#Cross-validated Train errors
ind = strat_train_set[["('rent_final', '0')", "('rent_final', '1')", "('rent_final', '10')",
       "('rent_final', '11')", "('rent_final', '12')", "('rent_final', '13')",
       "('rent_final', '14')", "('rent_final', '15')", "('rent_final', '16')",
       "('rent_final', '17')", "('rent_final', '18')", "('rent_final', '19')",
       "('rent_final', '2')", "('rent_final', '20')", "('rent_final', '21')",
       "('rent_final', '22')", "('rent_final', '23')", "('rent_final', '24')",
       "('rent_final', '25')", "('rent_final', '26')", "('rent_final', '27')",
       "('rent_final', '28')", "('rent_final', '29')", "('rent_final', '3')",
       "('rent_final', '4')", "('rent_final', '5')", "('rent_final', '6')",
       "('rent_final', '7')", "('rent_final', '8')", "('rent_final', '9')"]].copy()
dep = strat_train_set['rent_final'].copy()
rent_predictions = ind.mean(axis=1)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#697.93

sklearn.metrics.r2_score(dep, rent_predictions)*100
# 66.65

def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs(y_pred - y_true) / y_true) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_pred - y_true) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#25.13

mean_percentage_error(dep, rent_predictions)
#12.70

#test error
ind_test = strat_test_set[["('rent_final', '0')", "('rent_final', '1')", "('rent_final', '10')",
       "('rent_final', '11')", "('rent_final', '12')", "('rent_final', '13')",
       "('rent_final', '14')", "('rent_final', '15')", "('rent_final', '16')",
       "('rent_final', '17')", "('rent_final', '18')", "('rent_final', '19')",
       "('rent_final', '2')", "('rent_final', '20')", "('rent_final', '21')",
       "('rent_final', '22')", "('rent_final', '23')", "('rent_final', '24')",
       "('rent_final', '25')", "('rent_final', '26')", "('rent_final', '27')",
       "('rent_final', '28')", "('rent_final', '29')", "('rent_final', '3')",
       "('rent_final', '4')", "('rent_final', '5')", "('rent_final', '6')",
       "('rent_final', '7')", "('rent_final', '8')", "('rent_final', '9')"]].copy()
dep_test = strat_test_set['rent_final'].copy()

rent_predictions_test = ind_test.mean(axis=1)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#756.37

sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#66.35

mean_absolute_percentage_error(dep_test, rent_predictions_test)
#26.29

mean_percentage_error(dep_test, rent_predictions_test)
#12.88

#Error buckets - by MAPE/MPE, rent range and city - train
a=strat_train_set.iloc[:,90].values
b=strat_train_set.iloc[:,91].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)
train_eb['MAPE'] = (numpy.abs(train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
train_eb['MPE'] = ((train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
bins = [0, 10, 20, 30, 50, 100, 100000]
train_eb['MAPE_bins'] = pandas.cut(train_eb['MAPE'], bins)
bins = [-100000, -100, -50, -30, -20, -10,  0, 10, 20, 30, 50, 100, 100000]
train_eb['MPE_bins'] = pandas.cut(train_eb['MPE'], bins)
pandas.pivot_table(train_eb,index=["MAPE_bins"],values=["MAPE"],aggfunc=[len])

#

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_final'], bins)

def mean_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.mean(numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.mean((df[y_pred] - df[y_true]) / df[y_true]) * 100

a = train_eb.groupby('Rent_bins').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['Rent_bins','MAPE']
b = train_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = train_eb.groupby('Rent_bins').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['Rent_bins','MPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#
   
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#

del(a,b,c,d)

#Error buckets - by MAPE/MPE, rent range and city - test

a=strat_test_set.iloc[:,90].values
b=strat_test_set.iloc[:,91].values
train_eb = pandas.DataFrame({'rent_final': a, 'city_final': b, 'rent_predictions': rent_predictions_test}, columns=['rent_final', 'city_final', 'rent_predictions'])
del(a,b)
train_eb['MAPE'] = (numpy.abs(train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
train_eb['MPE'] = ((train_eb['rent_final'] - train_eb['rent_predictions']) / train_eb['rent_final']) * 100
bins = [0, 10, 20, 30, 50, 100, 100000]
train_eb['MAPE_bins'] = pandas.cut(train_eb['MAPE'], bins)
bins = [-100000, -100, -50, -30, -20, -10,  0, 10, 20, 30, 50, 100, 100000]
train_eb['MPE_bins'] = pandas.cut(train_eb['MPE'], bins)
pandas.pivot_table(train_eb,index=["MAPE_bins"],values=["MAPE"],aggfunc=[len])

#

pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])

#

bins = [0, 500, 1000, 2000, 3000, 4000, 100000]
train_eb['Rent_bins'] = pandas.cut(train_eb['rent_final'], bins)

def mean_absolute_percentage_error_df(df,y_true, y_pred): 
    return numpy.mean(numpy.abs(df[y_pred] - df[y_true]) / df[y_true]) * 100

def mean_percentage_error_df(df, y_true, y_pred): 
    return numpy.mean((df[y_pred] - df[y_true]) / df[y_true]) * 100

a = train_eb.groupby('Rent_bins').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['Rent_bins','MAPE']
b = train_eb.groupby('Rent_bins').size().reset_index()
b.columns = ['Rent_bins','count']
c = train_eb.groupby('Rent_bins').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['Rent_bins','MPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)
  
#
  
del(a,b,c,d)

a = train_eb.groupby('city_final').apply(mean_absolute_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
a.columns = ['city_final','MAPE']
b = train_eb.groupby('city_final').size().reset_index()
b.columns = ['city_final','count']
c = train_eb.groupby('city_final').apply(mean_percentage_error_df,y_true='rent_final',y_pred='rent_predictions').reset_index()
c.columns = ['city_final','MPE']
d = pandas.merge(a, b[['count']], how='left', left_index=True, right_index=True)
pandas.merge(d, c[['MPE']], how='left', left_index=True, right_index=True)

#

del(a,b,c,d)



