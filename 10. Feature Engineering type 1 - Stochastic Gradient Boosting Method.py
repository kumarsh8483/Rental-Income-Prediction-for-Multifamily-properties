from sklearn.ensemble import GradientBoostingRegressor

#Stochastic GBM for rental predictions

param_grid = [
{'learning_rate': [0.005,0.01,0.05,0.1,0.15], 'n_estimators': [300,500,700,900], 'max_depth': [2,4,6], 'subsample': [0.25,0.5,0.75,1], 'max_features':["auto","sqrt","log2"] },
]

#param_grid = [
#{'learning_rate': [0.01,0.03,0.05,0.07,0.09], 'n_estimators': [300,500,700,900], 'max_depth': [2,3,4], 'subsample': [0.33,0.67,1], 'max_features':["auto","sqrt","log2"] },
#]

gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)

ind = strat_train_set[["('weighted_rent_final', '0')",  "('weighted_rent_final', '1')",
       "('weighted_rent_final', '10')", "('weighted_rent_final', '11')",
       "('weighted_rent_final', '12')", "('weighted_rent_final', '13')",
       "('weighted_rent_final', '14')", "('weighted_rent_final', '15')",
       "('weighted_rent_final', '16')", "('weighted_rent_final', '17')",
       "('weighted_rent_final', '18')", "('weighted_rent_final', '19')",
        "('weighted_rent_final', '2')", "('weighted_rent_final', '20')",
       "('weighted_rent_final', '21')", "('weighted_rent_final', '22')",
       "('weighted_rent_final', '23')", "('weighted_rent_final', '24')",
       "('weighted_rent_final', '25')", "('weighted_rent_final', '26')",
      "('weighted_rent_final', '27')", "('weighted_rent_final', '28')",
       "('weighted_rent_final', '29')",  "('weighted_rent_final', '3')",
        "('weighted_rent_final', '4')",  "('weighted_rent_final', '5')",
        "('weighted_rent_final', '6')",  "('weighted_rent_final', '7')",
        "('weighted_rent_final', '8')",  "('weighted_rent_final', '9')"]].copy()
dep = strat_train_set['rent_final'].copy()
grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_
gbm2 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='ls', max_depth=6,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=None, subsample=1, verbose=0,
             warm_start=False)
gbm2.fit(ind, dep)

#gbm3 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#             learning_rate=0.01, loss='ls', max_depth=4,
#             max_features='log2', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=300,
#             presort='auto', random_state=None, subsample=0.67, verbose=0,
#             warm_start=False)
#gbm3.fit(ind, dep)

#Cross-validated Train errors
rent_predictions = gbm2.predict(ind)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#61.31

sklearn.metrics.r2_score(dep, rent_predictions)*100
#99.74

def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs(y_pred - y_true) / y_true) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_pred - y_true) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#3.694

mean_percentage_error(dep, rent_predictions)
#1.06

#test error
ind_test = strat_test_set[["('weighted_rent_final', '0')",  "('weighted_rent_final', '1')",
       "('weighted_rent_final', '10')", "('weighted_rent_final', '11')",
       "('weighted_rent_final', '12')", "('weighted_rent_final', '13')",
       "('weighted_rent_final', '14')", "('weighted_rent_final', '15')",
       "('weighted_rent_final', '16')", "('weighted_rent_final', '17')",
       "('weighted_rent_final', '18')", "('weighted_rent_final', '19')",
        "('weighted_rent_final', '2')", "('weighted_rent_final', '20')",
       "('weighted_rent_final', '21')", "('weighted_rent_final', '22')",
       "('weighted_rent_final', '23')", "('weighted_rent_final', '24')",
       "('weighted_rent_final', '25')", "('weighted_rent_final', '26')",
      "('weighted_rent_final', '27')", "('weighted_rent_final', '28')",
       "('weighted_rent_final', '29')",  "('weighted_rent_final', '3')",
        "('weighted_rent_final', '4')",  "('weighted_rent_final', '5')",
        "('weighted_rent_final', '6')",  "('weighted_rent_final', '7')",
        "('weighted_rent_final', '8')",  "('weighted_rent_final', '9')"]].copy()

dep_test = strat_test_set['rent_final'].copy()

rent_predictions_test = gbm2.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#761.39

sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#65.90

mean_absolute_percentage_error(dep_test, rent_predictions_test)
#29.76

mean_percentage_error(dep_test, rent_predictions_test)
#13.46

#Variable importance
feature_importances = gbm2.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/stoachastic_gbm_varimp.csv')

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

#Store model
joblib.dump(gbm2, "gbm2.pkl")
#gbm2 = joblib.load("gbm2.pkl")

#joblib.dump(gbm2, "gbm3.pkl")
#gbm3 = joblib.load("gbm3.pkl")
