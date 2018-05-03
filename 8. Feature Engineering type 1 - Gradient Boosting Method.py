from sklearn.ensemble import GradientBoostingRegressor

#GBM for rental predictions

param_grid = [
{'learning_rate': [0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3], 'n_estimators': [300,500,700,900], 'max_depth': [1,2,3,4,5,6]},
]
gbrt = GradientBoostingRegressor()
grid_search = GridSearchCV(gbrt, param_grid, cv=5)

ind = strat_train_set[[('weighted_rent_final', '0'),  ('weighted_rent_final', '1'),
       ('weighted_rent_final', '10'), ('weighted_rent_final', '11'),
       ('weighted_rent_final', '12'), ('weighted_rent_final', '13'),
       ('weighted_rent_final', '14'), ('weighted_rent_final', '15'),
       ('weighted_rent_final', '16'), ('weighted_rent_final', '17'),
       ('weighted_rent_final', '18'), ('weighted_rent_final', '19'),
        ('weighted_rent_final', '2'), ('weighted_rent_final', '20'),
       ('weighted_rent_final', '21'), ('weighted_rent_final', '22'),
       ('weighted_rent_final', '23'), ('weighted_rent_final', '24'),
       ('weighted_rent_final', '25'), ('weighted_rent_final', '26'),
       ('weighted_rent_final', '27'), ('weighted_rent_final', '28'),
       ('weighted_rent_final', '29'),  ('weighted_rent_final', '3'),
        ('weighted_rent_final', '4'),  ('weighted_rent_final', '5'),
        ('weighted_rent_final', '6'),  ('weighted_rent_final', '7'),
        ('weighted_rent_final', '8'),  ('weighted_rent_final', '9')]].copy()
dep = strat_train_set['rent_final'].copy()
grid_search.fit(ind, dep)
grid_search.best_params_
grid_search.best_estimator_
gbm1 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.01, loss='ls', max_depth=2, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=300, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)

gbm1.fit(ind, dep)

#Cross-validated Train errors
rent_predictions = gbm1.predict(ind)
rf1_mse = mean_squared_error(dep, rent_predictions)
rf1_rmse = numpy.sqrt(rf1_mse)
rf1_rmse
#656.77

sklearn.metrics.r2_score(dep, rent_predictions)*100
#70.47

def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs(y_pred - y_true) / y_true) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_pred - y_true) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#28.54

mean_percentage_error(dep, rent_predictions)
#14.41

#test error
ind_test = strat_test_set[[('weighted_rent_final', '0'),  ('weighted_rent_final', '1'),
       ('weighted_rent_final', '10'), ('weighted_rent_final', '11'),
       ('weighted_rent_final', '12'), ('weighted_rent_final', '13'),
       ('weighted_rent_final', '14'), ('weighted_rent_final', '15'),
       ('weighted_rent_final', '16'), ('weighted_rent_final', '17'),
       ('weighted_rent_final', '18'), ('weighted_rent_final', '19'),
        ('weighted_rent_final', '2'), ('weighted_rent_final', '20'),
       ('weighted_rent_final', '21'), ('weighted_rent_final', '22'),
       ('weighted_rent_final', '23'), ('weighted_rent_final', '24'),
       ('weighted_rent_final', '25'), ('weighted_rent_final', '26'),
       ('weighted_rent_final', '27'), ('weighted_rent_final', '28'),
       ('weighted_rent_final', '29'),  ('weighted_rent_final', '3'),
        ('weighted_rent_final', '4'),  ('weighted_rent_final', '5'),
        ('weighted_rent_final', '6'),  ('weighted_rent_final', '7'),
        ('weighted_rent_final', '8'),  ('weighted_rent_final', '9')]].copy()
dep_test = strat_test_set['rent_final'].copy()

rent_predictions_test = gbm1.predict(ind_test)
rf1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
rf1_rmse_test = numpy.sqrt(rf1_mse_test)
rf1_rmse_test
#806.99

sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#61.70

mean_absolute_percentage_error(dep_test, rent_predictions_test)
#32.36

mean_percentage_error(dep_test, rent_predictions_test)
#15.99

#Variable importance
feature_importances = gbm1.feature_importances_
feature_importances
attributes = ind.columns
pandas.DataFrame(sorted(zip(feature_importances, attributes), reverse=True)).to_csv('D:/Desktop/Capstone_Greystone/Data aggregation/Analysis/gbm_varimp.csv')

#[(0.13560037957095641, ('weighted_rent_final', '0')),
# (0.099829465896401029, ('weighted_rent_final', '3')),
# (0.096599855275976435, ('weighted_rent_final', '1')),
# (0.094966099729407241, ('weighted_rent_final', '8')),
# (0.069572425942160537, ('weighted_rent_final', '9')),
# (0.069119946790120693, ('weighted_rent_final', '4')),
# (0.062608507080962453, ('weighted_rent_final', '11')),
# (0.036022033899881387, ('weighted_rent_final', '7')),
# (0.03184952663121849, ('weighted_rent_final', '15')),
# (0.030891711080917355, ('weighted_rent_final', '22')),
# (0.029267143976798908, ('weighted_rent_final', '25')),
# (0.0276833827468678, ('weighted_rent_final', '2')),
# (0.027394353638251309, ('weighted_rent_final', '6')),
# (0.026892911915017508, ('weighted_rent_final', '12')),
# (0.024928287869156369, ('weighted_rent_final', '19')),
# (0.023999116132903186, ('weighted_rent_final', '20')),
# (0.021321395136133799, ('weighted_rent_final', '23')),
# (0.01540768288087494, ('weighted_rent_final', '14')),
# (0.014692076446431088, ('weighted_rent_final', '18')),
# (0.014387608747645032, ('weighted_rent_final', '16')),
# (0.012341552363276987, ('weighted_rent_final', '5')),
# (0.009711149811315295, ('weighted_rent_final', '10')),
# (0.009341386956745747, ('weighted_rent_final', '29')),
# (0.0083047096635603824, ('weighted_rent_final', '17')),
# (0.0072672898170195429, ('weighted_rent_final', '24')),
# (0.0, ('weighted_rent_final', '28')),
# (0.0, ('weighted_rent_final', '27')),
# (0.0, ('weighted_rent_final', '26')),
# (0.0, ('weighted_rent_final', '21')),
# (0.0, ('weighted_rent_final', '13'))]

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
pandas.pivot_table(train_eb,index=["MPE_bins"],values=["MPE"],aggfunc=[len])
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
joblib.dump(gbm1, "gbm1.pkl")
#gbm1 = joblib.load("gbm1.pkl")

