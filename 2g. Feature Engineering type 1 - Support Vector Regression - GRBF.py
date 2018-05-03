from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

#Support Vector Regression - Gaussian Radial Basis Function

sc = StandardScaler(with_mean=True, with_std=True)
ind_scaled = sc.fit_transform(ind)
dep_scaled = (dep - dep.mean())/dep.std()
ind_scaled=pandas.DataFrame(ind_scaled)
ind_scaled.columns = [('weighted_rent_final', '0'),  ('weighted_rent_final', '1'),
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
        ('weighted_rent_final', '8'),  ('weighted_rent_final', '9')]

#parameters = {'C':[1,5,10,25,50,75,100],'epsilon':[0.1,0.2,0.3,0.5]}
#parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.3,0.5]}
#svr_linear = LinearSVR()
#polynomial
parameters = [{'kernel': ['rbf'],'gamma': [1e-4, 1e-3, 0.001, 0.01, 0.1, 0.2, 0.5,1,3,5,10],
                'C':[1,5,10,25,50,75,100],'epsilon':[0.1,0.2,0.3,0.5]}]

svr_grbf = SVR()
svr_gridsearch = GridSearchCV(svr_grbf, parameters,cv=5)
svr_gridsearch.fit(ind_scaled,dep_scaled)
svr_gridsearch.best_params_
svr_gridsearch.best_estimator_
sv3=SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.3, gamma=0.01,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
sv3.fit(ind_scaled, dep_scaled)

#Cross-validated Train errors
temp = sv3.predict(ind_scaled)
rent_predictions = temp*dep.std() + dep.mean()
sv1_mse = mean_squared_error(dep, rent_predictions)
sv1_rmse = numpy.sqrt(sv1_mse)
sv1_rmse
#710.37

sklearn.metrics.r2_score(dep, rent_predictions)*100
# 65.45

def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    return numpy.mean((y_true - y_pred) / y_true) * 100

mean_absolute_percentage_error(dep, rent_predictions)
#25.37

mean_percentage_error(dep, rent_predictions)
#-9.27

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

ind_test_scaled = sc.transform(ind_test) 
temp1 = sv3.predict(ind_test_scaled)
rent_predictions_test = temp1*dep.std() + dep.mean()
sv1_mse_test = mean_squared_error(dep_test, rent_predictions_test)
sv1_rmse_test = numpy.sqrt(sv1_mse_test)
sv1_rmse_test
#800

sklearn.metrics.r2_score(dep_test, rent_predictions_test)*100
#62.36

mean_absolute_percentage_error(dep_test, rent_predictions_test)
#29.50

mean_percentage_error(dep_test, rent_predictions_test)
#-11.12

#Error buckets - by MAPE/MPE, rent range and city
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

#Store model
joblib.dump(sv3, "sv3.pkl")
#my_model_loaded = joblib.load("sv1.pkl")


