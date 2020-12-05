from XGParis_no import XGParis
import pandas as pd
import decimal 


###################################################USE EXAMPLES

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)

#dataset = pd.read_excel('Oxide_glass_1_5_02142020.xlsx')
#features = pd.read_csv('clr_data.csv')
#features = pd.read_csv('alr_data.csv')
#features = pd.read_csv('ilr_data.csv')
#features  = dataset[dataset.columns[7:-3]]
#targets  = dataset[dataset.columns[-1]]

dataset = pd.read_csv('knn_2.csv')
x_train  = dataset[dataset.columns[:-1]]
y_train  = dataset[dataset.columns[-1]]

data =  pd.read_csv('test_set3.csv')
x_test = data[data.columns[:-1]]
y_test = data[data.columns[-1]]

print(x_train)
print(y_train)

params = {
    'objective' : ['reg:gamma','reg:logistic', 'reg:tweedie','binary:logistic','count:poisson','reg:pseudohubererror'], 
    'n_estimators' : list(range(0, 500,5)), 
    'max_depth' : list(range(2,50)),
    'tree_method' :  ['auto', 'exact','approx', 'hist'],
    'booster' :  ['gbtree', 'gblinear', 'dart'],
    'reg_alpha' : [.05,.1,.15,.20,.25,.30],
    'reg_lambda' : [0,.2,.4,.6,.8,1],
    'learning_rate' : [.05,.08,.1,.15,.20],
    'gamma' : [ 0.0, 0.1, 0.2],
    'min_child_weight' : list(range(2,50)),
    'colsample_bytree': list(float_range(decimal.Decimal(0), decimal.Decimal(1), '0.01')),
    'colsample_bylevel':list(float_range(decimal.Decimal(0), decimal.Decimal(1), '0.01')),
    'colsample_bynode': list(float_range(decimal.Decimal(0), decimal.Decimal(1), '0.01')),
    'importance_type' : ['gain', 'weight', 'cover', 'total_gain','total_cover']}




xgbrf = XGParis(5, 20, params, x_train, y_train, x_test, y_test, forrest = False)
xgbrf.optuna_tuner()
#xgbrf.sklearn_tuner()
print(xgbrf.best_param())
print(xgbrf.best_score())
print(xgbrf.average_score(xgbrf.best_param()))
print(xgbrf.all_trials())
xgbrf.get_parameter_graph('n_estimators')
xgbrf.r2_graph()
xgbrf.make_parameter_graph('n_estimators',range(20,500,10))
print(xgbrf.best_random_seed)


