import numpy as np 
import pandas as pd 
import xgboost as xg
import xlrd
import decimal 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import warnings


#This is  a list of all objectives to feed into the objective parameter of the xgboost regression 
objectives = ['survival:aft',
              'binary:hinge',
              'multi:softmax',
              'multi:softprob',
              'rank:pairwise',
              'rank:ndcg',
              'rank:map',
              'reg:gamma',
              'reg:tweedie',
              'reg:squarederror',
              'reg:squaredlogerror',
              'reg:logistic',
              'reg:pseudohubererror',
              'binary:logistic',
              'binary:logitraw',
              'reg:linear',
              'count:poisson',
              'survival:cox']

class XGParis:

    def __init__(self, epoch, n_trials, params, features, targets, split = .30, forrest = False):
        self.epoch = epoch
        self.n_trials = n_trials
        self.params = params
        self.features = features
        self.targets = targets
        self.forrest = forrest

        #Train Test Split
        x_train, x_test, y_train, y_test = train_test_split (self.features, self.targets, test_size=split, random_state = 24)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.split = split

        
    def sklearn_tuner(self,lower_bound = .90, upper_bound = 1.10):
        warnings.filterwarnings('ignore')
        epoch = self.epoch
        n_trials = self.n_trials
        params = self.params
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        forrest = self.forrest

        '''
        Method Summary. This method uses sklearns random search grid recurisvly to tune a XG Boost Regression model's hyper
        parameters. Sklearns random search tends to crash if epoch and n_trials are too high. 
        epoch: This parameter moderates how many times the tunnner will cycle through a random search. Keep in mind that every epoch
                the parameters used are narrowed down by the random search run during the last epoch
        n_trials: this parameter dictates how many trials will be run each time a random search is called. Cross validataion is set to 5
                    so if n_trial was set to 20 each random search will actualy run through 100 trials 
        params: this parameter is the hyperparamter grid you want to initialy feed into the tunner.
                An example parameter grid for the xgboost regression looks like this.
                params = {
                        'objective' : ['reg:gamma'], 
                        'n_estimators' : range(50, 130,10), #500
                        'max_depth' : range(2,25),
                        'tree_method' :  ['auto', 'exact','approx', 'hist'],
                        'booster' :  ['gbtree', 'gblinear', 'dart'],
                        'sampling_method' : ['gradient_based'],
                        'reg_alpha' : [.05,.1,.15,.20,.25,.30],
                        'reg_lambda' : [0,.2,.4,.6,.8,1],
                        'learning_rate' : [.05,.08,.1,.15,.20],
                        'gamma' : [ 0.0, 0.1, 0.2],
                        'min_child_weight' : [ 1, 3, 5, 7],
                        'colsample_bytree' : [0,.2,0.3, 0.4,.6,.8,1],
                        #'colsample_bylevel':[0,.2,0.3, 0.4,.6,.8,1],
                        #'colsample_bynode': [0,.2,0.3, 0.4,.6,.8,1],
                        'importance_type' : ['gain', 'weight', 'cover', 'total_gain','total_cover']}

        features : model featres (x)
        targets : model target (y)

        forrest: Boolean. If you would like the model you are training to be a XGBoost Regression Tree then keep forrest = False (default)
        If you want to train and tune a forrest change forrest to True. Default False

        upper_bound: Every epoch the tunner takes the last epoch's best params and creates a new param grid with the chosen param*lower_bound as the lowest
        number in the new grid, and chosen param*upper_bound as the highest number in the new param_gird. Default is 1.10

        lower_bound: Every epoch the tunner takes the last epoch's best params and creates a new param grid with the chosen param*lower_bound as the lowest
        number in the new grid, and chosen param*upper_bound as the highest number in the new param_gird. Default is .90   

        split: train/test split. .20 is the same as an 80/20 split. Default 
        
        Use this syntax to access the pandas data frame after tunning the model with either sklearn or optuna
        XGParis(**kwargs).best_params
        '''

        def float_range(start, stop, step):
          while start < stop:
            yield float(start)
            start += decimal.Decimal(step)
            
        top_params = []
        if forrest == False:
            xgb_r = xg.XGBRegressor(seed = 123)
        else:
            xgb_r = xg.XGBRFRegressor(seed = 123) 
            
        xgb_random = RandomizedSearchCV(estimator =xgb_r, param_distributions = params, n_iter = n_trials, cv = 5, verbose=2, random_state=42, n_jobs = -1)
        xgb_random.fit(x_train, y_train)

        param_random = (xgb_random.best_params_)
        top_params.append([(xgb_random.best_score_),(xgb_random.best_params_)])
        #print(param_random )
        counter = 2
        def repeater(epoch, counter, param_dic):
            new_params = {}
            for item in (param_dic):
                #print(item)
                if type(param_dic[str(item)]) == int:            
                    new_params[str(item)] = list(range(round(param_dic[str(item)]*lower_bound), round(param_dic[str(item)]*upper_bound)))
                    if new_params[str(item)] == []:
                        new_params[str(item)] =  [param_dic[str(item)]]                
                elif type(param_dic[str(item)]) == float:
                    if item == 'colsample_bytree' or item == 'colsample_bylevel' or item == 'colsample_bynode' :
                        if param_dic[str(item)]*1.10 >= 1:
                            new_params[str(item)] = (list(float_range(decimal.Decimal(param_dic[str(item)]*(upper_bound-lower_bound)), 1, '0.01')))
                        else:
                            new_params[str(item)] = (list(float_range(decimal.Decimal(param_dic[str(item)]*lower_bound), 
                                                                      decimal.Decimal(param_dic[str(item)]*upper_bound), '0.01')))
                    elif param_dic[str(item)] == 0.0:                    
                        new_params[str(item)] = (list(float_range(0, decimal.Decimal(param_dic[str(item)]*(upper_bound-lower_bound)), '0.01')))
                        if new_params[str(item)] == []:
                            new_params[str(item)] =  [param_dic[str(item)]] 
                            
                    elif item == 'learning_rate':
                        new_params[str(item)] = (list(float_range(decimal.Decimal(param_dic[str(item)]*lower_bound), 
                                                                  decimal.Decimal(param_dic[str(item)]*upper_bound), '0.001')))
                        
                    else:
                        new_params[str(item)] = (list(float_range(decimal.Decimal(param_dic[str(item)]*lower_bound), 
                                                                  decimal.Decimal(param_dic[str(item)]*upper_bound), '0.01')))
                elif type(param_dic[str(item)]) == str:
                    new_params[str(item)] = [(param_dic[str(item)])]

                else:
                    #print('error, skipped ' + str(item))
                    continue
            #print(new_params)       

            if forrest == False:
                xgb_r = xg.XGBRegressor(seed = 123)
            else:
                xgb_r = xg.XGBRFRegressor(seed = 123)
                
            xgb_random = RandomizedSearchCV(estimator =xgb_r, param_distributions = new_params, n_iter = n_trials, cv = 5, verbose=2, random_state=42, n_jobs = -1)
            xgb_random.fit(x_train, y_train)

            
            if counter >= epoch:
                counter += 1
                #print('epoch - ' + str(counter) + ' Done')
                return xgb_random.best_params_
            else:
                counter += 1
                #print('epoch - ' + str(counter))
                top_params.append([(xgb_random.best_score_),(xgb_random.best_params_)])
                #print(xgb_random.best_score_)
                return repeater(epoch,counter,(xgb_random.best_params_))

       

        final_param = repeater(epoch,counter,param_random)
        scores = []
        params_list = []
        for item in top_params:
            scores.append(item[0])
            params_list.append(item[1])
        top = pd.DataFrame(columns= ['scores','params'])
        top['scores'] = scores
        top['params'] = params_list


        if forrest == False:
            xgb_r = xg.XGBRegressor(seed = 123)
        else:
            xgb_r = xg.XGBRFRegressor(seed = 123)

        param_dic = top['params'][list(top['scores']).index(max(list(top['scores'])))]
        new_params = {}
        for item in param_dic:
            if type(param_dic[str(item)]) == str:
                new_params[str(item)] = params[str(item)]
            elif type(param_dic[str(item)]) == int:            
                new_params[str(item)] = [(param_dic[str(item)])]               
            elif type(param_dic[str(item)]) == float:
                new_params[str(item)] = [(param_dic[str(item)])]
            
        xgb_grid = GridSearchCV(estimator =xgb_r, param_distributions = new_params, n_iter = n_trials, cv = 5, verbose=2, random_state=42, n_jobs = -1)
        xgb_grid.fit(x_train, y_train)

        top_params.append([(xgb_grid.best_score_),(xgb_grid.best_params_)])

        self.best_grid = top['params'][list(top['scores']).index(max(list(top['scores'])))]
        self.best_params = top
        self.optuna = False


    def optuna_tuner(self, lower_bound = .80 , upper_bound = 1.20 ):
        warnings.filterwarnings('ignore')
        epoch = self.epoch
        n_trials = self.n_trials
        params = self.params
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        forrest = self.forrest

        '''
        Method Summary. This method uses Optuna's parameter tunning recurisvly to tune a XG Boost Regression model's hyper
        parameters. Optuna is more efficient then sklearns random grid as it prunes trees that are not promising to use
        more of its processing power on promising param combinations. This means that Optuna get better faster then sklearn's random
        or grid search
        epoch: This parameter moderates how many times the tunnner will cycle through a random search. Keep in mind that every epoch
                the parameters used are narrowed down by the random search run during the last epoch
        n_trials: this parameter dictates how many trials will be run each time a random search is called. Cross validataion is set to 5
                    so if n_trial was set to 20 each random search will actualy run through 100 trials 
        params: this parameter is the hyperparamter grid you want to initialy feed into the tunner.
                An example parameter grid for the xgboost regression looks like this.
                params = {
                    'objective' : ['reg:gamma'], 
                    'n_estimators' : range(50, 130,10), #500
                    'max_depth' : range(2,25),
                    'tree_method' :  ['auto', 'exact','approx', 'hist'],
                    'booster' :  ['gbtree', 'gblinear', 'dart'],
                    'sampling_method' : ['gradient_based'],
                    'reg_alpha' : [.05,.1,.15,.20,.25,.30],
                    'reg_lambda' : [0,.2,.4,.6,.8,1],
                    'learning_rate' : [.05,.08,.1,.15,.20],
                    'gamma' : [ 0.0, 0.1, 0.2],
                    'min_child_weight' : [ 1, 3, 5, 7],
                    'colsample_bytree': list(float_range(decimal.Decimal(0), decimal.Decimal(1), '0.01')),
                    'colsample_bylevel':list(float_range(decimal.Decimal(0), decimal.Decimal(1), '0.01')),
                    'colsample_bynode': list(float_range(decimal.Decimal(0), decimal.Decimal(1), '0.01')),
                    'importance_type' : ['gain', 'weight', 'cover', 'total_gain','total_cover']}

        x_train: The train features
        x_test: the test features
        y_train: the train targets
        y_test: the test targets



        Use this syntax to access the pandas data frame after tunning the model with either sklearn or optuna
                XGParis(**kwargs).best_params
        '''

        top_params = []
        all_params = pd.DataFrame(columns = ('scores', 'params'))
        def objective (trial: Trial, param_dic = params):
            new_params = {}
            for item in (param_dic):
                new_params[str(item)] = trial.suggest_categorical(str(item),list(param_dic[str(item)]))

            if forrest == False:
                xgb_r = xg.XGBRegressor(**new_params)
            else:
                xgb_r = xg.XGBRFRegressor(**new_params)
                
            xgb_r.fit(x_train,y_train)
            score = model_selection.cross_val_score(xgb_r, x_train, y_train, n_jobs=-1, cv=5)
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize',sampler=TPESampler())
        study.optimize(lambda trial : objective(trial),n_trials= n_trials)

        for item in study.trials:
            all_params.loc[len(all_params)] = (item.value, item.params)

            
        param_dic_random = study.best_trial.params
        top_params.append([study.best_trial.value,study.best_trial.params])
        
        counter = 2
        def repeater(epoch,counter, param_dic):
            def objective_2 (trial: Trial, param_dic = param_dic):
                new_params = {}
                for item in (param_dic):
                    if type(param_dic[str(item)]) == int:
                        new_params[str(item)] = trial.suggest_int(str(item), param_dic[str(item)]*lower_bound, 
                                                                             param_dic[str(item)]*upper_bound)
                    elif type(param_dic[str(item)]) == float:
                        if item == 'colsample_bytree' or item == 'colsample_bylevel' or item == 'colsample_bynode' :
                            if param_dic[str(item)]*1.25 >= 1:
                                new_params[str(item)] = trial.suggest_float(str(item), param_dic[str(item)]*lower_bound, 1)
                            else:
                                new_params[str(item)] = trial.suggest_float(str(item), param_dic[str(item)]*lower_bound, 
                                                                                     param_dic[str(item)]*upper_bound)
                        else:
                            new_params[str(item)] = trial.suggest_float(str(item), param_dic[str(item)]*lower_bound, 
                                                                                     param_dic[str(item)]*upper_bound)
                    elif type(param_dic[str(item)]) == str:
                        new_params[str(item)] = trial.suggest_categorical(str(item), [(param_dic[str(item)])])

                    else:
                        print('error, skipped ' + str(item))
                        continue 
                #print(new_params)  
                if forrest == False:
                    xgb_r = xg.XGBRegressor(**new_params)
                else:
                    xgb_r = xg.XGBRFRegressor(**new_params)
                    
                xgb_r.fit(x_train,y_train)
                score = model_selection.cross_val_score(xgb_r, x_train, y_train, n_jobs=-1, cv=5)
                accuracy = score.mean()
                return accuracy

            study = optuna.create_study(direction='maximize',sampler=TPESampler())
            study.optimize(lambda trial : objective_2(trial),n_trials= n_trials)

            for item in study.trials:
                all_params.loc[len(all_params)] = (item.value, item.params)
            
            
            #print(study)
            if counter >= epoch:
                counter += 1
                #print('epoch - ' + str(counter) + ' Done')
                return study.best_trial.params
            else:
                counter += 1
                #print('epoch - ' + str(counter))
                top_params.append([study.best_trial.value,study.best_trial.params])
                return repeater(epoch,counter,study.best_trial.params)

        final_param = repeater(epoch,counter,param_dic_random)
        scores = []
        params_list = []
        for item in top_params:
            scores.append(item[0])
            params_list.append(item[1])
        top = pd.DataFrame(columns= ['scores','params'])
        top['scores'] = scores
        top['params'] = params_list

        if forrest == False:
            xgb_r = xg.XGBRegressor(seed = 123)
        else:
            xgb_r = xg.XGBRFRegressor(seed = 123)

        param_dic = top['params'][list(top['scores']).index(max(list(top['scores'])))]
        new_params = {}
        for item in param_dic:
            if type(param_dic[str(item)]) == str:
                new_params[str(item)] = params[str(item)]
            elif type(param_dic[str(item)]) == int:            
                new_params[str(item)] = [(param_dic[str(item)])]               
            elif type(param_dic[str(item)]) == float:
                new_params[str(item)] = [(param_dic[str(item)])]

        print(new_params)
            
        xgb_grid = GridSearchCV(estimator =xgb_r, param_grid = new_params, cv = 5, verbose=2, n_jobs = -1)
        xgb_grid.fit(x_train, y_train)

        top_params.append([(xgb_grid.best_score_),(xgb_grid.best_params_)])

        self.best_grid = top['params'][list(top['scores']).index(max(list(top['scores'])))]
        self.best_params = top
        self.all_params = all_params 
        self.optuna = True



    def all_trials(self):
        ''''
        return a pandas dataframe of all trials
        '''
        
        all_params= self.all_params
        return all_params


    def best_param(self):
        '''
        returns the top param_grid based on R^2
        '''
        if self.optuna == True:
            top = self.all_params
            return top['params'][list(top['scores']).index(max(list(top['scores'])))]
        else:
            top = self.best_params
            return top['params'][list(top['scores']).index(max(list(top['scores'])))]
            
            
    def best_score(self):
        '''
        return the top score based on R^2
        '''
        if self.optuna == True:
            top = self.all_params
            return max(list(top['scores']))
        else:
            top = self.best_params
            return max(list(top['scores']))

    def predict(self,features):
        '''
        features: Input features, make sure the input has the same dimensionality as the training data
        '''
        
        best_param = self.best_grid
        if forrest == False:
            xgb_r = xg.XGBRegressor(**best_param)
        else:
            xgb_r = xg.XGBRFRegressor(**best_param)

        return xgb_r.predict(features)


    def r2_graph(self):
        '''
        plots the predicted value aginst the actual value for both x_train, and x_test
        '''
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        forrest = self.forrest
        params = self.params
        best_param = self.best_grid
        features = self.features 
        targets = self.targets
        split = self.split

        x_train, x_test, y_train, y_test = train_test_split (features, targets, test_size=split,
                                                             random_state = 24)
        if forrest == False:
            xgb_r = xg.XGBRegressor(**best_param)
        else:
            xgb_r = xg.XGBRFRegressor(**best_param)

        # Fitting the model 
        xgb_r.fit(x_train, y_train) 

        # Predict the model 
        pred_test = xgb_r.predict(x_test)
        pred_train =  xgb_r.predict(x_train)

        r2_test  = metrics.r2_score(y_test.values, pred_test)
        r2_train  = metrics.r2_score(y_train.values, pred_train)

        #print(num)

                
        plt.scatter(pred_test,y_test, marker = "D", color = 'blue', label = 'Test Data')
        plt.scatter(pred_train,y_train, marker = "D", color = 'red', label = 'Train Data')
        
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.legend()
        plt.show()
        
        

    def get_parameter_graph(self, parameter):
        '''
        This method only works for a model trained with Optuna
        parameter : this param is the paramter you would like a graph of
                    the graph is created using matplotlib and with the
                    parameter chosen on the x axis and the r^2 on the y axis

        ** Keep inmind that the graphs do not keep the rest of the variables constant, so the graphs
        are not as informative as you may think 
        '''

        if self.optuna == True:
            all_params = self.all_params
            
            points = []
            for item in all_params.values:
                points.append([item[1][parameter],item[0]])
                
            x = []
            y = []
            for item in (sorted(points)):
                x.append(item[0])
                y.append(item[1])
            plt.plot(x,y)
            plt.xlabel(parameter)
            plt.ylabel('R^2')
            plt.show()
        else:
            raise ValueError('Graph Error Only Works for Models Trained With Optuna Method')

    def make_parameter_graph(self, parameter, test_range):
        '''
        This method uses optuna to make a parameter graph, which keeps all other paramters constant as the 
        paramter chosen varies. The param grid used is the best_param grid from the tunning process.
        
        parameter : this param is the paramter you would like a graph of
                    the graph is created using matplotlib and with the
                    parameter chosen on the x axis and the r^2 on the y axis
        
        test_range: This param is a list of all of the values you would like tested for your graph
        ex. list(range(0,10)) or [1,2,3,4,5,6,7,8,9] or [3,6,9]
        
        
        '''
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        forrest = self.forrest
        params = self.params
        best_param = self.best_grid
        features = self.features 
        targets = self.targets
        split = self.split

        nums = []
        x = []
        for num in test_range:
            best_param[parameter] = num
            
            try:
                x_train, x_test, y_train, y_test = train_test_split (features, targets, test_size=split,
                                                                     random_state = 24)
                if forrest == False:
                    xgb_r = xg.XGBRegressor(**best_param)
                else:
                    xgb_r = xg.XGBRFRegressor(**best_param)

                # Fitting the model 
                xgb_r.fit(x_train, y_train) 

                # Predict the model 
                pred = xgb_r.predict(x_test) 

                r2  = metrics.r2_score(y_test.values, pred)
                x.append(num)
                nums.append(r2)
                #print(num)

            except Exception as e:
                print(e)
                continue

            

                
        plt.plot(x,nums)
        plt.xlabel(parameter)
        plt.ylabel('R^2')
        
        plt.show()


    def average_score(self, params):
        forrest = self.forrest
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        features = self.features 
        targets = self.targets
        split = self.split

        '''
        this method tries the first 50 random seeds for the train test split,
        then returns an average of the R^2s across the random seeds
        params : this parameter will usualy be the value of the best param
        method, but you can enter any paramter you want, as long as they
        are included in the xgboost.Regression documentation 
        '''
        
        nums = []
        for num in range(0,50):
            try:
                x_train, x_test, y_train, y_test = train_test_split (features, targets, test_size=split, random_state = num)
                if forrest == False:
                    xgb_r = xg.XGBRegressor(**params)
                else:
                    xgb_r = xg.XGBRFRegressor(**params)

                # Fitting the model 
                xgb_r.fit(x_train, y_train) 

                # Predict the model 
                pred = xgb_r.predict(x_test) 

                r2  = metrics.r2_score(y_test.values, pred)
                nums.append(r2)


            except Exception as e:
                print(e)
                continue 
            
        self.best_random_seed = max(nums)
        return (sum(nums)/len(nums))
        








            
    
