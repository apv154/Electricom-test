def function_creator(input_dir, output_dir)
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.ensemble import AdaBoostRegressor
  from sklearn.metrics import mean_squared_error, make_scorer, r2_score
  from sklearn.model_selection import GridSearchCV
  import xgboost as xgb

  elect = pd.read_csv(input_dir)
  elect = elect.drop(columns=['time'])
  elect = elect.dropna()
  X = elect.loc[:, elect.columns != 'Sub_metering_3']
  y = elect.loc[:, elect.columns == 'Sub_metering_3']
  xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.25)
  # Choose the type of classifier.
  abreg = AdaBoostRegressor()
  # Choose some parameter combinations to try
  params = {
   'n_estimators': [50, 100],
   'learning_rate' : [0.01, 0.05, 0.1, 0.5],
   'loss' : ['linear', 'square', 'exponential']
   }
  score = make_scorer(mean_squared_error)

  gridsearch=GridSearchCV(abreg, params, scoring=score, cv=5, return_train_score=True)
  gridsearch.fit(X, y)

  # Choose the type of classifier.
  abreg = AdaBoostRegressor()
  # Choose some parameter combinations to try
  params = {
   'n_estimators': [50, 100],
   'learning_rate' : [0.01, 0.05, 0.1, 0.5],
   'loss' : ['linear', 'square', 'exponential']
   }
  score = make_scorer(mean_squared_error)

  gridsearch=GridSearchCV(abreg, params, scoring=score, cv=5, return_train_score=True)
  gridsearch.fit(X, y)

  best_estim=gridsearch.best_estimator_

  best_estim.fit(xtrain,ytrain)
  ytr_pred=best_estim.predict(xtrain)
  mse = mean_squared_error(ytr_pred,ytrain)
  r2 = r2_score(ytr_pred,ytrain)

  print('The MSE error for the train is: ', mse)
  print('The correlation for the train is: ', r2)

  ypred=best_estim.predict(xtest)
  mse = mean_squared_error(ytest, ypred)
  r2 = r2_score(ytest, ypred)

  print('The MSE error for the test is: ', mse)
  print('The correlation for the test is: ', r2)

  plt.scatter(ytest, ypred)
  plt.show()

  xgb_reg = xgb.XGBRegressor(objective = 'reg:logistic')
  xgb_params = {
  'learning_rate': [0.01, 0.05, 0.1],
  'n_estimators' : [15, 30, 50, 80, 100, 150, 200],
  'max_depth': [3,6],
  'min_child_weight':[7],
  'gamma': [0.01, 0.05, 0.1, 0.5, 1.5, 10],
  'subsample':[ .8, .9]
  }
  grid_search = GridSearchCV(xgb_reg,xgb_params,cv=5,verbose=90,n_jobs=1)

  grid_search = grid_search.fit(x_train,y_train)

  xgboost_reg = grid_search.best_estimator_
  preds_xgb_reg_test = xgboost_reg.predict(x_test)
  
  plt.scatter(ytest, preds_xgb_reg_test)
  plt.show()
  
  mse = mean_squared_error(ytest, preds_xgb_reg_test)
  r2 = r2_score(ytest, preds_xgb_reg_test)
  print('The MSE error for the test is: ', mse)
  print('The correlation for the test is: ', r2)
